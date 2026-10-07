"""FastAPI backend: LLM API integration (AI-1) + RAG (AI-3) + prompts (AI-2), served as a web app (AI-4)."""
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)  # .env wins locally; on Render there is no .env, so dashboard vars are used

import openai  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from prompts import build_messages  # noqa: E402
from rag import KnowledgeBase, client  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-oss-20b")
MIN_SCORE = 0.40  # below this, retrieval found nothing relevant -> skip the LLM call
STATIC_DIR = Path(__file__).parent / "static"

kb = KnowledgeBase()
startup_error = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global startup_error
    try:
        kb.load_or_build()
    except Exception as exc:  # keep the server up so /api/health explains the problem
        startup_error = str(exc)
        log.exception("Failed to build knowledge base")
    yield


app = FastAPI(title="Docs Assistant", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[dict] = Field(default_factory=list)


@app.get("/api/health")
def health():
    return {"ready": kb.ready, "chunks": len(kb.chunks), "error": startup_error}
        "key_length": len(key),
        "key_starts_with": key[:4],
        "model": CHAT_MODEL,
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not kb.ready:
        raise HTTPException(503, f"Knowledge base is not ready: {startup_error or 'still loading'}")

    question = req.message.strip()

    # Retrieval. Add the previous user turn so follow-ups like "and how long?" still retrieve well.
    prev_users = [t["content"] for t in req.history if t.get("role") == "user"]
    search_query = f"{prev_users[-1]} {question}" if prev_users else question
    try:
        passages = kb.search(search_query, k=4)
    except Exception as exc:
        log.exception("Retrieval failed")
        raise HTTPException(500, f"Search failed: {exc.__class__.__name__}")

    if not passages or passages[0]["score"] < MIN_SCORE:
        return {
            "answer": "I could not find anything about that in the documentation.",
            "answerable": False,
            "sources": [],
        }

    messages = build_messages(question, passages, req.history)
    try:
        resp = client().chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=1500,
            response_format={"type": "json_object"},  # structured output
            timeout=30,
        )
    except openai.RateLimitError:
        raise HTTPException(429, "Rate limit reached on the free AI plan. Wait a moment and try again.")
    except openai.AuthenticationError:
        raise HTTPException(500, "The server's GROQ_API_KEY is missing or invalid.")
    except openai.OpenAIError as exc:
        log.exception("Chat call failed")
        raise HTTPException(502, f"The AI request failed ({exc.__class__.__name__}). Please retry.")

    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"answer": raw, "answerable": True, "sources": []}

    used = data.get("sources") or []
    cited = []
    for n in used:
        if isinstance(n, int) and 1 <= n <= len(passages):
            p = passages[n - 1]
            cited.append({"n": n, "source": p["source"], "snippet": p["text"][:300]})

    return {
        "answer": data.get("answer", ""),
        "answerable": bool(data.get("answerable", True)),
        "sources": cited,
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")