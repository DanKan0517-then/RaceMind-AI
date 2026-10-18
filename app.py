"""RaceMind AI web app: FastAPI backend + static chat page."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import rag

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("racemind")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="RaceMind AI", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class Source(BaseModel):
    page: str | None
    source: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.on_event("startup")
def warm_up():
    # Load the embedding model and vector store once, so the first question isn't slow.
    try:
        rag.get_retriever()
        log.info("Retriever ready (index at %s)", rag.CHROMA_PATH)
    except Exception:
        log.exception("Failed to load retriever")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_present": Path(rag.CHROMA_PATH).exists(),
        "groq_key_set": bool(os.getenv("GROQ_API_KEY")),
    }


@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    question = req.question.strip()
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(503, "Server is missing GROQ_API_KEY. Add it as a secret.")
    try:
        return await run_in_threadpool(rag.ask, question)
    except Exception as exc:
        log.exception("ask failed")
        raise HTTPException(500, f"Something went wrong: {exc}") from exc
