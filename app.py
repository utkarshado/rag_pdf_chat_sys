from __future__ import annotations

import contextlib
import io
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import google.generativeai as genai

load_dotenv()

# Suppress the debug prints in chat_pdf.py during import, without changing that file.
with contextlib.redirect_stdout(io.StringIO()):
    import chat_pdf as core  # type: ignore


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Chat with PDF")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class ChatRequest(BaseModel):
    session_id: str
    question: str


@dataclass
class SessionState:
    current_pdf_path: Optional[str] = None
    current_pdf_name: Optional[str] = None
    history: "WebChatHistory" = field(default_factory=lambda: WebChatHistory())


class WebChatHistory:
    """Lightweight per-session history for the web UI.

    This intentionally avoids chat_pdf.py's file-backed global history so that
    multiple browser sessions don't overwrite each other.
    """

    def __init__(self) -> None:
        self.turns: List[dict] = []

    def add(self, question: str, answer: str) -> None:
        self.turns.append({"question": question, "answer": answer})
        if len(self.turns) > core.MAX_HISTORY:
            self.turns = self.turns[-core.MAX_HISTORY :]

    def clear(self) -> None:
        self.turns = []

    def is_empty(self) -> bool:
        return len(self.turns) == 0

    def summary(self) -> str:
        return f"{len(self.turns)} turn(s) in memory"

    def format_for_prompt(self) -> str:
        if not self.turns:
            return "(No previous conversation)"
        lines = []
        for i, turn in enumerate(self.turns, 1):
            answer = turn["answer"]
            truncated = (answer[:400] + "…") if len(answer) > 400 else answer
            lines.append(f"[Turn {i}]")
            lines.append(f"  Q: {turn['question']}")
            lines.append(f"  A: {truncated}")
        return "\n".join(lines)


_SESSIONS: Dict[str, SessionState] = {}
_LLMBACKEND = {"llm": None, "collection": None}


def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    name = name.replace(" ", "_")
    safe = "".join(ch if ch.isalnum() or ch in "._-()" else "_" for ch in name)
    return safe or "document.pdf"


def get_session(session_id: str) -> SessionState:
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = SessionState()
    return _SESSIONS[session_id]


def get_llm_and_collection():
    """Lazily initialize Gemini + Chroma only when the first request arrives."""
    if _LLMBACKEND["llm"] is not None and _LLMBACKEND["collection"] is not None:
        return _LLMBACKEND["llm"], _LLMBACKEND["collection"]

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set in the environment.",
        )

    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(core.LLM_MODEL)
    collection = core.get_collection()

    _LLMBACKEND["llm"] = llm
    _LLMBACKEND["collection"] = collection
    return llm, collection


def retrieve_chunks_for_pdf(collection, question: str, pdf_source: str) -> list[dict]:
    results = collection.query(
        query_texts=[question],
        n_results=core.TOP_K,
        include=["documents", "metadatas"],
        where={"source": pdf_source},
    )

    documents = results.get("documents", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []

    return [
        {"text": doc, "source": meta.get("source", pdf_source)}
        for doc, meta in zip(documents, metadatas)
    ]


@app.get("/")
def index():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=500, detail="frontend/index.html is missing")
    return FileResponse(index_file)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_pdf(
    session_id: str = Form(...),
    pdf: UploadFile = File(...),
):
    if not pdf.filename:
        raise HTTPException(status_code=400, detail="No file name provided")

    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    session = get_session(session_id)
    llm, collection = get_llm_and_collection()

    safe_name = _sanitize_filename(pdf.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    saved_path = session_dir / unique_name

    try:
        with saved_path.open("wb") as buffer:
            shutil.copyfileobj(pdf.file, buffer)
    finally:
        await pdf.close()

    try:
        indexed = core.index_pdf(collection, str(saved_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to index PDF: {exc}") from exc

    session.current_pdf_path = str(saved_path)
    session.current_pdf_name = saved_path.name

    return {
        "message": "PDF uploaded successfully",
        "indexed": indexed,
        "pdf_name": saved_path.name,
        "session_id": session_id,
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    session = get_session(request.session_id)
    if not session.current_pdf_name:
        raise HTTPException(status_code=400, detail="Upload a PDF before asking questions")

    llm, collection = get_llm_and_collection()

    history = session.history
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        standalone = question
        if not history.is_empty():
            standalone = core.rewrite_query_if_needed(question, history, llm).strip() or question

        chunks = retrieve_chunks_for_pdf(collection, standalone, session.current_pdf_name)

        if chunks:
            answer, sources = core.generate_answer(question, chunks, history, llm)
        else:
            answer = "I couldn't find that in the indexed documents."
            sources = [session.current_pdf_name]

        history.add(question, answer)

        return {
            "answer": answer,
            "sources": sources,
            "rewritten_question": standalone,
            "current_pdf": session.current_pdf_name,
            "history_turns": len(history.turns),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc


@app.post("/api/reset")
def reset_session(session_id: str = Form(...)):
    session = get_session(session_id)
    session.history.clear()
    session.current_pdf_path = None
    session.current_pdf_name = None
    return {"message": "Session cleared", "session_id": session_id}


@app.get("/api/session/{session_id}")
def session_status(session_id: str):
    session = get_session(session_id)
    return {
        "session_id": session_id,
        "current_pdf": session.current_pdf_name,
        "history_turns": len(session.history.turns),
    }
