"""
FastAPI server for the Ops RAG Assistant.

Serves the chat UI and a small JSON API:
    POST /api/upload   add documents (md / txt / pdf)
    POST /api/ask      ask a question -> grounded, cited answer + sources
    GET  /api/docs     list ingested documents
    POST /api/reset    clear the index

Optional access gate: set APP_PASSWORD to require it on every /api call.
"""

import os

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag_engine import RagEngine

app = FastAPI(title="Ops RAG Assistant")
engine = RagEngine()
FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")


def check_auth(x_app_password):
    expected = os.environ.get("APP_PASSWORD")
    if expected and x_app_password != expected:
        raise HTTPException(status_code=401, detail="Incorrect access password.")


class AskBody(BaseModel):
    question: str


@app.get("/")
def home():
    return FileResponse(FRONTEND)


@app.get("/api/health")
def health():
    return {"ok": True, "auth_required": bool(os.environ.get("APP_PASSWORD")),
            "answers_enabled": bool(os.environ.get("ANTHROPIC_API_KEY"))}


@app.get("/api/docs")
def docs(x_app_password: str = Header(None)):
    check_auth(x_app_password)
    return {"docs": engine.list_docs()}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...), x_app_password: str = Header(None)):
    check_auth(x_app_password)
    added = []
    for f in files:
        raw = await f.read()
        try:
            n = engine.add_document(f.filename, raw)
            added.append({"name": f.filename, "chunks": n})
        except Exception as e:
            added.append({"name": f.filename, "error": str(e)})
    return {"added": added, "docs": engine.list_docs()}


@app.post("/api/ask")
def ask(body: AskBody, x_app_password: str = Header(None)):
    check_auth(x_app_password)
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Empty question.")
    return engine.answer(body.question)


@app.post("/api/load-samples")
def load_samples(x_app_password: str = Header(None)):
    check_auth(x_app_password)
    import glob
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "sample_docs")
    added = []
    for path in sorted(glob.glob(os.path.join(sample_dir, "*.md"))):
        n = engine.add_document(path, open(path, "rb").read())
        added.append({"name": os.path.basename(path), "chunks": n})
    return {"added": added, "docs": engine.list_docs()}


@app.post("/api/reset")
def reset(x_app_password: str = Header(None)):
    check_auth(x_app_password)
    engine.reset()
    return {"ok": True}
