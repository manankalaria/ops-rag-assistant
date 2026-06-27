"""
RAG engine: ingest documents, embed + store them, retrieve, and answer
questions grounded ONLY in the retrieved passages (with citations, or an
honest "not covered" when the documents don't contain the answer).

The embedder is pluggable so the store/retrieval/answer logic can be tested
without downloading a model. The shipped default is fastembed (local, ONNX).
"""

import os
import re
import json
import pickle
import hashlib

import numpy as np

DATA_DIR = os.environ.get("RAG_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
ANSWER_MODEL = "claude-sonnet-4-6"
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------
class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed. Model downloads once on first use."""
    def __init__(self, model_name=EMBED_MODEL):
        self.model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(self.model_name)

    def embed(self, texts):
        self._ensure()
        vecs = np.array(list(self._model.embed(list(texts))), dtype="float32")
        return _normalize(vecs)


class StubEmbedder:
    """Deterministic hashing embedder — used for tests only (no model download)."""
    def __init__(self, dim=256):
        self.dim = dim

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return _normalize(out)


def _normalize(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_document(text, filename):
    """Markdown -> one chunk per `## Section`. Other text -> overlapping windows."""
    name = os.path.splitext(os.path.basename(filename))[0]
    if filename.lower().endswith(".md") and "##" in text:
        doc_title = _first_heading(text) or name
        chunks = []
        for section in re.split(r"\n(?=##\s)", text):
            m = re.match(r"##\s+(.*)", section.strip())
            if not m:
                continue
            body = section.split("\n", 1)[1].strip() if "\n" in section else ""
            if body:
                chunks.append({
                    "label": f"{_short(doc_title)} §{m.group(1).strip()}",
                    "doc": doc_title, "title": m.group(1).strip(),
                    "text": body, "source": filename,
                })
        if chunks:
            return chunks
    return _window_chunks(text, name, filename)


def _window_chunks(text, name, filename, size=900, overlap=150):
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    label_name = name[:18]
    chunks, start, idx = [], 0, 1
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append({
                "label": f"{label_name} #{idx}", "doc": name,
                "title": f"part {idx}", "text": piece, "source": filename,
            })
            idx += 1
        start += size - overlap
    return chunks


def _first_heading(text):
    m = re.search(r"^#\s+(.*)", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _short(title):
    head = title.split("&")[0].split()
    if len(head) == 1:
        return re.sub(r"[^A-Za-z]", "", head[0]).upper()[:3] or "DOC"
    return "".join(w[0] for w in head if w[0].isalpha()).upper()[:4] or "DOC"


def extract_text(filename, raw_bytes):
    """Return plain text from .md/.txt/.pdf bytes."""
    if filename.lower().endswith(".pdf"):
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw_bytes))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    return raw_bytes.decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
class RagEngine:
    def __init__(self, embedder=None, data_dir=DATA_DIR):
        self.embedder = embedder or FastEmbedEmbedder()
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.meta = []                 # list of chunk dicts
        self.matrix = None             # np.ndarray [N, dim]
        self._load()

    # ---- ingest ----
    def add_document(self, filename, raw_bytes):
        text = extract_text(filename, raw_bytes)
        chunks = chunk_document(text, filename)
        if not chunks:
            return 0
        vecs = self.embedder.embed([f"{c['title']}. {c['text']}" for c in chunks])
        self.matrix = vecs if self.matrix is None else np.vstack([self.matrix, vecs])
        self.meta.extend(chunks)
        self._save()
        return len(chunks)

    # ---- retrieve ----
    def search(self, query, k=4):
        if self.matrix is None or not len(self.meta):
            return []
        q = self.embedder.embed([query])[0]
        scores = self.matrix @ q
        order = np.argsort(scores)[::-1][:k]
        return [{**self.meta[i], "score": float(scores[i])} for i in order]

    # ---- answer ----
    def answer(self, query, k=4):
        hits = self.search(query, k=k)
        sources = [{"label": h["label"], "title": h["title"],
                    "text": h["text"], "score": round(h["score"], 3)} for h in hits]
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return {"grounded": None, "answer": "Set ANTHROPIC_API_KEY on the server to enable answers.",
                    "citations": [], "sources": sources}
        if not hits:
            return {"grounded": False, "answer": "No documents have been added yet.",
                    "citations": [], "sources": []}
        result = self._generate(query, hits)
        result["sources"] = sources
        return result

    def _generate(self, query, hits):
        import anthropic
        context = "\n\n".join(f"[{h['label']}] {h['title']}\n{h['text']}" for h in hits)
        prompt = (
            "You are an assistant answering questions about an organization's documents. "
            "Use ONLY the passages below. Do not use outside knowledge.\n\n"
            "If the passages answer the question, write a clear 2-4 sentence answer and list "
            "the exact source labels used. If they do NOT, set grounded to false and say the "
            "documents don't cover it. Never guess.\n\n"
            f"Passages:\n{context}\n\nQuestion: {query}\n\n"
            'Respond with ONLY JSON: {"grounded": bool, "answer": str, "citations": [labels]}'
        )
        msg = anthropic.Anthropic().messages.create(
            model=ANSWER_MODEL, max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"grounded": True, "answer": text, "citations": [h["label"] for h in hits]}

    # ---- docs list / reset ----
    def list_docs(self):
        counts = {}
        for c in self.meta:
            counts[c["source"]] = counts.get(c["source"], 0) + 1
        return [{"name": os.path.basename(s), "chunks": n} for s, n in counts.items()]

    def reset(self):
        self.meta, self.matrix = [], None
        for f in ("meta.json", "matrix.npy"):
            p = os.path.join(self.data_dir, f)
            if os.path.exists(p):
                os.remove(p)

    # ---- persistence ----
    def _save(self):
        with open(os.path.join(self.data_dir, "meta.json"), "w") as f:
            json.dump(self.meta, f)
        if self.matrix is not None:
            np.save(os.path.join(self.data_dir, "matrix.npy"), self.matrix)

    def _load(self):
        mp = os.path.join(self.data_dir, "meta.json")
        xp = os.path.join(self.data_dir, "matrix.npy")
        if os.path.exists(mp) and os.path.exists(xp):
            self.meta = json.load(open(mp))
            self.matrix = np.load(xp)
