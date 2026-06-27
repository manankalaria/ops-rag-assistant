# Ops RAG Assistant

**A self-hostable app that lets a team chat with their own documents — handbooks, SOPs, policies, contracts — and get answers grounded in the source, with citations.** Upload documents, ask questions in plain English, get an answer tied to the exact passage it came from. When the documents don't contain the answer, it says so instead of guessing.

This is a real, runnable web application, not a demo: a Python backend that holds the API key, ingests documents, builds a vector index, and serves a chat UI. Deploy one private instance per client, on their documents.

## Screenshots


*A cited, grounded answer:*

<img width="1000" height="917" alt="S1" src="https://github.com/user-attachments/assets/fa9b80e1-5a92-41c0-acd8-1eb7b9e9a9d3" />

<img width="990" height="742" alt="S2" src="https://github.com/user-attachments/assets/bb0eb470-7ab7-4c9e-8bb6-21d72ad4361d" />



---

## What it does

- **Upload your documents** — Markdown, plain text, or PDF
- **Semantic retrieval** — questions are matched to passages by meaning, so "can I work overseas?" finds the "working abroad" policy even without the exact words
- **Grounded, cited answers** — every answer is constrained to the retrieved passages and shows its sources
- **Honest refusals** — if the documents don't cover the question, it tells you, rather than inventing an answer
- **Optional access password** — protect a deployed instance so only the client can use it (and your API key)

---

## Quickstart (local)

```bash
git clone <your-repo-url> && cd ops-rag-app
cp .env.example .env          # add your ANTHROPIC_API_KEY
./run.sh
```

Open http://localhost:8000, click **Load sample handbook** (or upload your own files), and start asking. The embedding model (~90 MB) downloads automatically on first use.

No key yet? The app still runs and shows retrieval — it just asks you to set a key before it writes answers.

---

## Deploy (one instance per client)

It's a standard FastAPI app, so anywhere that runs Python or Docker works.

**Docker:**
```bash
docker build -t ops-rag .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... -e APP_PASSWORD=clientsecret ops-rag
```

**Render / Railway / Fly.io:** point the platform at this repo, set `ANTHROPIC_API_KEY` and `APP_PASSWORD` as environment variables, and use the start command:
```
cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT
```

Always set `APP_PASSWORD` on a public deployment — without it, anyone with the URL can spend your API budget.

---

## How it works

```
  upload (md/txt/pdf) ─► extract text ─► chunk ─► embed ─► vector index (persisted)
                                                              │
  question ─► embed ─► cosine similarity ─► top-k passages ───┘
                                  │
                  prompt grounded ONLY in those passages
                                  │
                    cited answer  ─or─  "not in your documents"
```

- **Embeddings:** `fastembed` with BAAI/bge-small-en — runs locally, no embedding API or GPU
- **Index:** an in-memory NumPy matrix persisted to `data/`; fine for thousands of chunks
- **Answers:** Claude, instructed to use only the retrieved passages and to refuse when they don't cover the question
- **Backend:** FastAPI · **Frontend:** a single static page served by the backend (no build step)

---

## Selling this

The realistic model for small clients is **deploy-per-client**: stand up a private instance on their documents, set an access password, hand them a URL. A typical engagement is a setup fee plus a small monthly fee for hosting and updates. Because each client's data lives only in their own instance, there's no shared-data or multi-tenant complexity to explain.

Good first clients: small firms with a thick handbook or knowledge base, an ops team drowning in "where's the policy on…" questions, or a support team that wants answers grounded in their own docs.

---

## Scope and honest limitations

This is a clean, single-tenant foundation — deliberately not a multi-tenant SaaS. For larger or higher-stakes deployments, the natural next steps are: a real vector database (FAISS, Chroma, pgvector) as the corpus grows past a few thousand chunks; per-user accounts and logging; richer ingestion (Word, HTML, Confluence, shared drives); and evaluation on a client's real question set before launch. Each is a defined add-on rather than a rewrite.

---

Built by **Manan Kalaria** — AI consultant and agent developer with a finance and strategy background. I build RAG assistants, AI agents, and automation for teams.

- [github.com/manankalaria](https://github.com/manankalaria) · manankalaria97@gmail.com
