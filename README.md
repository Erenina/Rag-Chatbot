# RAG Chatbot

A document question-answering web app built with **FastAPI**, **local embeddings**, **ChromaDB**, and a **provider-agnostic LLM layer** (Google Gemini by default, Claude optional). Upload your documents (PDF / TXT / MD), then ask questions and get answers grounded in your content — **with cited sources** — through a clean web UI.

This is a Retrieval-Augmented Generation (RAG) system: instead of letting the LLM answer from memory (and hallucinate), it retrieves the most relevant chunks from your documents and instructs the model to answer **only** from that context.

---

## Features

- 💬 **Web UI** — drag-and-drop upload, chat with streaming-style feedback, source citations
- 📄 **Ingest** PDF, TXT, and Markdown files
- 🔍 **Semantic search** with local embeddings (free, no extra API key, multilingual TR/EN)
- 🤖 **Grounded answers** with inline citations (`[1]`, `[2]`) and expandable source cards
- 🔌 **Provider-agnostic LLM** — swap between Gemini (free tier) and Claude via one env var
- 🗂️ **Persistent vector store** (ChromaDB) — data survives restarts
- ⚡ **Auto-generated API docs** at `/docs` (Swagger UI)
- ✅ **Clean error handling** (missing key → `401`, rate limit → `429`/`502`) and unit tests

---

## How RAG works here

```
INGESTION (once per document)
  file ──▶ extract text ──▶ chunk ──▶ embed (vector) ──▶ store in ChromaDB

QUERY (every question)
  question ──▶ embed ──▶ find nearest chunks ──▶ feed to LLM as context ──▶ cited answer
```

The core idea: tell the LLM *"don't make things up — answer based on these retrieved passages."* This reduces hallucination and makes answers auditable.

---

## Tech stack

| Layer        | Choice                                   | Why                                                |
|--------------|------------------------------------------|----------------------------------------------------|
| Frontend     | HTML + Tailwind (CDN) + vanilla JS       | Single page, zero build, served by FastAPI         |
| API          | FastAPI + Uvicorn                        | Modern, fast, automatic OpenAPI docs               |
| Embeddings   | `intfloat/multilingual-e5-small` (local) | Free, runs offline, supports Turkish + English     |
| Vector DB    | ChromaDB                                 | Local, persistent, stores metadata for citations   |
| LLM          | Gemini `gemini-2.5-flash` (default) / Claude | Answer generation, swappable via `LLM_PROVIDER` |
| PDF parsing  | pypdf                                     | Text extraction                                    |

---

## Project structure

```
rag-chatbot/
├── app/
│   ├── config.py        # Settings (loaded from .env)
│   ├── models.py        # Pydantic request/response schemas
│   ├── ingestion.py     # Text extraction + chunking
│   ├── embeddings.py    # Local embedding model wrapper
│   ├── vectorstore.py   # ChromaDB wrapper (add / query / delete)
│   ├── llm.py           # Provider-agnostic LLM layer (Gemini / Claude)
│   ├── rag.py           # Retrieval + generation
│   └── main.py          # FastAPI app, endpoints, serves the frontend
├── static/
│   └── index.html       # Single-page web UI
├── tests/
│   └── test_chunking.py # Unit tests for the chunker
├── examples/
│   └── sirket_politikalari.txt   # Sample document to try
├── requirements.txt
├── .env.example
└── run.sh
```

---

## Setup

```bash
# 1. (Recommended) create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
#   then edit .env and set GEMINI_API_KEY=...
#   get a free key (no credit card) at https://aistudio.google.com/apikey

# 4. Run
./run.sh            # or: uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** for the web UI, or **/docs** for the interactive API.

### Using Claude instead of Gemini

Set these in `.env`:

```
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-...
CHAT_MODEL=claude-opus-4-8
```

---

## API

| Method | Path         | Description                          |
|--------|--------------|--------------------------------------|
| GET    | `/`          | Web UI                               |
| POST   | `/ingest`    | Upload a document (multipart `file`) |
| POST   | `/chat`      | Ask a question → answer + sources    |
| GET    | `/documents` | List ingested documents              |
| DELETE | `/documents` | Clear all documents                  |
| GET    | `/health`    | Health check + chunk count           |

**Example:**
```bash
curl -X POST http://localhost:8000/ingest -F "file=@examples/sirket_politikalari.txt"
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How many vacation days do employees get?"}'
```

---

## Tests

```bash
pytest -q
```

---

## Possible improvements (roadmap)

- **Streaming responses** for the chat endpoint (token-by-token)
- **Re-ranking** retrieved chunks with a cross-encoder for better precision
- **Evaluation harness** to measure answer quality (faithfulness, relevance)
- **Hybrid search** (keyword + semantic)
- **Authentication** and per-user document isolation
- Swap ChromaDB for **Qdrant / pgvector** for production scale

---

*The interesting parts to read first: [`app/rag.py`](app/rag.py) (retrieval + generation), [`app/ingestion.py`](app/ingestion.py) (chunking strategy), and [`app/llm.py`](app/llm.py) (provider abstraction).*
