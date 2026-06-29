---
title: RAG Chatbot
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# RAG Chatbot

**🚀 Live demo: [huggingface.co/spaces/eraxes/rag-chatbot](https://huggingface.co/spaces/eraxes/rag-chatbot)**

A document question-answering web app built with **FastAPI**, **local embeddings**, **ChromaDB**, and a **provider-agnostic LLM layer** (Groq, Google Gemini, or Claude). Upload your documents (PDF / TXT / MD), then ask questions and get answers grounded in your content — **with cited sources** — through a clean web UI.

This is a Retrieval-Augmented Generation (RAG) system: instead of letting the LLM answer from memory (and hallucinate), it retrieves the most relevant chunks from your documents and instructs the model to answer **only** from that context.

---

## Features

- 💬 **Web UI** — drag-and-drop upload, chat with streaming-style feedback, source citations
- 📄 **Ingest** PDF, TXT, and Markdown files
- 🔍 **Semantic search** with local embeddings (free, no extra API key, multilingual TR/EN)
- 🤖 **Grounded answers** with inline citations (`[1]`, `[2]`) and expandable source cards
- 🔌 **Provider-agnostic LLM** — swap between Groq, Gemini, and Claude via one env var
- 🧪 **Evaluation harness** — scores retrieval recall, faithfulness & correctness (LLM-as-a-judge)
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
| LLM          | Groq / Gemini / Claude (default Groq) | Answer generation, swappable via `LLM_PROVIDER` |
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
│   ├── llm.py           # Provider-agnostic LLM layer (Groq / Gemini / Claude)
│   ├── rag.py           # Retrieval + generation
│   └── main.py          # FastAPI app, endpoints, serves the frontend
├── static/
│   └── index.html       # Single-page web UI
├── evals/
│   ├── dataset.json     # Golden Q&A test set (2 docs + distractor + refusals)
│   ├── metrics.py       # Deterministic metric (context recall)
│   ├── judge.py         # LLM-as-a-judge (faithfulness + correctness)
│   └── run_eval.py      # Eval runner (rate limit, retry, resume, report)
├── tests/
│   └── test_chunking.py # Unit tests for the chunker
├── examples/
│   ├── sirket_politikalari.txt         # Sample document (Acme)
│   └── beta_teknoloji_politikalari.txt # Distractor document (Beta)
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

# 3. Configure a provider (pick one — all have a free option)
cp .env.example .env
#   then open .env and set your key(s). See "Choosing a provider" below.

# 4. Run
./run.sh            # or: uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** for the web UI, or **/docs** for the interactive API.

### Choosing a provider

The LLM layer is provider-agnostic; pick one in `.env` via `LLM_PROVIDER`:

| Provider | `.env` keys | Free tier | Where |
|----------|-------------|-----------|-------|
| **Groq** (default) | `LLM_PROVIDER=groq` · `GROQ_API_KEY=gsk_...` | Generous daily limit | [console.groq.com](https://console.groq.com) |
| **Gemini** | `LLM_PROVIDER=gemini` · `GEMINI_API_KEY=...` | Low daily cap on some models | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| **Claude** | `LLM_PROVIDER=claude` · `ANTHROPIC_API_KEY=sk-ant-...` · `CHAT_MODEL=claude-opus-4-8` | Paid | [console.anthropic.com](https://console.anthropic.com) |

The eval **judge** is configured separately (`JUDGE_PROVIDER` / `JUDGE_MODEL`), so you can grade answers with a different / stronger model than the one that generates them.

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

## Evaluation (eval harness)

A RAG system that *seems* to work isn't enough — you have to **measure** it. This repo ships a small but real evaluation harness (`evals/`) that scores the pipeline against a golden dataset and acts as a quality gate.

### What it measures

| Metric | What it checks | How |
|--------|----------------|-----|
| **Context recall** | Did retrieval fetch the chunk that holds the answer? | Deterministic keyword check — no LLM, free, repeatable |
| **Faithfulness**  | Is the answer grounded in the retrieved context (no hallucination)? | LLM-as-a-judge |
| **Correctness**   | Does the answer match the reference (ground-truth) answer? | LLM-as-a-judge |

Two design choices worth calling out:

- **Judge ≠ generator.** The judge goes through the same provider-agnostic layer but points at a *different* model (`JUDGE_PROVIDER` / `JUDGE_MODEL`). Grading a model with itself invites *self-preference bias*.
- **Resilient & cheap.** Rate limiting, retry-with-backoff on transient/quota (`429`/`503`) errors, and a resumable cache mean a run that hits a free-tier limit picks up where it left off instead of starting over.

### Run it

```bash
python3 -m evals.run_eval          # resume from cache (skips completed questions)
python3 -m evals.run_eval --fresh  # run everything from scratch
```

You get a per-question pass/fail table (against thresholds) plus a detailed `evals/report.json` with every answer and the judge's reasoning.

### A real eval-driven fix

The dataset includes a **distractor document**: two companies (Acme & Beta) with the *same topics but different numbers*, and questions that force the system to pick the right one. That immediately caught a real bug — the generator answered Acme's training-budget question with **Beta's** figure, and leaked Beta's health-insurance policy into an Acme answer:

```
              recall  faith  correct
acme-egitim    1.00    0.00    0.00   ❌  gave Beta's 7500 ₺ instead of Acme's 10000 ₺
acme-saglik      —     0.00    0.00   ❌  claimed Acme has insurance (that's Beta's)
                                          → pass rate 80%
```

`recall = 1.00` while `faithfulness = 0` pinpointed the failure to **generation, not retrieval**. Root cause: chunking stripped the company name from later chunks, so the model couldn't tell the two documents apart. The fix was **contextual chunking** (`chunk_text_with_context` — prepend every chunk with its document title). Changing *only that* (same prompt, same model) took the score to:

```
ORTALAMA       1.00    0.98    1.00   ✅  pass rate 100%
```

That loop — **measure → find the weakness → fix → re-measure** — is the entire point of the harness.

---

## Possible improvements (roadmap)

- **Streaming responses** for the chat endpoint (token-by-token)
- **Re-ranking** retrieved chunks with a cross-encoder for better precision
- **Bigger eval set** — more documents, adversarial / multi-hop questions, a stronger judge model
- **Hybrid search** (keyword + semantic)
- **Authentication** and per-user document isolation
- Swap ChromaDB for **Qdrant / pgvector** for production scale

---

*The interesting parts to read first: [`app/rag.py`](app/rag.py) (retrieval + generation), [`app/ingestion.py`](app/ingestion.py) (contextual chunking), [`app/llm.py`](app/llm.py) (provider abstraction), and [`evals/run_eval.py`](evals/run_eval.py) (the eval harness).*
