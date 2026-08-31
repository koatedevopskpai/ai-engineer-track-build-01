# AI Engineer Track — Build 01 · Production-Style Agent Platform

A production-hardened customer-service agent: **agent orchestration, RAG, guardrails, evals, and
human-in-the-loop approval** — packaged for Docker with CI-ready structure.

This is the **flagship build** of the portfolio: it takes the working vertical slice from
[`ai-engineer-track`](https://github.com/koatedevopskpai/ai-engineer-track) and adds the layers a
real system needs before it touches production — **safety, verification, and cost awareness.**

---

## What it does

A support ticket flows through a LangGraph agent with a **human decision gate**:

1. **Triage** — classifies the ticket L1 / L2 / L3.
2. **Resolve** — retrieves grounded context (pgvector) and drafts an answer from the knowledge base.
3. **QA** — scores grounding confidence.
4. **HITL gate** — confident answers require explicit human **approval**; low-confidence answers
   are **escalated with full context** (never auto-sent).

Security and evaluation run alongside: **PII redaction, prompt-injection blocking, cost logging,
and RAGAS-based faithfulness evals** with an LLM judge.

---

## Architecture

```
              ┌────────────────────────────────────────────────────────┐
              │                    FastAPI (api/app.py)                │
              │                        /resolve  /health               │
              └───────────────────────────┬────────────────────────────┘
                                          ▼
  ┌─────────────┐   ┌────────────────┐   ┌────────────────┐   ┌──────────────────┐
  │   Triage    │──▶│  Resolve (RAG) │──▶│  QA (grounding)│──▶│  HITL gate       │
  └─────────────┘   └────────────────┘   └────────────────┘   └──────────────────┘
                        │  pgvector                                 │  approve / escalate
                        ▼                                          ▼
                guardrails/security.py                      context + draft handed to human
```

| Layer | Location | Role |
|---|---|---|
| **Agent graph** | `agent/graph.py` | LangGraph state machine with `MemorySaver` checkpointer |
| **HITL approval** | `agent/hitl.py` | Human decision gate — approve / block / escalate |
| **RAG pipeline** | `rag/pipeline.py` | Chunk, embed, retrieve (pgvector cosine) |
| **API** | `api/app.py` | FastAPI `/resolve` endpoint + health |
| **Guardrails** | `guardrails/security.py` | PII redaction, injection blocking, cost logging |
| **Evals** | `evals/` | RAGAS faithfulness + answer relevance; prompt A/B comparison |
| **Deployment** | `Dockerfile`, `docker-compose.yml` | Containerised API + pgvector |

---

## The eval layer (measured, not guessed)

`evals/evaluate.py` runs RAGAS metrics (faithfulness, answer relevancy) over a golden set, using a
separate **judge LLM** so scoring isn't self-graded.

Key measured finding (from the build log):

> **Model choice ≠ grounding quality.** Local `llama3.2` with default prompt achieved the best
> faithfulness (0.756) — *higher* than the larger cloud model under a strict prompt (0.644).
> Conclusion: keep generation local and spend the budget on retrieval/prompt quality instead.

This is the kind of decision the rest of the pipeline is built around: **local-first generation,
cloud judge, deterministic guardrails.**

---

## Quick start

### Prerequisites
- Python 3.12+ · [Ollama](https://ollama.com) with `llama3.2` pulled
- Optional: OpenAI-compatible API key for the judge LLM

### 1. Start the stack

```bash
docker compose up --build
# API on :8080, pgvector on :5432
```

### 2. Configure

```bash
cp env.example .env
# MODEL_PROVIDER=ollama  OLLAMA_MODEL=llama3.2  OPENCODE_API_KEY=  PGVECTOR_DSN=...
```

### 3. Run locally (no Docker)

```bash
pip install -r requirements.txt
python -m uvicorn api.app:app --reload --port 8080
```

### 4. Exercise the agent

```bash
curl -X POST localhost:8080/resolve -H "Content-Type: application/json" \
  -d '{"text": "How do I reset my password?"}'
# → status: SENT | BLOCKED | ESCALATE, with context + draft
```

### 5. Run the evals

```bash
python evals/evaluate.py       # RAGAS faithfulness + relevance on golden set
python evals/test_prompt.py    # A/B: grounding prompt vs baseline
```

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PROVIDER` | `ollama` | Generation backend |
| `OLLAMA_MODEL` | `llama3.2` | Local generation model |
| `OPENCODE_API_KEY` | *(empty)* | Judge LLM key — **never commit a real key** |
| `PGVECTOR_DSN` | `postgresql://ai:ai@localhost:5433/rag` | Vector store |

All secrets load from `.env` (git-ignored). `.env.example` ships empty.

---

## Portfolio context

| Build | Repo | Focus |
|---|---|---|
| 01 | [`ai-engineer-track`](https://github.com/koatedevopskpai/ai-engineer-track) | Working vertical slice |
| **02** | **this repo** | **Production hardening: guardrails, evals, HITL, Docker** |
| 03 | [`ai-engineer-track-build-02`](https://github.com/koatedevopskpai/ai-engineer-track-build-02) | Agentic workflow (n8n) + cost/ROI |

---

## What this demonstrates for an AI-engineering role
- **Agent orchestration** — LangGraph state machines with checkpoints and conditional routing.
- **RAG done properly** — pgvector retrieval, grounding prompts, and *measuring* faithfulness.
- **Production mindset** — human-in-the-loop gating, PII redaction, injection blocking, cost logging.
- **Evidence over opinion** — eval-driven model/prompt decisions, documented in the repo.
- **Delivery discipline** — Dockerised, config-driven, secrets-safe.

---

## License
MIT — free to use, learn from, and build on.

---

*Built as part of a personal AI engineering portfolio. Questions or feedback welcome via issues.*