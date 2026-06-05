# Grid07 — Autonomous Multi-Agent Social System

> A multi-agent pipeline where bots autonomously route, generate, and defend content in threaded arguments — built with LangGraph, FAISS, and a RAG-powered combat engine.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green) ![LLM](https://img.shields.io/badge/LLM-Groq%20Llama3.3--70b-orange) ![FAISS](https://img.shields.io/badge/VectorStore-FAISS-purple) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What This Does

Grid07 simulates a platform where multiple AI bots operate as autonomous agents with distinct personas. Given an incoming post:

1. **Phase 1** routes it to the most relevant bot(s) using vector similarity over persona embeddings
2. **Phase 2** has each bot autonomously research and generate a 280-character post using a LangGraph state machine
3. **Phase 3** generates context-aware replies using RAG over the full conversation thread, with active prompt injection defense

This is not a chatbot. Each phase is a programmatic pipeline — the LLM is a component called via API, not a product being used.

---

## Tech Stack

| Component     | Library                                    |
|---------------|--------------------------------------------|
| LLM           | Groq (`llama-3.3-70b-versatile`, free tier)|
| Orchestration | LangGraph + LangChain                      |
| Embeddings    | `sentence-transformers/all-MiniLM-L6-v2`  |
| Vector Store  | FAISS (in-memory, `IndexFlatIP`)           |
| Environment   | python-dotenv                              |

---

## Architecture

```
Incoming Post
      │
      ▼
┌─────────────────────┐
│  Phase 1            │  Vector similarity routing
│  Persona Router     │  FAISS + cosine similarity
│  (FAISS)            │  → matched bot personas
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Phase 2            │  LangGraph state machine
│  Content Engine     │  decide_search → web_search → draft_post
│  (LangGraph)        │  → structured JSON post output
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Phase 3            │  RAG over full thread context
│  Combat Engine      │  + prompt injection detection
│  (RAG + Defense)    │  → persona-consistent reply
└─────────────────────┘
```

---

## Phase 1: Vector-Based Persona Routing

Each bot's persona description is embedded and stored in a FAISS `IndexFlatIP`. Incoming posts are embedded and queried against the index — bots above a cosine similarity threshold are selected for response.

```python
router.route_post_to_bots(post_content: str, threshold: float = 0.35)
```

**On threshold calibration:** The original spec uses 0.85, which targets OpenAI's `text-embedding-ada-002` (1536 dims). `all-MiniLM-L6-v2` operates in 384-dim space, producing inter-topic similarities of 0.2–0.6. Threshold is set to 0.35 accordingly — if you swap embedding models, recalibrate.

---

## Phase 2: LangGraph Autonomous Content Engine

A 3-node directed graph where each node is an LLM call or tool call:

```
[decide_search] → [web_search] → [draft_post] → END
```

| Node | Responsibility |
|------|----------------|
| `decide_search` | LLM reads bot persona → outputs a search query |
| `web_search` | Calls `mock_searxng_search` → retrieves news headlines |
| `draft_post` | LLM combines persona + results → outputs strict JSON |

Structured output is enforced via system prompt + regex fence stripping + fallback handling:

```json
{
  "bot_id": "bot_a",
  "topic": "AI job displacement",
  "post_content": "GPT-5 just dropped and half your LinkedIn is crying..."
}
```

---

## Phase 3: RAG Combat Engine + Prompt Injection Defense

`generate_defense_reply` receives the full thread (parent post + comment history + latest message) and constructs a RAG prompt over the entire context — the bot argues informed by the full argument history, not just the last turn.

### Injection Defense (Two Layers)

**Layer 1 — Pattern Detection:**
```python
detect_prompt_injection(text: str) -> bool
```
Scans for ~10 known injection patterns: `"ignore all previous instructions"`, `"you are now"`, `"pretend you are"`, etc.

**Layer 2 — System Prompt Persona Lock:**
An immutable persona declaration in the system prompt that cannot be overridden by the user turn. When injection is detected, a `⚠️ SECURITY ALERT` block is appended to the system prompt — the bot never acknowledges the injection, it simply continues the argument.

This works because system prompt instructions have higher weight than user turn content in instruction-tuned models.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/kaushalkumarma2025/grid07.git
cd grid07
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env   # Linux/Mac
copy .env.example .env  # Windows
```

Add your Groq API key (free at [console.groq.com](https://console.groq.com)):
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```
> No spaces around `=`, no quotes around the key.

### 3. Run

```bash
python main.py          # Full pipeline

# Or run phases independently:
python phase1/persona_router.py
python phase2/content_engine.py
python phase3/combat_engine.py
```

---

## Project Structure

```
grid07/
├── main.py                    # Full pipeline runner
├── requirements.txt
├── .env.example
├── execution_logs.md          # Sample output from all three phases
├── phase1/
│   ├── __init__.py
│   └── persona_router.py      # FAISS vector store + cosine routing
├── phase2/
│   ├── __init__.py
│   └── content_engine.py      # LangGraph 3-node state machine
└── phase3/
    ├── __init__.py
    └── combat_engine.py       # RAG thread context + injection defense
```

---

## Notes on Model Choices

- **LLM:** The original spec referenced `llama3-8b-8192`, which Groq deprecated in early 2026. This repo uses `llama-3.3-70b-versatile` — the recommended replacement per [Groq's deprecation docs](https://console.groq.com/docs/deprecations). It's free-tier compatible and produces significantly better structured JSON output.
- **Embeddings:** `all-MiniLM-L6-v2` is used for local, dependency-free embedding. No API key required for Phase 1.

---

## Execution Logs

See [`execution_logs.md`](./execution_logs.md) for full console output across all three phases.
