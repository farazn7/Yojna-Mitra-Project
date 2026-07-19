# Yojana Mitra

Yojana Mitra is an autonomous, multi-user AI agent deployed on Discord that discovers Indian government welfare schemes a citizen qualifies for and guides them through eligibility evaluation, document collection, and live web form submission.

The system handles natural language in Hindi, English, and Hinglish. It is built specifically for users who face language, literacy, and technical barriers when interacting with government portals.

---

## Architecture & Tech Stack

| Layer | Technology |
|---|---|
| **Core Language** | Python 3.13 |
| **Interface** | Discord (DM-based gateway, async multi-user) |
| **Agentic Framework** | LangGraph (`StateGraph`, `PostgresSaver`, Conditional Edge Routing) |
| **Local Inference** | Ollama — Qwen3-8B (reasoning/recommendation), Llama 3.1 (routing/structuring), Gemma3:4B (vision/document classification), nomic-embed-text (embeddings) |
| **Vector Store & Persistence** | PostgreSQL + pgvector (Docker container) |
| **Checkpointing** | LangGraph `PostgresSaver` backed by `psycopg_pool` |
| **Web Automation** | Playwright (Sync API, isolated via ThreadPoolExecutor) |
| **Data Scraping** | Playwright-based crawler targeting myScheme.gov.in |
| **Testing** | pytest (parametrized persona matrix), custom integration harness |

---

## Engineering Milestones

### Phase 1 — Data Engineering & Extraction
Built a resilient Playwright-based scraper to crawl myScheme.gov.in, which runs on a Next.js/React architecture with dynamic DOM hydration.

- **React Hydration Bypassing:** Used a green button DOM-polling strategy so pagination only triggered after React fully painted new elements, preventing stale-element duplication.
- **Idempotency & Crash Recovery:** Two-file architecture (`schemes.json` + `progress.json`) saves state every 10 extractions, allowing seamless resume without duplication.
- **Hidden DOM Extraction:** Targeted parent container IDs (`#eligibility`, `#documents-required`, `#sources`) directly instead of interacting with unstable visible UI tabs.
- **Application Mode Classification:** Parsed the `#sources` section of each scheme page to classify each scheme's application mode: `online`, `pdf_form`, or `physical_only`.

**Result:** 230+ unique Tamil Nadu scheme records with eligibility criteria, required documents, portal URLs, and application mode stored in PostgreSQL.

---

### Phase 2 — Hybrid RAG Architecture & Persona Evaluation
Replaced a legacy regex keyword matcher with a multi-tiered Hybrid RAG pipeline:

1. **Hard Filter Layer:** SQL `WHERE` clauses instantly eliminate schemes outside strict numerical bounds (age, income, gender, disability status).
2. **Semantic Ranking Layer:** pgvector cosine similarity (`<=>`) ranks remaining candidates by semantic closeness to the user's query.
3. **Hybrid Re-scoring:** Post-retrieval scores combine 70% vector similarity + 30% string overlap (`difflib.SequenceMatcher`) for final ranking.
4. **Automated Persona Testing:** Built `test_hybrid_rag.py` parametrized across 10 citizen personas (e.g., `rural_student`, `widowed_farmer`) to validate the full retrieval and generation pipeline.

---

### Phase 3 — Conversational State Machine & Live Runtime Routing
Turned the static testing harness into a live, interactive Discord interface.

- **13-Step Onboarding:** Progressive state machine sequentially captures and validates comprehensive user profiles (gender, caste, income, minority status, BPL, etc.).
- **JSONB Profile Storage:** Replaced rigid boolean database columns with flexible JSONB + atomic append operations.
- **Discord Message Chunking:** Splits LLM outputs at newline boundaries for payloads exceeding 1,900 characters to avoid Discord's 2,000-character API limit.
- **Live RAG Context Injection:** Pulls the user's verified profile from PostgreSQL at runtime using their Discord Snowflake ID and injects it into every LLM prompt.

---

### Phase 4 — Thread-Safe Async Multi-Threading & Intent Routing
- **Non-Blocking Execution:** Dispatches heavy LLM inference and pgvector queries to isolated background threads via `asyncio.to_thread()`, preventing Discord Gateway heartbeat blocks.
- **Intent Classifier Middleware:** A lightweight Llama 3.1 routing layer classifies every message as `SCHEME_QUERY`, `PROFILE_UPDATE`, `CHIT_CHAT`, or `APPLY_SCHEME` with temperature 0.0. Defaults to `CHIT_CHAT` on ambiguous input to avoid triggering the full RAG pipeline on casual messages.

---

### Phase 5 — Agentic Workflow & LangGraph State Machine
Re-architected the entire runtime into a LangGraph `StateGraph` with persistent memory:

- **Stateful Graph Architecture:** `ConversationState` TypedDict with LangGraph message reducers (`add_messages`, `RemoveMessage`) for sliding conversation history.
- **PostgreSQL Checkpointer:** `PostgresSaver` backed by `psycopg_pool` (`autocommit=True`) persisting conversation threads keyed by Discord Snowflake IDs.
- **Multi-Intent Routing:** Conditional edges routing across `SCHEME_QUERY`, `PROFILE_UPDATE`, `CHIT_CHAT`, `DOC_RECEIVED`, and `APPLY_SCHEME` intents.
- **Thought Extraction Middleware:** Custom regex middleware strips Qwen3's `<think>` reasoning blocks before sending output to Discord, with empty-payload guardrails preventing Discord API `50006` errors.
- **Short-Term Memory Summarizer:** Auto-triggered when conversation history exceeds 8 messages — summarizes prior context to maintain VRAM efficiency while preserving continuity.

---

### Phase 6 — Document Intelligence Pipeline & PII Vault
Full document handling pipeline for Aadhaar, PAN, Voter ID, College IDs, and certificates uploaded directly via Discord DM.

- **3-Pass Vision LLM Pipeline (`document_extractor.py`):**
  - **Pass 0 — Classification:** Gemma3:4B identifies document type (`aadhaar`, `pan_card`, `voter_id`, `college_id`, `back_of_card`, etc.) in ~1 second.
  - **Pass 1 — Targeted Extraction:** Document-specific prompts extract only cardholder PII (name, DOB, address split into components) using Gemma3 vision.
  - **Pass 2 — Structuring & Validation:** Llama 3.1 formats raw VLM output into clean JSON. Regex validators confirm Aadhaar (12 digits), PAN (`AAAAA9999A`), and pincode (6 digits) formats.
- **PII Vault (`pii_vault` table):** Physically separated from `user_profiles`. Stores document-verified sensitive data with a 24-hour TTL, deep-merge UPSERT semantics, ghost-filter expiry (`WHERE expires_at > NOW()`), and a garbage collector (`purge_expired_vault()`).
- **Document Cache System:** Before any AI inference, checks `pii_vault` for existing valid data of the exact document type. Cache hits skip VLM entirely.
- **Security Bouncer (`document_handler.py`):** Three-layer security — pre-download size/extension validation, magic byte integrity check, and on-demand EXIF stripping + JPEG compression for government portal submission.

---

### Phase 7 & 7.1 — HITL Gating, Corrective RAG & Fuzzy Interceptor
- **HITL Document Checklist Gating:** Stateful document collection loop (`request_next_document` in `graph.py`) driven by `db.get_scheme_documents_needed`. Separates scannable files (`aadhaar`, `income_certificate`) from offline manual prerequisites. Generates a `user_manifest.md` audit trail under `temp_documents/{user_id}/`.
- **Corrective Hybrid RAG:** Upgraded retrieval to Top-7 hybrid search combining pgvector cosine distance and keyword string overlap via `difflib.SequenceMatcher`. Replaced unstructured generation with a structured 3-tier internal reasoning framework (Exact Match → Near-Miss → No Match) that shapes the LLM's evaluation logic without exposing tier labels to the user.
- **Fuzzy Trigram Application Interceptor (`db.find_similar_scheme_name`):** When a user says "apply for Moovalur Ramamirtham Higher Education", executes a sub-1-second PostgreSQL trigram match (`pg_trgm`) and prompts a direct confirmation instead of triggering a full RAG search.
- **Pre-Playwright Mandatory Enforcement Gate:** Hard security gate before any web automation. If mandatory documents were skipped, the system intercepts and enforces upload with double-check cache validation.

---

### Phase 8 — Universal Web Automation & ReAct Agent
Playwright-driven autonomous form filler that navigates government portals without any hardcoded selectors.

- **Persistent Session Isolation (`browser_manager.py`):** Each user gets a dedicated browser profile directory (`browser_profiles/{user_id}/`) retaining cookies and session tokens across automation turns, enabling multi-day application workflows.
- **Async Thread Isolation Boundary (`_PW_EXECUTOR`):** All Playwright `sync_api` calls are routed through a dedicated `ThreadPoolExecutor` (`PW_Sync_Worker`) isolated from Discord's `asyncio` event loop, eliminating the `Playwright Sync API inside asyncio loop` crash.
- **Application Mode Fast-Track:**
  - `online` → Launches ReAct automation agent.
  - `pdf_form` → Instantly returns download link with print & submit instructions. No Playwright invoked.
  - `physical_only` → Returns office address and guidelines.
- **Perception Engine (`perception_engine.py`):** Captures DOM element snapshots and page signatures (`compute_page_signature`) to detect page state changes between automation steps.
- **Dual-Stage LLM Planner (`llm_planner.py`):** VLM perceives the current page state, matches form fields to PII Vault data, and outputs a structured list of actions (type, click, select, upload) for the next step.
- **ReAct State Machine (`automation_graph.py`):** LangGraph graph iterating through `perceive_node` → `plan_node` → `act_node` cycles until the form is complete.
- **Zero Auto-Submit Safety Gate (`final_confirmation_node`):** Hard halt before final submission. Takes a full-page screenshot, sends it to the user on Discord, and requires explicit text confirmation before submitting.
- **Integration Test Harness (`test_integration.py`):** Validates the full PDF fast-track path and the online ReAct state machine independently, including final gate halt verification.

---

## Repository Structure

```text
Yojna-Mitra-Project/
│
├── core_inference/
│   ├── graph.py                   # LangGraph StateGraph — all intents, routing, HITL gating
│   ├── hybrid_rag.py              # Top-7 Hybrid RAG pipeline + 3-tier internal reasoning
│   ├── doc_requirements.py        # Scheme document requirements parser
│   ├── clear_vault.py             # PII Vault manual purge utility
│   └── inspect_db.py              # DB inspection utility
│
├── product_inference/
│   ├── bot.py                     # Discord DM gateway — async message loop, graph invocation
│   ├── bot_audio.py               # Voice-enabled variant with Sarvam STT integration
│   ├── db.py                      # PostgreSQL helpers (user_profiles + pii_vault + scheme queries)
│   ├── document_handler.py        # Security bouncer — validation, magic byte check, sanitization
│   ├── document_extractor.py      # 3-Pass Vision LLM pipeline — classify → extract → validate
│   ├── automation_graph.py        # ReAct LangGraph — perceive → plan → act → final gate
│   ├── browser_manager.py         # Playwright session manager + _PW_EXECUTOR thread boundary
│   ├── perception_engine.py       # DOM snapshot + page signature engine
│   ├── llm_planner.py             # Dual-stage VLM action planner
│   ├── universal_automation.py    # Automation orchestrator — mode routing + graph launcher
│   ├── test_integration.py        # Integration test harness (PDF fast-track + online ReAct)
│   └── __init__.py
│
├── data_extraction/
│   ├── scraping.py                # Playwright crawler for myScheme.gov.in
│   ├── migrate_and_update_schemes.py  # DB migration + application mode classification
│   ├── extract_schema_llm.py      # LLM-based schema field extractor
│   └── schemes.json               # Raw scraped scheme data
│
├── testing_n_diagnostics/
│   ├── test_hybrid_rag.py         # Parametrized persona evaluation suite (10 personas)
│   ├── test_phase7_flow.py        # Phase 7 HITL flow tests
│   ├── rag_evaluation_log.md      # Evaluation results log
│   └── personas/                  # Citizen persona JSON definitions
│
├── configuration/
│   ├── requirements.txt
│   ├── user_profile_schema.json
│   └── progress.json              # Scraper progress tracker
│
├── browser_profiles/              # Persistent Playwright user profiles (gitignored)
├── screenshots/                   # Automation screenshots (gitignored)
├── temp_documents/                # Ephemeral document staging area (gitignored)
└── Yojana Mitra Architecture.drawio.png
```

---

## Local Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/farazn7/Yojna-Mitra-Project.git
cd Yojna-Mitra-Project
```

### 2. Create Virtual Environment & Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r configuration/requirements.txt
```

### 3. Start the PostgreSQL + pgvector Container (Docker)

```bash
docker run --name pgvector-db \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
```

> Database tables (user profiles, PII vault, LangGraph checkpointer) are auto-initialized on first bot start.

### 4. Pull Local Models via Ollama

```bash
ollama pull hf.co/qwen/qwen3-8b-gguf:q4_k_m   # Reasoning + recommendation
ollama pull llama3.1                             # Routing + structuring
ollama pull gemma3:4b                            # Vision + document classification
ollama pull nomic-embed-text                     # Vector embeddings
```

### 5. Configure Environment

Create a `.env` file in the project root:

```env
DISCORD_TOKEN=your_bot_token_here
```

### 6. Run the Persona Evaluation Suite

```bash
pytest testing_n_diagnostics/test_hybrid_rag.py -v -s
```

### 7. Launch the Bot

```bash
python -m product_inference.bot
```

### 8. Run the Automation Integration Tests

```bash
python -m product_inference.test_integration
```