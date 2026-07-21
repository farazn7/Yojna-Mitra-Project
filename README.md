# Yojana Mitra — योजना मित्र

> Your AI friend for Indian Government Schemes.

Yojana Mitra is an autonomous, multi-user AI agent deployed on **Telegram** and **Discord** that discovers Indian government welfare schemes a citizen qualifies for and autonomously navigates the application portals on their behalf — all through a simple chat conversation.

The system handles natural language in Hindi, English, and Hinglish. It is built specifically for users who face language, literacy, and technical barriers when interacting with government portals.

**Live Demo:** [yojanamitra.vercel.app](https://yojna-mitra.vercel.app) &nbsp;|&nbsp;
**Telegram:** [@yojnamitra_bot](https://t.me/yojnamitra_bot) &nbsp;|&nbsp;
**Discord:** [Add to Server](https://discord.com/oauth2/authorize?client_id=1518138316510199808)

---

## Architecture & Tech Stack

| Layer | Technology |
|---|---|
| **Core Language** | Python 3.13 |
| **Interfaces** | Telegram Bot API, Discord (DM-based, async multi-user) |
| **Agentic Framework** | LangGraph (`StateGraph`, `PostgresSaver`, Conditional Edge Routing) |
| **Local Inference** | Ollama — Qwen3-8B (reasoning/recommendation), Llama 3.1 (routing/structuring), Gemma3:4B (vision/document extraction), nomic-embed-text (embeddings) |
| **Vector Store & Persistence** | PostgreSQL + pgvector (Docker container) |
| **Checkpointing** | LangGraph `PostgresSaver` backed by `psycopg_pool` |
| **Web Automation** | Playwright (Sync API, isolated via ThreadPoolExecutor) |
| **Data Scraping** | Playwright-based crawler targeting myScheme.gov.in |
| **Voice Support** (optional) | Sarvam AI — Saaras v3 STT + Bulbul v3 TTS (Hindi & English) |
| **Landing Page** | Static HTML/CSS/JS deployed on Vercel with a Vercel serverless status API |
| **Status Monitoring** | Healthchecks.io — live bot heartbeat with per-bot indicator on the landing page |

---

## Engineering Milestones

### Phase 1 — Data Engineering & Extraction
Built a resilient Playwright-based scraper to crawl myScheme.gov.in, which runs on a Next.js/React architecture with dynamic DOM hydration.

- **React Hydration Bypassing:** DOM-polling strategy waits for React to fully paint new elements before paginating, preventing stale-element duplication.
- **Idempotency & Crash Recovery:** Two-file architecture (`schemes.json` + `progress.json`) saves state every 10 extractions, enabling seamless resume without duplication.
- **Hidden DOM Extraction:** Targeted parent container IDs (`#eligibility`, `#documents-required`, `#sources`) directly instead of interacting with unstable visible UI tabs.
- **Application Mode Classification:** Parsed the `#sources` section of each scheme page to classify each scheme's application mode: `online`, `pdf_form`, or `physical_only`.

**Result:** 230+ unique Tamil Nadu scheme records with eligibility criteria, required documents, portal URLs, and application mode stored in PostgreSQL.

---

### Phase 2 — Hybrid RAG Architecture & Persona Evaluation
Replaced a legacy regex keyword matcher with a multi-tiered Hybrid RAG pipeline:

1. **Hard Filter Layer:** SQL `WHERE` clauses instantly eliminate schemes outside strict numerical bounds (age, income, gender, disability status).
2. **Semantic Ranking Layer:** pgvector cosine similarity (`<=>`) ranks remaining candidates by semantic closeness to the user's query.
3. **Hybrid Re-scoring:** Post-retrieval scores combine 70% vector similarity + 30% string overlap (`difflib.SequenceMatcher`) for final ranking.
4. **Automated Persona Testing:** `test_hybrid_rag.py` parametrized across 10 citizen personas (e.g., `rural_student`, `widowed_farmer`) validates the full retrieval and generation pipeline.

---

### Phase 3 — Conversational State Machine & Live Runtime Routing
Turned the static testing harness into a live, interactive bot interface.

- **13-Step Onboarding:** Progressive state machine sequentially captures and validates comprehensive user profiles (gender, caste, income, minority status, BPL, etc.).
- **JSONB Profile Storage:** Flexible JSONB + atomic append operations replace rigid boolean database columns.
- **Message Chunking:** Splits LLM outputs at newline boundaries for payloads exceeding Discord's 2,000-character and Telegram's 4,096-character API limits.
- **Live RAG Context Injection:** Pulls the user's verified profile from PostgreSQL at runtime using their platform ID and injects it into every LLM prompt.

---

### Phase 4 — Thread-Safe Async Multi-Threading & Intent Routing
- **Non-Blocking Execution:** Dispatches heavy LLM inference and pgvector queries to isolated background threads via `asyncio.to_thread()`, preventing bot gateway heartbeat blocks.
- **Intent Classifier Middleware:** A lightweight Llama 3.1 routing layer classifies every message as `SCHEME_QUERY`, `PROFILE_UPDATE`, `CHIT_CHAT`, or `APPLY_SCHEME` at temperature 0.0. Defaults to `CHIT_CHAT` on ambiguous input to avoid triggering the full RAG pipeline on casual messages.

---

### Phase 5 — Agentic Workflow & LangGraph State Machine
Re-architected the entire runtime into a LangGraph `StateGraph` with persistent memory:

- **Stateful Graph Architecture:** `ConversationState` TypedDict with LangGraph message reducers (`add_messages`, `RemoveMessage`) for sliding conversation history.
- **PostgreSQL Checkpointer:** `PostgresSaver` backed by `psycopg_pool` (`autocommit=True`) persisting conversation threads keyed by platform user IDs.
- **Multi-Intent Routing:** Conditional edges routing across `SCHEME_QUERY`, `PROFILE_UPDATE`, `CHIT_CHAT`, `DOC_RECEIVED`, and `APPLY_SCHEME` intents.
- **Thought Extraction Middleware:** Custom regex middleware strips Qwen3's `<think>` reasoning blocks before sending output to users, with empty-payload guardrails.
- **Short-Term Memory Summarizer:** Auto-triggered when conversation history exceeds 8 messages — summarizes prior context to maintain VRAM efficiency while preserving continuity.

---

### Phase 6 — Document Intelligence Pipeline & PII Vault
Full document handling pipeline for Aadhaar, PAN, Voter ID, College IDs, and certificates uploaded directly via bot DM.

- **3-Pass Vision LLM Pipeline (`document_extractor.py`):**
  - **Pass 0 — Classification:** Gemma3:4B identifies document type (`aadhaar`, `pan_card`, `voter_id`, etc.) in ~1 second.
  - **Pass 1 — Targeted Extraction:** Document-specific prompts extract cardholder PII (name, DOB, address) using Gemma3 multimodal vision.
  - **Pass 2 — Structuring & Validation:** Llama 3.1 formats raw VLM output into clean JSON. Regex validators confirm Aadhaar (12 digits), PAN (`AAAAA9999A`), and pincode (6 digits) formats.
- **PII Vault (`pii_vault` table):** Physically separated from `user_profiles`. Stores document-verified sensitive data with a 24-hour TTL, deep-merge UPSERT semantics, ghost-filter expiry (`WHERE expires_at > NOW()`), and an auto garbage collector (`purge_expired_vault()`).
- **Document Cache System:** Checks `pii_vault` for existing valid data before any AI inference. Cache hits skip VLM entirely.
- **Security Bouncer (`document_handler.py`):** Three-layer security — pre-download size/extension validation, magic byte integrity check, and on-demand EXIF stripping + JPEG compression.

---

### Phase 7 & 7.1 — HITL Gating, Corrective RAG & Fuzzy Interceptor
- **HITL Document Checklist Gating:** Stateful document collection loop (`request_next_document` in `graph.py`) driven by `db.get_scheme_documents_needed`. Separates scannable files from offline manual prerequisites. Generates a `user_manifest.md` audit trail.
- **Mode Bypass Guard:** Unconditional DB check in `request_next_document` prevents stale LangGraph checkpoints from bypassing application mode gates when users switch schemes mid-session.
- **Corrective Hybrid RAG:** Upgraded retrieval to Top-7 hybrid search combining pgvector cosine distance and keyword string overlap via `difflib.SequenceMatcher`. Structured 3-tier internal reasoning framework (Exact Match → Near-Miss → No Match) shapes LLM evaluation logic.
- **Fuzzy Trigram Application Interceptor (`db.find_similar_scheme_name`):** PostgreSQL trigram match (`pg_trgm`) + acronym/abbreviation search (e.g., matching "PEACE" to the full scheme name in parentheses) with sub-1-second latency.
- **Pre-Playwright Mandatory Enforcement Gate:** Hard security gate before any web automation. If mandatory documents were skipped, the system enforces upload with double-check cache validation.

---

### Phase 8 — Universal Web Automation & ReAct Agent
Playwright-driven autonomous form filler that navigates government portals without any hardcoded selectors.

- **Persistent Session Isolation (`browser_manager.py`):** Each user gets a dedicated browser profile directory (`browser_profiles/{user_id}/`) retaining cookies and session tokens across automation turns.
- **Async Thread Isolation Boundary:** All Playwright `sync_api` calls routed through a dedicated `ThreadPoolExecutor` (`PW_Sync_Worker`) isolated from the bot's `asyncio` event loop.
- **Application Mode Fast-Track:**
  - `online` → Launches ReAct automation agent.
  - `pdf_form` → Returns download link with print & submit instructions. No Playwright invoked.
  - `physical_only` / `unknown` → Returns nearest office location and manual instructions.
- **Perception Engine (`perception_engine.py`):** Captures DOM element snapshots and page signatures to detect page state changes between automation steps.
- **Dual-Stage LLM Planner (`llm_planner.py`):** VLM perceives current page state, matches form fields to PII Vault data, and outputs structured actions (type, click, select, upload).
- **ReAct State Machine (`automation_graph.py`):** LangGraph graph iterating `perceive_node` → `plan_node` → `act_node` cycles until the form is complete.
- **Zero Auto-Submit Safety Gate:** Hard halt before final submission. Takes a full-page screenshot, sends it to the user, and requires explicit text confirmation before submitting.

---

### Phase 9 — Telegram Support, Cross-Platform Parity & Quality Fixes
Extended the system from Discord-only to a fully dual-platform deployment.

- **Telegram Bot (`bot_telegram.py`):** Full feature parity with Discord — onboarding, scheme recommendation, document upload & vault, HITL automation, screenshots, and `/reset`/`/stop` commands.
- **Universal `/reset`:** Nuclear wipe of `user_profiles`, `pii_vault`, `checkpoints`, and `checkpoint_writes` tables for the user. Starts a clean session immediately.
- **Universal `/stop`:** Cancels any in-progress LangGraph task. Silent if nothing is running (no false "nothing running" replies).
- **Fuzzy Match Fixes:** Added acronym-in-parentheses search and fixed a logic bug where word-overlap scores were calculated but never assigned, causing empty fuzzy match results.
- **Optional Voice Mode (`bot_audio.py`):** Run instead of the two regular bots to enable Sarvam AI voice notes on both platforms simultaneously. Sarvam Saaras v3 STT transcribes voice messages; Sarvam Bulbul v3 TTS synthesizes replies. Not enabled by default to conserve Sarvam API credits.

---

### Phase 10 — Landing Page & Live Status
- **Landing Page (`front_end/`):** Static site deployed on Vercel. Features the real project logo, embedded demo video, platform CTA buttons, architecture pills, and a scroll-reveal animated layout.
- **Live Bot Status Indicator:** Green/red pulsing dot on the landing page showing real-time bot status. Each bot pings Healthchecks.io every 60 seconds. A Vercel serverless function (`/api/status`) proxies the Healthchecks read-only API (keeping the key server-side) and returns `{ telegram, discord }` status. The frontend updates every 60 seconds showing:
  - "Live on Telegram & Discord" — both bots are running
  - "Telegram Offline" / "Discord Offline" — one bot is down
  - "Servers Offline" — both bots are down

---

## Repository Structure

```text
Yojna-Mitra-Project/
│
├── core_inference/
│   ├── graph.py                    # LangGraph StateGraph — all intents, routing, HITL gating
│   ├── hybrid_rag.py               # Top-7 Hybrid RAG pipeline + 3-tier internal reasoning
│   ├── doc_requirements.py         # Scheme document requirements parser
│   ├── clear_vault.py              # PII Vault manual purge utility
│   └── inspect_db.py               # DB inspection utility
│
├── product_inference/
│   ├── bot.py                      # Discord DM gateway — async message loop, graph invocation
│   ├── bot_telegram.py             # Telegram gateway — full feature parity with Discord
│   ├── bot_audio.py                # Optional voice mode for BOTH platforms (Sarvam AI)
│   ├── db.py                       # PostgreSQL helpers (user_profiles, pii_vault, scheme queries)
│   ├── document_handler.py         # Security bouncer — validation, magic byte check, sanitization
│   ├── document_extractor.py       # 3-Pass Vision LLM pipeline — classify → extract → validate
│   ├── automation_graph.py         # ReAct LangGraph — perceive → plan → act → final gate
│   ├── browser_manager.py          # Playwright session manager + thread isolation boundary
│   ├── perception_engine.py        # DOM snapshot + page signature engine
│   ├── llm_planner.py              # Dual-stage VLM action planner
│   ├── universal_automation.py     # Automation orchestrator — mode routing + graph launcher
│   ├── test_integration.py         # Integration test harness
│   └── __init__.py
│
├── front_end/
│   ├── index.html                  # Landing page (Vercel static)
│   ├── vercel.json                 # Vercel routing config
│   ├── api/
│   │   └── status.js               # Serverless function — proxies Healthchecks.io status
│   └── assets/
│       ├── logo.png                # Project logo
│       └── demo.mp4                # Demo video
│
├── data_extraction/
│   ├── scraping.py                 # Playwright crawler for myScheme.gov.in
│   ├── migrate_and_update_schemes.py   # DB migration + application mode classification
│   ├── extract_schema_llm.py       # LLM-based schema field extractor
│   └── schemes.json                # Raw scraped scheme data (230+ schemes)
│
├── testing_n_diagnostics/
│   ├── test_hybrid_rag.py          # Parametrized persona evaluation suite (10 personas)
│   ├── test_phase7_flow.py         # Phase 7 HITL flow tests
│   ├── rag_evaluation_log.md       # Evaluation results log
│   └── personas/                   # Citizen persona JSON definitions
│
├── configuration/
│   ├── requirements.txt
│   ├── user_profile_schema.json
│   └── progress.json               # Scraper progress tracker
│
├── browser_profiles/               # Persistent Playwright user profiles (gitignored)
├── screenshots/                    # Automation screenshots (gitignored)
├── temp_documents/                 # Ephemeral document staging area (gitignored)
└── temp_voice/                     # Ephemeral audio staging area (gitignored)
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
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS
pip install -r configuration/requirements.txt
```

### 3. Start PostgreSQL + pgvector (Docker)

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
ollama pull gemma3:4b                            # Vision + document extraction
ollama pull nomic-embed-text                     # Vector embeddings
```

### 5. Configure Environment

Create a `.env` file in the project root:

```env
DISCORD_TOKEN=your_discord_bot_token
TELEGRAM_TOKEN=your_telegram_bot_token
SARVAM_API_KEY=your_sarvam_key         # Only needed for bot_audio.py
HEALTHCHECKS_API=your_healthchecks_readonly_key   # Only needed for Vercel status API
```

### 6. Launch the Bots

**Standard mode (recommended):**
```bash
# Run both bots in separate terminals
python -m product_inference.bot           # Discord
python -m product_inference.bot_telegram  # Telegram
```

**Audio/voice mode (uses Sarvam credits):**
```bash
# Replaces both standard bots — runs Discord + Telegram with voice support
python -m product_inference.bot_audio
```

### 7. Run the Persona Evaluation Suite

```bash
pytest testing_n_diagnostics/test_hybrid_rag.py -v -s
```

### 8. Run the Automation Integration Tests

```bash
python -m product_inference.test_integration
```

---

## Deploying the Landing Page on Vercel

1. Push the repository to GitHub.
2. Import the project on [Vercel](https://vercel.com).
3. Set the **Root Directory** to `front_end`.
4. Add the environment variable `HEALTHCHECKS_API` in Vercel project settings.
5. Deploy — the serverless `/api/status` function is automatically picked up.

---

## Contact

**Email:** nezamifaraz318@gmail.com &nbsp;|&nbsp; **GitHub:** [farazn7](https://github.com/farazn7/Yojna-Mitra-Project)