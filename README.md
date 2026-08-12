# Yojana Mitra — योजना मित्र

> Your AI friend for Indian Government Schemes.

Yojana Mitra is an autonomous, multi-user AI agent deployed on **Telegram** and **Discord** that discovers Indian government welfare schemes a citizen qualifies for and autonomously navigates the application portals on their behalf — all through a simple chat conversation.

The system handles natural language in Hindi, English, and Hinglish. It is built specifically for users who face language, literacy, and technical barriers when interacting with government portals.

**Live Demo:** [yojanamitra.vercel.app](https://yojna-mitra-project.vercel.app) &nbsp;|&nbsp;
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
- **ReAct State Machine (`automation_graph.py`):** LangGraph subgraph iterating `perceive_node` → `plan_node` → `execute_node` → `check_status_node` until the form is complete or a human is needed.
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

### Phase 11 — Ground-Truth Retrieval Evaluation
The original persona sweep (`test_hybrid_rag.py`) only asserted the pipeline returned non-empty text — a flat LLM refusal on a real welfare query still passed. Added a ground-truth eval that asserts against `schemes_fetched`, the pipeline's real pre-LLM retrieval output, instead of the LLM's prose.

- **Labeled Dataset (`scheme_eval_labels.json`):** 10 positive cases + 5 hard-negative boundary cases, every expected match verified against the live `government_schemes` table (not the scraped JSON), plus a `near_miss_blocklist` for personas with no genuine corpus match — avoiding an unfalsifiable `== []` assertion.
- **Hard-Negative Coverage:** 5 cases, one per SQL-filterable field (`min_age`, `max_age`, `max_income`, `is_women_only`, `is_differently_abled`), each mutating a real persona just past a real scheme's boundary value to test the hard-filter's exclusion behavior.
- **Three Live Data Bugs Found:** Verified the pipeline's actual SQL filter against production data and surfaced a mis-scoped age boundary, a gender-neutral scheme mislabeled women-only, and a duplicate ingestion row — documented in `scheme_eval_labels.json`'s `known_issues_surfaced`, not silently worked around.
- **Result:** 24/25 assertions passed; the one failure (`p10_senior_citizen_bpl` at `top_k=3`) is a genuine retrieval-ranking signal — the correct scheme ranks 9th and only surfaces at `top_k=10`.

See `testing_n_diagnostics/eval_scheme_recommendation.md` for full methodology and results.

---

### Phase 12 — Mock Portal, End-to-End Automation Eval & Loop Hardening

Phase 8 built the ReAct loop; this phase was the first time it was driven end to end against a **real running web app**, and running it was itself the finding — several code paths had provably never executed successfully.

**`mock_govt_portal/` (React + Vite):** a 4-step wizard built as a realistic adversary, not a friendly fixture — a genuinely async State→District cascade, a hidden file input behind a styled label, a generated CAPTCHA with its own Refresh button, controls that stay disabled until their field is filled, and a Submit wired to a native `window.confirm()`.

**Result:** a single unattended run now completes Steps 1–4, halts at the review gate, and files the application on `CONFIRM` — with the portal's own acknowledgement asserted rather than inferred from the click. The only human inputs are answers to questions the agent asks (the CAPTCHA it cannot read, the OTP it cannot receive, two consents) plus the literal word `CONFIRM`.

- **Zero Auto-Submit, now three independent layers:** the planner **refuses the press before it executes** (detection that runs after an irreversible action is not a safeguard — the original check ran post-click); the graph still requires a literal `CONFIRM`; and the dialog interceptor accepts a native confirm only under a one-shot, expiring authorization armed exclusively by the CONFIRM code path. Nothing the page does can arm it.
- **Every English keyword list removed from the automation path.** Four existed; two of them decided *when the run halts for CONFIRM* and *when the citizen is told their application was filed*, so a Hindi or Tamil portal would have sailed past its own review screen. Replaced with semantic judgments (`is_final_review_screen`, `page_confirms_submission`) covered in English, Hindi and Tamil.
- **Observation-based loop rules** — direction from visit order, destruction from erasure, effectiveness from aftermath, and binding from responsiveness. None keys on wording; each is learned at runtime from what the browser actually did. They survive HITL pauses via `automation_page_memory`, since every resume is a fresh subgraph invocation.
- **Deterministic action derivation:** the planner's second LLM call — whose only job was mapping Stage 1's output to Playwright verbs, a lookup with exactly one correct answer — was deleted. Four "override the model in Python" patches ceased to exist rather than being kept, including an English `("back", "cancel", "previous")` list on a product serving citizens who may see **"पीछे"**. Planner latency roughly halved.
- **Anti-hallucination provenance check:** a proposed value must be backed by a real vault key. The model correctly inferred *"Chennai is in Tamil Nadu"* and the check rejected it anyway, because no vault key backed it. On a form that becomes a legal government application, a plausible guess is not good enough. It also never ticks "I'm not a robot" — **nothing in this codebase attempts to solve a CAPTCHA**, by design.

See `testing_n_diagnostics/eval_web_automation.md` for the full walkthrough, the defect table, and the design rule the eval enforces.

---

## How This Was Built

The architecture, the product decisions and the engineering constraints in this repository are mine. **Claude Code** was used as an implementation partner on two specific bodies of work, where I set the direction and reviewed the result and it wrote the bulk of the code.

**The browser chat interface** (`product_inference/web/`, `core_inference/events.py`, `core_inference/session.py`). The constraint I set was that the browser must be a *renderer* over the same turn logic the bots already run — never a second decision point on submitting a government form. That ruled out the obvious implementation, where the web surface grows its own copy of the pipeline; instead the turn moved into one shared session layer emitting typed events, and every surface just draws them. Two decisions followed from it and both held: the web Confirm button sends the literal string `CONFIRM` through the ordinary chat path rather than a privileged route, so the graph's HITL gate remains the only thing that can authorize a submit; and browser identity is *delegated* from an already-authenticated Telegram session via a signed handoff token, because building a login next to a PII vault is not a trade I wanted to make.

**The evaluation suites** (`testing_n_diagnostics/`). Chiefly the ground-truth eval for hybrid RAG scheme matching — the labeled query→expected-scheme dataset verified against live `government_schemes` values rather than scraped eligibility text, and the hard-negative boundary cases. Before it, the persona suite asserted only that a response came back non-empty, which is not a measurement of retrieval.

Review stayed with me, and it was load-bearing. One example: the shared `/reset` implementation initially derived the browser profile directory as `user_<user_id>`, when automation actually creates `user_auto_<user_id>` — the two halves of that name are assembled in different files. It would have deleted nothing and reported success, leaving live government-portal logins on disk after a citizen asked to be forgotten. It was caught by running the reset against a real profile directory instead of trusting the code path.

Commits on the browser-interface work carry a `Co-Authored-By: Claude` trailer. The evaluation work predates that convention and is credited here instead.

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
├── mock_govt_portal/               # React + Vite 4-step wizard — the automation eval's target
│   ├── src/
│   │   ├── App.jsx                 # Wizard shell + shared form state
│   │   └── steps/                  # Personal details, document upload, OTP/CAPTCHA, review & submit
│   ├── index.html
│   └── package.json                # `npm run dev` → http://localhost:5173
│
├── testing_n_diagnostics/
│   ├── test_scheme_recommendation_matching.py   # Ground-truth retrieval eval — asserts on schemes_fetched
│   ├── scheme_eval_labels.json     # Labeled dataset — 10 positive cases + 5 hard negatives
│   ├── test_hybrid_rag.py          # Qualitative persona sweep (10 personas) — LLM tone/quality, not correctness
│   ├── test_web_automation_react_loop.py        # Web automation eval — drives the real mock portal
│   ├── eval_scheme_recommendation.md            # Retrieval eval methodology, results, known issues surfaced
│   ├── eval_web_automation.md                   # Automation eval walkthrough, defect table, design rule
│   ├── check_*.py                  # Local-only diagnostics (gitignored) — see eval_web_automation.md
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

### 7. Run the Evaluation Suites

```bash
# Ground-truth retrieval eval — asserts on schemes_fetched against a verified label set
pytest testing_n_diagnostics/test_scheme_recommendation_matching.py -v -s

# Qualitative persona sweep — LLM tone/quality, logged for human review
pytest testing_n_diagnostics/test_hybrid_rag.py -v -s
```

### 8. Run the Automation Integration Tests

```bash
python -m product_inference.test_integration
```

### 9. Run the Web Automation Eval (live, slow)

Needs the mock portal and Ollama running first:

```bash
cd mock_govt_portal && npm install && npm run dev   # serves http://localhost:5173
```

Then, from the project root:

```bash
pytest testing_n_diagnostics/test_web_automation_react_loop.py -v -s
```

A real Chrome window opens and you can watch the agent work. Each planner cycle is a real local LLM call (~15–20s), so a full pass takes upwards of half an hour — same scoping as the RAG eval, not for a fast CI loop.

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
