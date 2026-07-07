# Yojana Mitra Project

Yojana Mitra is an autonomous, multi-user AI agent deployed on Discord designed to discover Indian government welfare schemes a citizen qualifies for and guide them through eligibility evaluation.

The system leverages natural language processing (Hindi, English, and Hinglish), advanced local Hybrid RAG pipelines, stateful LangGraph agentic workflows, and automated document-schema extraction to navigate complex welfare criteria, specifically targeting users who face language, literacy, and technical barriers.

---

## Project Architecture & Tech Stack

* **Core Language:** Python 3.13
* **Automation & Scraping:** Playwright (Sync API)
* **Data Layer & Persistence:** PostgreSQL with pgvector extension & LangGraph Checkpointer (Hosted inside a Docker Sandbox Container)
* **Agentic Framework:** LangGraph (`StateGraph`, `PostgresSaver`, Conditional Edge Routing)
* **Intelligence & Local Inference Engine:** Ollama running Qwen 3 (8B Reasoning Model), Llama 3.1 (8B Execution Model), Gemma 3 (4B Vision Model) & nomic-embed-text (High-Dimensional Vector Model)
* **Testing Infrastructure:** pytest (Parametrized Automation Matrix Harness)
* **Interface & Presentation Layer:** bot.py & bot_audio.py (Multi-user isolated DM gateway with an Asynchronous, Non-Blocking Event Loop)

---

## Engineering Milestones Achieved

### Phase 1: Data Engineering & Extraction
Developed a highly resilient, Playwright-based crawler designed to navigate the dynamic Next.js/React hydration architecture of myScheme.gov.in.
* **React State Hydration Bypassing:** Utilized a Green Button DOM-polling strategy to ensure pagination only triggered after React fully painted new page elements, preventing stale-element duplication.
* **Idempotency & State Persistence:** Implemented a two-file architecture (schemes.json and progress.json). Saves state frames after every 10 extractions, allowing seamless crash recovery without asset duplication.
* **Hidden DOM Extraction:** Bypassed Strict Mode layout hiding by targeting parent container IDs (#eligibility, #documents-required) directly instead of interacting with unstable visible UI tabs.

Result: Extracted explicit eligibility criteria, required documents, and URLs for 230 unique State/UT schemes in Tamil Nadu into a structured reference database.

### Phase 2: Hybrid RAG Architecture & Persona Evaluation
Migrated away from a legacy regex keyword heuristic matching engine (matcher.py) to eliminate semantic blindspots when parsing complex legislative texts. Built a multi-tiered Hybrid Retrieval-Augmented Generation (RAG) pipeline:
1. **The Hard Filter Layer:** Structured profile data extracted via extract_schema_llm.py is evaluated through high-speed SQL WHERE clauses to instantly filter rigid numerical bounds (e.g., bounding user parameters inside min_age, max_age, or checking max_income).
2. **The Semantic Ranking Layer:** Candidate schemes undergo a pgvector cosine similarity search matching user chat intent with multi-dimensional document chunks.
3. **Contextual Evaluation Engine:** Built an automated testing suite (test_hybrid_rag.py) parametrized across 10 distinct citizen personas (e.g., rural_student, widowed_farmer). Incorporates strict system configurations to force total fact synthesis, manage situational sensitivity constraints, and handle null financial values cleanly under proactive thermal throttling.

### Phase 3: Conversational State Machine & Live Runtime Routing
Transitioned the architecture from a static testing harness into a live, production-ready interactive interface.
* **13-Step Conversational Onboarding:** Built a progressive state machine inside bot.py to sequentially capture and validate comprehensive user profiles (Gender, Caste, Minority Status, BPL records, etc.), eliminating structural data omission gaps.
* **Dynamic Database Schema Evolution:** Upgraded the PostgreSQL layer via db.py to utilize a native JSONB data type combined with atomic appends. Replaced rigid boolean architectures with a flexible occupation string identifier to prevent hardcoded keyword constraints.
* **Discord API Guardrails & Chunking:** Implemented an aggressive streaming chunking algorithm at the presentation layer to split LLM context blocks at newline boundaries if payloads exceed 1,900 characters, bypassing Discord's 2,000-character truncation errors.
* **Live RAG Context Injection:** Configured the production message loop to dynamically pull the user's explicit profile out of PostgreSQL at runtime based on their unique, unchangeable Snowflake User ID, feeding those verified facts straight into the Llama 3.1 evaluation prompt.

### Phase 4: Thread-Safe Async Multi-Threading & Intent Routing (Week 4)
* **Non-Blocking Execution Model:** Upgraded the bot.py event loop to dispatch the heavy, blocking local LLM inferences and PostgreSQL vector queries onto isolated background execution threads via asyncio.to_thread(). This prevents CPU-bound stalls and eliminates the Discord Gateway Heartbeat Blocked crash.
* **Upfront Intent Classifier Middleware:** Integrated a lightweight local Llama routing layer to analyze input strings instantly. If a user communicates via casual chat (greetings, small talk, thanks), the bot intercepts it as CHIT_CHAT and responds dynamically using the user's input language, bypassing the PostgreSQL vector framework completely to optimize compute efficiency.

### Phase 5: Agentic Workflow & LangGraph State Machine Integration
Re-architected the runtime execution engine into an autonomous LangGraph agentic graph (`core_inference/graph.py`) with persistent memory and reasoning capabilities:
* **Stateful Graph Architecture:** Replaced monolithic procedural chains with a robust `StateGraph` managing a structured `ConversationState` TypedDict. Leverages LangGraph message reducers (`add_messages`, `RemoveMessage`) for sliding conversation history.
* **PostgreSQL Checkpointer & Dual-Write Persistence:** Integrated `PostgresSaver` backed by a high-performance `ConnectionPool` (`psycopg_pool` with `autocommit=True`) to persist conversation threads keyed by Discord Snowflake IDs. Implemented atomic dual-writes updating both graph checkpoint state and PostgreSQL `user_profiles` analytics tables during onboarding.
* **Multi-Intent Routing & Thought Extraction:** Configured conditional edges routing across three core intents (`SCHEME_QUERY`, `PROFILE_UPDATE`, `CHIT_CHAT`). Developed custom regex middleware (`extract_and_print_thoughts`) to extract `<think>` tags generated by reasoning models (such as Qwen 3), output live hacker-aesthetic diagnostics to the terminal, and enforce empty-payload guardrails to prevent Discord API 50006 errors.
* **Short-Term Memory Summarizer:** Implemented an automatic sliding memory window node triggered when conversation history exceeds 8 messages, seamlessly summarizing prior exchanges to optimize local VRAM and token budgets while retaining long-term contextual awareness.

### Phase 6: Document Intelligence Pipeline & PII Vault (Week 5)
Engineered a complete document handling pipeline enabling users to upload Indian government documents (Aadhaar, PAN, Voter ID, College IDs, Certificates) directly via Discord DM, with automated AI-powered data extraction and privacy-first temporary storage.
* **3-Pass Document Intelligence Pipeline:** Replaced a failed Tesseract OCR approach (which returned empty strings on Hindi text and garbled output on ID cards with holograms/gradients) with a multi-stage Vision LLM architecture:
  * **Pass 0 — Classification:** Gemma 3 Vision (`gemma3:4b`) inspects the uploaded image and identifies the document type (`aadhaar`, `pan_card`, `voter_id`, `college_id`, `back_of_card`, etc.) in ~1 second, enabling document-specific cache checks and targeted extraction.
  * **Pass 1 — Targeted Extraction:** Instead of a generic "extract all text" prompt, each document type has a dedicated `EXTRACTION_PROMPTS` entry instructing the VLM to extract only the card holder's personal information (excluding organizational data like college emails/websites) with addresses split into structured components (`address`, `city`, `state`, `pincode`).
  * **Pass 2 — Structuring & Validation:** Llama 3.1 formats raw VLM output into clean JSON, followed by structural regex validation (`VALIDATORS`) confirming Aadhaar numbers are 12 digits, PAN matches `AAAAA9999A` format, and pincodes are 6-digit codes. Empty/missing fields are silently omitted rather than treated as validation failures.
* **PII Vault — Privacy-Isolated Temporary Storage:** Created a dedicated `pii_vault` PostgreSQL table physically separated from the permanent `user_profiles` table. Stores document-verified sensitive data (Aadhaar numbers, addresses, PAN) in canonical JSONB with a strict 24-hour TTL. Features deep-merge (`||`) UPSERT semantics so multiple document uploads accumulate into a single user row without redundancy, a ghost-filter (`WHERE expires_at > NOW()`) making data instantly invisible upon expiry, and a garbage collector (`purge_expired_vault()`) for physical deletion.
* **Document-Specific Cache System:** Before invoking any AI model, the pipeline checks `pii_vault` for existing unexpired data of the specific document type (`source_documents ? 'aadhaar'`). Cache hits skip VLM entirely, saving GPU compute and API latency. The `back_of_card` and `unknown` types bypass caching to prevent false positives.
* **Security Bouncer Pipeline (`document_handler.py`):** Three-layer file security: pre-download size/extension validation, magic byte integrity verification (detecting malware disguised as PDFs/JPGs), and on-demand government portal sanitization (EXIF stripping, PNG→JPG conversion, binary-search JPEG compression to 20–100KB). Portal sanitization is disabled by default to preserve high-res originals for AI Vision accuracy.


## Repository Structure

```text
Yojna-Mitra-Project/
│
├── core_inference/
│   ├── graph.py
│   ├── hybrid_rag.py
│   ├── inspect_db.py
│   └── extract_schema_llm.py
│
├── product_inference/
│   ├── bot.py                    # Discord DM gateway with document interceptor & PII vault wiring
│   ├── bot_audio.py              # Voice-enabled variant with Sarvam STT integration
│   ├── db.py                     # PostgreSQL helpers (user_profiles + pii_vault + cache/expiry)
│   ├── document_handler.py       # Security bouncer: validation, integrity, govt portal sanitization
│   └── document_extractor.py     # 3-Pass AI pipeline: classify → targeted extract → structure & validate
│
├── data_extraction/
│   ├── scraping.py
│   ├── schemes.json
│   └── progress.json
│
├── testing_&_diagnostics/
│   ├── test_hybrid_rag.py
│   ├── rag_evaluation_log.md
│   └── personas/
│
├── temp_documents/               # Ephemeral staging area for uploaded files (gitignored)
│
└── configuration/
    ├── user_profile_schema.json
    └── requirements.txt
```

---

## Local Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/farazn7/Yojna-Mitra-Project.git
cd Yojna-Mitra-Project
```

### 2. Install Dependencies

```bash
pip install -r configuration/requirements.txt
```

### 3. Initialize Local Vector & Checkpoint Store (Docker)

Ensure Docker Desktop is active and spin up a PostgreSQL container equipped with pgvector:

```bash
docker run --name pgvector-db -e POSTGRES_PASSWORD=mysecretpassword -p 5432:5432 -d pgvector/pgvector:pg16
```

*Note: When the application starts, LangGraph checkpointer tables and database profile schemas are automatically initialized.*

### 4. Setup Local Inference (Ollama)

Download and boot Ollama, then fetch your execution and reasoning models locally:

```bash
ollama pull hf.co/qwen/qwen3-8b-gguf:q4_k_m
ollama pull llama3.1
ollama pull nomic-embed-text
ollama pull gemma3:4b
```

### 5. Run the Automated Persona Evals

Execute the core testing pipeline to trace data integration from database retrieval to final markdown output generation:

```bash
pytest testing_&_diagnostics/test_hybrid_rag.py -v -s
```

### 6. Launch the Live Bot Interface

Create a local .env file in the root directory and add your secure Discord developer application token:

```env
DISCORD_TOKEN=your_bot_token_here
```

Execute your bot using Python's module flag (-m) from the root workspace directory to ensure relative packages resolve perfectly:

```bash
python -m product_inference.bot
```