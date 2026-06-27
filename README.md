# Yojana Mitra Project

Yojana Mitra is an autonomous, multi-user AI agent deployed on Discord designed to discover Indian government welfare schemes a citizen qualifies for and guide them through eligibility evaluation.

The system leverages natural language processing (Hindi, English, and Hinglish), advanced local Hybrid RAG pipelines, and automated document-schema extraction to navigate complex welfare criteria, specifically targeting users who face language, literacy, and technical barriers.

---

## Project Architecture & Tech Stack

* **Core Language:** Python 3.13
* **Automation & Scraping:** Playwright (Sync API)
* **Data Layer:** PostgreSQL with pgvector extension (Hosted inside a Docker Sandbox Container)
* **Intelligence & Local Inference Engine:** Ollama running llama3.1 (8B Execution Model) & nomic-embed-text (High-Dimensional Vector Model)
* **Testing Infrastructure:** pytest (Parametrized Automation Matrix Harness)
* **Interface & Presentation Layer:** bot.py (Multi-user isolated DM gateway with an Asynchronous, Non-Blocking Event Loop)

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

---

## Repository Structure

```text
Yojna-Mitra-Project/
│
├── core_inference/
│   ├── hybrid_rag.py
│   ├── inspect_db.py
│   └── extract_schema_llm.py
│
├── product_inference/
│   ├── bot.py
│   └── db.py
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

### 3. Initialize Local Vector Store (Docker)

Ensure Docker Desktop is active and spin up a PostgreSQL container equipped with pgvector:

```bash
docker run --name pgvector-db -e POSTGRES_PASSWORD=mysecretpassword -p 5432:5432 -d pgvector/pgvector:pg16
```

### 4. Setup Local Inference (Ollama)

Download and boot Ollama, then fetch your execution models locally:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
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