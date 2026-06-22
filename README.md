# Yojana Mitra Project

Yojana Mitra is an autonomous AI agent deployed on WhatsApp designed to discover Indian government welfare schemes a user qualifies for and automatically apply on their behalf. 

The system leverages natural language processing (Hindi/English), document retrieval via DigiLocker OAuth, and browser automation to navigate complex government portals, specifically targeting users in rural and semi-urban areas who face language and technical barriers.

##  Project Architecture & Tech Stack
* **Language:** Python 3.13
* **Automation & Scraping:** Playwright (Sync API)
* **Data Layer:** PostgreSQL with `pgvector` extension (Hybrid RAG)
* **Intelligence & Inference Engine:** Ollama (Local Node) running `llama3.1` (8B Execution Model) & `nomic-embed-text` (Vector Models)
* **Testing Infrastructure:** `pytest` (Automated Matrix Harness)
* **Interface & Messaging Layer:** `discord.py` (Secure asynchronous WebSocket interface managing multi-user DM isolation)

---

##  Engineering Milestones Achieved

### Phase 1: Data Engineering & Extraction
Developed a highly resilient, Playwright-based spider to navigate the dynamic Next.js/React architecture of `myScheme.gov.in`.
* **React State Hydration Bypassing:** Utilized a "Green Button" DOM-polling strategy to ensure pagination only triggered after React fully painted new page elements, preventing stale-element duplication.
* **Idempotency & State Persistence:** Implemented a two-file architecture (`schemes.json` and `progress.json`). The script saves progress after every 10 extractions, allowing it to seamlessly resume from crashes without duplicating data.
* **Anti-Bot Evasion:** Integrated randomized politeness delays to avoid triggering HTTP 429 server blocks, and recycled browser contexts to prevent memory bloat.
* **Hidden DOM Extraction:** Bypassed "Strict Mode" layout hiding by targeting parent container IDs (`#eligibility`, `#documents-required`) directly instead of triggering unstable visible UI tabs.

**Result:** Extracted eligibility criteria, required documents, and URLs for **230 unique State/UT schemes in Tamil Nadu** into a structured reference database.

### Phase 2: Hybrid RAG Architecture & Persona Evaluation
We initially explored a legacy keyword heuristic engine (`test_matcher.py` / `matcher.py`) utilizing regular expressions to filter attributes. However, classical NLP proved unscalable for highly nuanced legislative texts (e.g., matching the phrase *"The boy should be 21"* inside a female marriage assistance scheme with a 21-year-old male profile due to raw keyword collision).

To eliminate semantic blindspots, the backend was migrated to a robust **Hybrid Retrieval-Augmented Generation (RAG)** pipeline:
1. **The Hard Filter Layer:** Structured profile columns extracted via `extract_schema_llm.py` are evaluated through high-speed SQL `WHERE` clauses to instantly filter rigid numerical bounds (e.g., bounding user parameters inside `min_age`, `max_age`, or checking `max_income`).
2. **The Semantic Ranking Layer:** Remaining candidate schemes undergo a `pgvector` cosine similarity search (`<=>`) matching user chat intent with multi-dimensional document chunks.
3. **Contextual Evaluation Engine:** Built an automated testing suite (`test_hybrid_rag.py`) parametrized across 10 distinct, structured citizen personas (e.g., `p1_widowed_farmer`, `p2_rural_student`). The generation routine utilizes a strict system configuration to force total fact synthesis, manage situational sensitivity constraints, and handle `null` financial values cleanly while preserving system performance using proactive thermal throttling.

### Phase 3: Conversational State Machine & Live Runtime Routing
Transitioned the architecture from a static testing harness to a live, production-ready interactive interface.
* **13-Step Conversational Onboarding:** Built a progressive state machine inside `bot.py` to sequentially capture and validate comprehensive user profiles (Gender, Caste, Minority Status, BPL records, etc.), eliminating structural data omission gaps.
* **Dynamic Database Schema Evolution:** Upgraded the PostgreSQL layer via `db.py` to utilize a native `JSONB` data type combined with atomic appends (`||`). Replaced rigid boolean architectures with a flexible `occupation` string identifier to prevent hardcoded keyword constraints.
* **Discord API Guardrails & Chunking:** Implemented an aggressive streaming chunking algorithm at the presentation layer to split LLM context blocks at newline boundaries if payloads exceed 1,900 characters, bypassing Discord's 2,000-character truncation errors.
* **Live RAG Context Injection:** Configured the production message loop to dynamically pull the user's explicit profile out of PostgreSQL at runtime based on their unique, unchangeable Snowflake User ID, feeding those verified facts straight into the Llama 3.1 evaluation prompt.
---

##  Repository Structure

* **Data Extraction Layer:**
  * `scraping.py`: Playwright web crawler targeting web portals.
  * `extract_schema_llm.py`: Transforms raw scraped text into structured, queryable database elements.
  * `schemes.json`: Raw structural data deliverable containing scraped parameters.
* **Core Inference & Vector Layer:**
  * `hybrid_rag.py`: Master search query orchestration engine.
  * `inspect_db.py`: Database health utility checking table configurations and vector index validity.
* **Testing & Diagnostic Logs:**
  * `test_hybrid_rag.py`: Automated `pytest` suite simulating live user interactions across all 10 custom baseline templates.
  * `rag_evaluation_log.md`: Markdown audit tracker documenting model answers and precision metrics.
  * `personas/`: Dynamic local user data profile templates (`personas.json`).
  * `match_results.json` & `test.py`: Runtime configuration and temporary scratch files.
* **Configuration:**
  * `user_profile_schema.json`: System design blueprints defining parameters for user object fields.
  * `progress.json`: State-persistence marker for web tracking stability.
  * `requirements.txt`: Master project dependency manifest.
* **Production Interface Layer:**
  * `bot.py`: The main runtime gateway managing multi-user asynchronous event loops, DM routing, and the onboarding state machine.
  * `db.py`: Data persistence module responsible for initializing relational configurations, updating chat states, and merging profile JSONB payloads.

---

##  Local Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/farazn7/Yojna-Mitra-Project.git
cd Yojna-Mitra-Project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Initialize Local Vector Store (Docker)
Ensure Docker is active and spin up a PostgreSQL container equipped with `pgvector`:
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
pytest test_hybrid_rag.py -v -s
```
*(Note: A 4-second internal cooldown clock is intentionally enforced at the end of each persona cycle to allow your local CPU core temperatures to return to resting state before computing subsequent embeddings).*

### 6. Launch the Live Bot Interface
Create a local `.env` file in the root directory and add your secure Discord developer application token:
```env
DISCORD_TOKEN=your_bot_token_here
```

Boot the interface engine to initialize the database tables and establish the secure gateway connection:
```bash
py bot.py
```
(Once active, users can message the bot via Direct Messages to clear the onboarding state sequence and run live vector-filtering scheme evaluations).

