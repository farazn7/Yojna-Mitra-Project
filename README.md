# Yojana Mitra Project

Yojana Mitra is an autonomous AI agent deployed on WhatsApp designed to discover Indian government welfare schemes a user qualifies for and automatically apply on their behalf. 

The system leverages natural language processing (Hindi/English), document retrieval via DigiLocker OAuth, and browser automation to navigate complex government portals, specifically targeting users in rural and semi-urban areas who face language and technical barriers.

## 🛠️ Project Architecture & Tech Stack
* **Language:** Python 3.13
* **Automation & Scraping:** Playwright (Sync API)
* **Data Layer:** RAG (Retrieval-Augmented Generation) corpus built from `myScheme.gov.in` (JSON/Vector DB)
* **Orchestration (Upcoming):** Claude/GPT-4, LangChain/LangGraph
* **Integration (Upcoming):** WhatsApp Business API (or Gupshup), DigiLocker OAuth, Bhashini/Sarvam (Vernacular Voice STT/TTS)

## 🏁 Current Status: Week 1 Milestone (Completed)
**Objective:** Data Engineering & Extraction

Successfully built the foundation of the RAG corpus. Developed a highly resilient, Playwright-based spider to navigate the dynamic Next.js/React architecture of `myScheme.gov.in`. 

**Key Engineering Achievements in Phase 1:**
* **React State Hydration Bypassing:** Utilized a "Green Button" DOM-polling strategy to ensure pagination only triggered after React fully painted new page elements, preventing stale-element duplication.
* **Idempotency & State Persistence:** Implemented a two-file architecture (`schemes.json` and `progress.json`). The script saves progress after every 10 extractions, allowing it to seamlessly resume from crashes without duplicating data.
* **Anti-Bot & Rate Limit Evasion:** Integrated randomized "Politeness Delays" to avoid triggering HTTP 429 (Too Many Requests) server bans, and recycled browser contexts to prevent memory bloat.
* **Hidden DOM Extraction:** Bypassed "Strict Mode" errors by directly targeting parent container IDs (`#eligibility`, `#documents-required`) instead of relying on visible UI tabs.

**Result:** Extracted eligibility criteria, required documents, and URLs for **230 unique State/UT schemes in Tamil Nadu** into a structured `schemes.json` database.

## 📂 Repository Structure
* `scraping.py`: The master crawler and extraction script.
* `schemes.json`: The final database deliverable containing the extracted rules, documents, and URLs.
* `progress.json`: The local state-tracker file (dynamically generated during runs).

## ⚙️ Local Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/farazn7/Yojna-Mitra-Project.git](https://github.com/farazn7/Yojna-Mitra-Project.git)
cd Yojna-Mitra-Project
```

### 2. Install Dependencies
```bash
pip install playwright
playwright install chromium
```

### 3. Execute the Pipeline
```bash
python scraping.py
```
*Note: Phase 1 (URL Crawling) takes approximately 2 minutes. Phase 2 (Extraction) utilizes intentional politeness delays and takes about 20-30 minutes to safely complete a full state database.*

## 🔮 Next Steps (Phase 2)
With the ground-truth database constructed, the next phase involves migrating the development environment to Kaggle to leverage cloud GPUs. We will ingest `schemes.json` into a LangChain/RAG pipeline to build the intelligent matching engine and develop the core conversational AI logic.