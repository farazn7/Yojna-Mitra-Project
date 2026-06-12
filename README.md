# Yojana Mitra Project

Yojana Mitra is an autonomous AI agent deployed on WhatsApp designed to discover Indian government welfare schemes a user qualifies for and automatically apply on their behalf. 

The system leverages natural language processing (Hindi/English), document retrieval via DigiLocker OAuth, and browser automation to navigate complex government portals, specifically targeting users in rural and semi-urban areas who face language and technical barriers.

##  Project Architecture & Tech Stack
* **Language:** Python 3.13
* **Automation & Scraping:** Playwright
* **Data Layer:** RAG (Retrieval-Augmented Generation) corpus built from `myScheme.gov.in`
* **Orchestration (Upcoming):** Claude/GPT-4, LangChain/LangGraph
* **Integration (Upcoming):** WhatsApp Business API (or Gupshup), DigiLocker OAuth, Bhashini/Sarvam (Vernacular Voice STT/TTS)

##  Current Status: Week 1 Milestone
**Objective:** Data Engineering & Extraction
Currently building the foundation of the RAG corpus. Developing a Playwright-based spider to navigate the Next.js/React architecture of `myScheme.gov.in`, scrape eligibility criteria and required documents for 234 State/UT schemes in Tamil Nadu, and compile them into a structured `schemes.json` schema.

##  Local Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/farazn7/Yojna-Mitra-Project.git
cd yojana-mitra
