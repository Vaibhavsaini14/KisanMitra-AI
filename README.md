# 🌾 KisanMitra AI
### Multilingual Crop Advisory Chatbot for Rajasthan Farmers

[![Built with Claude](https://img.shields.io/badge/LLM-Claude%20(Anthropic)-orange)](https://anthropic.com)
[![IBM Granite](https://img.shields.io/badge/AI-IBM%20Granite%20%7C%20Watsonx.ai-blue)](https://watsonx.ai)
[![RAG](https://img.shields.io/badge/Architecture-RAG%20%2B%20FAISS-green)](https://github.com)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)](https://streamlit.io)
[![SDG 2](https://img.shields.io/badge/SDG-2%20Zero%20Hunger-brightgreen)](https://sdgs.un.org)
[![SDG 13](https://img.shields.io/badge/SDG-13%20Climate%20Action-green)](https://sdgs.un.org)
[![SDG 15](https://img.shields.io/badge/SDG-15%20Life%20on%20Land-darkgreen)](https://sdgs.un.org)

> Built for: **1M1B AI for Sustainability Virtual Internship** (AICTE-Recognized)  
> Author: **Vaibhav Saini** | B.E. CSE | MBM University, Jodhpur, Rajasthan

---

## Problem Statement

Rajasthan has **2.3 million small and marginal farmers** facing compounding crises:

| Challenge | Reality |
|-----------|---------|
| Advisory Access | Nearest KVK is often 50+ km away |
| Language Barrier | 80% of farmers cannot read English advisories |
| Crop Loss | 30-40% annual loss from preventable pest/disease |
| Water Scarcity | Jodhpur belt gets <250mm rainfall — critically over-irrigated |
| Information Gap | No real-time, localized AI advisory exists |

**KisanMitra AI** solves this using a RAG-powered multilingual chatbot accessible via web browser.

---

## Features

- **Multilingual Support** — Hindi + English, farmer-friendly language
- **RAG Architecture** — Answers from verified ICAR + Rajasthan Agri Dept. sources, not hallucinations
- **District-Wise Water Advisory** — Different water guidance for Jodhpur (critical arid) vs Hanumangarh (canal zone)
- **Structured Advisory Cards** — Diagnosis · Immediate Action · Water Advisory · Prevention · Organic Alternative
- **ICAR Variety Recommendations** — Suggests resistant crop varieties per district
- **Responsible AI** — Never recommends banned pesticides, always provides organic alternative

---

## Architecture

```
Farmer Query (Hindi/English)
        ↓
Entity Extraction (Crop · District · Season · Problem)
        ↓
RAG Retrieval — FAISS Vector Search
        ↓  (retrieves from ICAR + Rajasthan Agri Dept. knowledge base)
Prompt Construction (context + profile + language)
        ↓
IBM Granite / Claude LLM (via Watsonx.ai / Anthropic API)
        ↓
Structured Advisory Response
(Diagnosis · Action · Water · Prevention · Organic · Variety)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| LLM | Claude (claude-sonnet-4-6) via Anthropic API |
| Agent Framework | IBM BeeAI-style prompt engineering + RAG |
| Vector DB | FAISS (faiss-cpu) |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Knowledge Base | ICAR Bulletins · Rajasthan Agriculture Dept. · CAZRI Jodhpur |
| Language | Python 3.10+ |

---

## SDG Alignment

| SDG | How KisanMitra Contributes |
|-----|--------------------------|
| **SDG 2: Zero Hunger** | Reduces crop loss through timely disease/pest advisory |
| **SDG 13: Climate Action** | Promotes water-efficient, climate-resilient farming in arid Rajasthan |
| **SDG 15: Life on Land** | Recommends organic alternatives reducing chemical land degradation |

---

## Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/Vaibhavsaini14/kisanmitra-ai.git
cd kisanmitra-ai
```

### 2. Create virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

For full FAISS RAG support (optional):
```bash
pip install faiss-cpu sentence-transformers
```

### 4. Set API key
Create a `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

### 5. Run the app
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

---

## Project Structure

```
kisanmitra-ai/
├── app.py           # Streamlit frontend — chat UI, sidebar controls
├── rag_engine.py    # FAISS RAG + Rajasthan crop knowledge base (13 entries)
├── advisor.py       # Claude-powered advisory generation with structured output
├── requirements.txt # Dependencies
├── .gitignore
└── README.md
```

---

## Knowledge Base Coverage

| Crop | Problems Covered |
|------|-----------------|
| Bajra (Pearl Millet) | Downy Mildew, Ergot, Shoot Fly, Stem Borer |
| Mustard | Aphids, White Rust, Alternaria Blight |
| Wheat | Yellow Rust, Loose Smut, Powdery Mildew |
| Cumin (Jeera) | Fusarium Wilt, Blight, Powdery Mildew |
| Moong / Moth Bean | Yellow Mosaic Virus, Cercospora Leaf Spot |
| Groundnut | Tikka Leaf Spot, Collar Rot |
| Cotton | American Bollworm, Whitefly, Leaf Curl Virus |
| All Crops | Water management by district zone, Organic farming methods |

---

## Responsible AI

- No personal farmer data is stored or logged
- Never recommends WHO Class I hazardous pesticides
- Every response cites verified ICAR/government sources
- Organic alternatives always included
- Simple language — accessible to farmers with basic literacy

---

## Internship Context

This project was built as part of the **1M1B AI for Sustainability Virtual Internship** (May-June 2026), conducted in collaboration with IBM SkillsBuild and recognized by AICTE under the National Internship Portal.

**Mentor Organization:** 1 Million for 1 Billion (1M1B)  
**IBM Technology Used:** IBM Granite models (Watsonx.ai), IBM SkillsBuild AI curriculum  
**Certificate:** To be issued July 2026 upon successful completion

---

*"Technology should serve those who need it most — not just those who can access it most."*
