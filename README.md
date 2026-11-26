<div align="center">

# 🎯 B2B Lead Scoring Analytics Engine

**A dynamic RevOps intelligence platform that recalculates live lead scores, simulates email decay, and generates executive PDF reports.**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Try_it_now-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://b2b-lead-scoring-rag-kazhjrkrfko5tzneybtycs.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Groq](https://img.shields.io/badge/LLM-Groq-F55036?style=for-the-badge)](https://groq.com/)

</div>

---

## 📌 Overview

Traditional dashboards show static lead scores. This application is a **live simulation engine**. 

Upload your raw CSVs (Accounts, Contacts, Email Events), or use our synthetic dataset, and the app will recalculate the entire scoring pipeline in real-time. 

You can toggle an **Email Decay** simulation (half-life) to see how engagement fades over time, and instantly see the impact on your MQLs and Priority MQLs.

### 🛡️ The "Safety Net" Architecture
This app solves the AI hallucination problem in high-stakes data environments:
1.  **100% Python Math:** All Fit Scores, Engagements, and Tiers are calculated in Pandas. The LLM never touches the numbers.
2.  **LLM as Storyteller:** The LLM receives the exact, calculated numbers and writes a 2-sentence insight and 1-sentence recommendation.
3.  **Numeric Guardrails:** A `storyteller.py` safety net verifies that every number the LLM writes exists in the calculated data. If it hallucinates a number, the LLM text is rejected, and the Python-computed text is used.

---

## ✨ Features

- **Dynamic Data Ingestion:** Upload your own CSVs or use the built-in synthetic dataset.
- **Email Decay Simulation:** Toggle a half-life decay on email events to see how it changes lead engagement over time.
- **Live Scoring Engine:** Recalculates Total Scores, Tiers, and Archetypes instantly based on your date and decay settings.
- **Executive PDF Export:** Generates a beautiful, multi-page PDF report using `reportlab` with charts, insights, and a top-10 priority account table.
- **LLM Insights:** Writes business-ready insights based on the exact Python-calculated numbers.

---

## 🚀 Quick Start

### Run Locally

```bash
git clone https://github.com/jhav5086-lab/B2B-Lead-Scoring-RAG.git
cd B2B-Lead-Scoring-RAG

pip install -r requirements.txt

mkdir -p .streamlit
echo 'GROQ_API_KEY = "your_key_here"' > .streamlit/secrets.toml

streamlit run app.py
