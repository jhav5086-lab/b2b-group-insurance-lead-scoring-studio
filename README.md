 <div align="center">

# 🎯 B2B Lead Scoring Analytics Studio

**A dynamic RevOps intelligence platform that recalculates live lead scores, simulates email decay, and generates executive PDF reports.**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Try_it_now-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://b2b-group-insurance-lead-scoring-studio-59droaudpjfwfuf586mbmz.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Groq](https://img.shields.io/badge/LLM-Groq-F55036?style=for-the-badge)](https://groq.com/)

</div>

---

## 📌 Overview

Traditional dashboards show static lead scores. This application is a **live simulation engine**.

Upload your six CSV datasets (**Accounts, Contacts, Email Events, Webinar Events, Quotes, and Proposals**) or use the built-in synthetic dataset. Accounts, Contacts, and Email Events are the required core files; Webinar Events, Quotes, and Proposals are optional. The app recalculates the scoring pipeline in real time using Pandas.

You can toggle an **Email Decay** simulation (half-life) to see how email engagement fades over time, and instantly see the impact on your MQLs and Priority MQLs.

### 🛡️ The "Safety Net" Architecture

This app solves the AI hallucination problem in high-stakes data environments:

1. **100% Python Math:** All Fit Scores, Engagements, and Tiers are calculated deterministically in Python. The LLM never touches the numbers.
2. **LLM as Storyteller:** The LLM receives the exact, calculated numbers and writes a 2-sentence insight and 1-sentence recommendation.
3. **Numeric Guardrails:** A custom `storyteller.py` safety net verifies that *every number* the LLM writes exists in the calculated data. If it hallucinates a number, the text is rejected, and the Python-computed text is used.

---

## ✨ Features

- **Dynamic Data Ingestion:** Upload all six supported CSV datasets or use the default synthetic dataset. Accounts, Contacts, and Email Events are required together; Webinar Events, Quotes, and Proposals are optional.
- **Email Decay Simulation:** Toggle a half-life decay on email events to see how it changes lead engagement over time.
- **Live Scoring Engine:** Recalculates Total Scores, Tiers, and Archetypes instantly based on your date and decay settings.
- **Executive PDF Export:** Generates a beautiful, multi-page PDF report using `reportlab` with charts, insights, a decay simulation table, and a Top 10 priority account list.
- **LLM Insights:** Writes business-ready insights based on the exact Python-calculated numbers.

---

## 🏗️ Architecture

The repository is split into specialized modules for production-grade reliability:

- `app.py`: Streamlit UI and state management.
- `scoring.py`: Pandas data pipeline (Fit, Engagement, Tiers, Archetypes).
- `analytics.py`: Chart generation and summary data prep.
- `storyteller.py`: The LLM safety net (verifies numbers).
- `pdf_generator.py`: ReportLab engine for the executive PDF.

---

## 🚀 Quick Start

### Run Locally

```bash
git clone https://github.com/jhav5086-lab/b2b-group-insurance-lead-scoring-studio.git
cd b2b-group-insurance-lead-scoring-studio

pip install -r requirements.txt

mkdir -p .streamlit
echo 'GROQ_API_KEY = "your_key_here"' > .streamlit/secrets.toml

streamlit run app.py
```
