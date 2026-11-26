<div align="center">

# 🎯 B2B Lead Scoring Analytics Studio

**A dynamic RevOps intelligence platform that recalculates live lead scores, simulates email decay, and generates executive PDF reports.**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Try_it_now-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://b2b-group-insurance-lead-scoring-studio-59droaudpjfwfuf586mbmz.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Groq](https://img.shields.io/badge/LLM-Groq-F55036?style=for-the-badge)](https://groq.com/)

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [The Safety Net Architecture](#️-the-safety-net-architecture)
- [Features](#-features)
- [Data Inputs](#-data-inputs)
- [Architecture](#️-architecture)
- [Quick Start](#-quick-start)
- [Configuration](#️-configuration)
- [Testing](#-testing)
- [Tech Stack](#-tech-stack)

---

## 📌 Overview

Traditional dashboards show static lead scores. This application is a **live simulation engine** built for B2B group insurance sales and RevOps teams.

Upload your six CSV datasets or use the built-in synthetic dataset, and the app recalculates the full scoring pipeline in real time using Pandas. Toggle an **Email Decay** simulation (half-life) to see how email engagement fades over time, and instantly see the impact on your **MQLs** and **Priority MQLs**.

---

## 🛡️ The "Safety Net" Architecture

This app tackles the AI hallucination problem in high-stakes data environments:

1. **100% Python Math:** All Fit Scores, Engagement Scores, and Tiers are calculated deterministically in Python. The LLM never touches the numbers.
2. **LLM as Storyteller:** The LLM receives the exact, calculated numbers and writes a 2-sentence insight and a 1-sentence recommendation.
3. **Numeric Guardrails:** `storyteller.py` verifies that *every number* the LLM writes exists in the calculated data. If the LLM hallucinates a number, its text is rejected and the Python-generated fallback text is used instead.

```
CSV data ──► scoring.py (deterministic) ──► calculated numbers
                                                  │
                                                  ▼
                                   LLM (Groq) writes the narrative
                                                  │
                                                  ▼
                          storyteller.py checks every number in the text
                                   │                       │
                              all match                any mismatch
                                   ▼                       ▼
                           LLM text is used      Python fallback text is used
```

---

## ✨ Features

- **Dynamic Data Ingestion:** Upload up to six CSV datasets or use the default synthetic dataset.
- **Email Decay Simulation:** Toggle a half-life decay on email events to see how it changes lead engagement over time, with a side-by-side decay comparison.
- **Live Scoring Engine:** Recalculates Total Scores, Tiers, and Archetypes instantly based on your date and decay settings.
- **Diagnostic Charts:** Visual breakdowns of the scored pipeline to spot where leads concentrate and why.
- **LLM Insights:** Business-ready insights written from the exact Python-calculated numbers, protected by the numeric guardrail.
- **Executive PDF Export:** A multi-page report (built with `reportlab`) containing charts, insights, a decay simulation table, and a Top 10 priority account list.

---

## 📂 Data Inputs

| Dataset | Status |
|---|---|
| Accounts | **Required** |
| Contacts | **Required** |
| Email Events | **Required** |
| Webinar Events | Optional |
| Quotes | Optional |
| Proposals | Optional |

Accounts, Contacts, and Email Events must be provided together. Webinar Events, Quotes, and Proposals are optional and enrich the scoring when present.

Not sure about the expected columns? Check the sample CSV files in the [`data/`](data/) folder, which show the supported schema.

---

## 🏗️ Architecture

The repository is split into specialized modules:

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI and state management |
| `data_io.py` | CSV loading and input handling |
| `scoring.py` | Pandas data pipeline (Fit, Engagement, Tiers, Archetypes) |
| `analytics.py` | Chart generation, summary data prep, decay comparison |
| `storyteller.py` | LLM narration plus the numeric-verification safety net |
| `pdf_generator.py` | ReportLab engine for the executive PDF |
| `test_data_pipeline.py` | Tests for the data pipeline |
| `data/` | Sample B2B accounts and related CSV files |

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

Then open the local URL Streamlit prints (usually http://localhost:8501).

---

## ⚙️ Configuration

| Setting | Where | Purpose |
|---|---|---|
| `GROQ_API_KEY` | `.streamlit/secrets.toml` (local) or the Streamlit Cloud **Secrets** panel (deployed) | Enables LLM insights |

> 🔐 **Never commit your API key.** Add `.streamlit/secrets.toml` to your `.gitignore`.

If no key is configured, the Python-computed text can still be used for insights, since all numbers are produced by the deterministic pipeline.

---

## 🧪 Testing

```bash
python -m pytest test_data_pipeline.py
```

---

## 🧰 Tech Stack

- **UI:** Streamlit
- **Data:** Pandas
- **LLM:** Groq (narration only)
- **Reporting:** ReportLab
- **Language:** Python 3.10+
