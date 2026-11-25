# B2B Group Insurance Lead Scoring & Guarded RAG Studio

A deterministic RevOps scoring and guarded Retrieval-Augmented Generation (RAG) application for B2B group insurance account prioritization, diagnostic visualization, PDF reporting, and artifact-grounded Q&A.

## Overview

Lead scoring and account prioritization logic often sits inside black-box spreadsheets or static dashboards. This studio pairs:
- **Deterministic scoring in Python** (`0.4 * fit_score + 0.6 * engagement_score + 0.8 * broker_bonus`, qualification tiers, and behavioral archetypes)
- **Custom CSV upload with synthetic fallback** (upload your own account CSV or run immediately on the deterministic default synthetic dataset, plus download a ready-to-use sample CSV template)
- **Diagnostic charts, interpretations, and PDF export** (5 interactive Altair charts in the UI with data-backed interpretations, recommended actions, and a downloadable multi-page Matplotlib + FPDF report)
- **Guarded RAG pipeline** (semantic retrieval over `embeddings/chunks.json`, numeric and citation verification, false-premise pushback, hypothetical math guardrails, and an evaluation test suite in `eval_cases.json`)

## Repository Structure

```text
revops-lead-scoring-rag-studio/
├── app.py
├── requirements.txt
├── eval_cases.json
├── README.md
├── data/
│   └── sample_b2b_accounts.csv
└── embeddings/
    └── chunks.json
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure Streamlit secrets (`.streamlit/secrets.toml` or Streamlit Community Cloud Secrets):
   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Model Caveats

- Composite scores, qualification tiers, and behavioral archetypes reflect rule-based alignment and engagement signals—they do not guarantee conversion or closed-won revenue.
- Factual model answers are constrained to `embeddings/chunks.json` and the active dataset summary, with post-generation checks for numbers and source citations.
