import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional dependencies: the scoring application still works if the RAG
# dependencies are unavailable. Install them through requirements.txt to enable chat.
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from groq import Groq
except Exception:
    Groq = None


# =============================================================================
# APP CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="RevOps Lead Scoring & RAG Studio",
    page_icon="🎯",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR_CANDIDATES = [APP_DIR / "data", APP_DIR]
CHUNKS_PATH = APP_DIR / "embeddings" / "chunks.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
MIN_RAW_SIM = 0.10
MAX_TOKENS_DEFAULT = 1200

# Keep this evaluation date aligned with the original Model B implementation.
# Change it deliberately when you want to score against a different snapshot date.
EVALUATION_DATE = pd.Timestamp("2024-09-15")

DATA_FILES = {
    "accounts": "accounts.csv",
    "contacts": "contacts.csv",
    "emails": "email_events.csv",
    "webinars": "webinar_events.csv",
    "quotes": "quotes.csv",
    "proposals": "proposals.csv",
}

REQUIRED_COLUMNS = {
    "accounts": {
        "account_id", "account_name", "account_type", "industry", "industry_fit",
        "lives", "est_premium", "carrier", "broker_id", "renewal_date",
    },
    "contacts": {
        "contact_id", "account_id", "email", "first_name", "last_name",
        "persona", "email_status", "domain_type",
    },
    "emails": {"event_id", "contact_id", "link_category", "event_date"},
    "webinars": {
        "webinar_id", "contact_id", "attended", "minutes", "polls",
        "questions", "downloaded", "live",
    },
    "quotes": {
        "quote_id", "account_id", "product_code", "quoted_premium",
        "quote_date", "status",
    },
    "proposals": {
        "proposal_id", "account_id", "proposal_date", "proposed_premium",
    },
}

REFUSAL = "The provided model artifacts do not contain this information."


# =============================================================================
# API KEY
# =============================================================================
def get_groq_api_key() -> str:
    """Read Groq key from Streamlit secrets or environment variables."""
    try:
        value = st.secrets.get("GROQ_API_KEY", "")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "").strip()


GROQ_API_KEY = get_groq_api_key()

