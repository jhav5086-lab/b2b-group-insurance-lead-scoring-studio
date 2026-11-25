import json
import os
import re
import time
import difflib
import tempfile
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
from fpdf import FPDF
from sentence_transformers import SentenceTransformer
from groq import Groq

# =============================================================================
# CONFIG & SAFETY CHECKS WITH COLAB USERDATA FALLBACK
# =============================================================================
st.set_page_config(
    page_title="RevOps Lead Scoring & RAG Studio",
    page_icon="🎯",
    layout="wide",
)

# Safely resolve secrets without throwing StreamlitSecretNotFoundError
GROQ_API_KEY = ""
try:
    if "GROQ_API_KEY" in st.secrets:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not GROQ_API_KEY:
    # Fallback to Colab userdata or Environment variable
    try:
        from google.colab import userdata
        GROQ_API_KEY = userdata.get("GROQ_API_KEY")
    except Exception:
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.warning("⚠️ **Missing API Key:** Set GROQ_API_KEY in Streamlit Secrets, environment variables, or Google Colab secrets to enable Chat.")

CHUNKS_PATH = "./embeddings/chunks.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
MIN_RAW_SIM = 0.10
HISTORY_TURNS = 2
HISTORY_MSG_CHARS = 600
MAX_TOKENS_DEFAULT = 1200
MAX_TOKENS_VISUAL = 2500

REFUSAL = "The provided model artifacts do not contain this information."
QUOTE_CHARS = "\"'\u201c\u201d\u2018\u2019"

# =============================================================================
# UNIFIED MODEL B PIPELINE RUNNER
# =============================================================================
def run_model_b_pipeline(accounts, contacts, emails, webinars, quotes, proposals) -> pd.DataFrame:
    EVAL = pd.Timestamp("2024-09-15")
    accounts = accounts.copy()
    contacts = contacts.copy()
    emails = emails.copy()
    webinars = webinars.copy()
    quotes = quotes.copy()
    proposals = proposals.copy()

    # Normalize columns
    for col in ["broker_id", "carrier", "industry", "industry_fit"]:
        if col in accounts.columns:
            accounts[col] = accounts[col].fillna("").astype(str)
        else:
            accounts[col] = ""

    for col in ["attended", "downloaded", "live", "minutes", "polls", "questions"]:
        if col in webinars.columns:
            webinars[col] = pd.to_numeric(webinars[col], errors="coerce").fillna(0).astype(int)
        else:
            webinars[col] = 0

    # Scoring configurations
    TIER_POINTS = {"generic": 2, "content": 5, "product": 8, "high_intent": 20}
    TIER_HL = {"generic": 60, "content": 60, "product": 45, "high_intent": 30}

    emails["points"] = emails["link_category"].map(TIER_POINTS).fillna(0)
    emails["hl"] = emails["link_category"].map(TIER_HL).fillna(60)
    emails["days_since"] = (EVAL - pd.to_datetime(emails["event_date"])).dt.days.clip(lower=0)
    emails["decayed"] = emails["points"] * (0.5 ** (emails["days_since"] / emails["hl"]))
    email_s = emails.groupby("contact_id")["decayed"].sum().clip(upper=50).rename("email_score")

    webinars["wpts"] = (
        webinars["attended"] * 5 +
        (webinars["minutes"].fillna(0) // 10).clip(upper=5) +
        webinars["polls"].fillna(0) * 2 +
        webinars["questions"].fillna(0) * 5 +
        webinars["downloaded"].fillna(0) * 5 +
        webinars["live"].fillna(0) * 3
    )
    web_s = webinars.groupby("contact_id")["wpts"].sum().clip(upper=30).rename("web_score")

    contacts = contacts.merge(email_s, on="contact_id", how="left").fillna({"email_score": 0})
    contacts = contacts.merge(web_s, on="contact_id", how="left").fillna({"web_score": 0})
    contacts["engagement"] = contacts["email_score"] + contacts["web_score"]
    contacts.loc[contacts["email_status"] != "Active", "engagement"] = 0

    rows = []
    for acc_id, g in contacts.groupby("account_id"):
        s = sorted(g["engagement"].tolist(), reverse=True)
        best = s[0] if s else 0
        others = sum(s[1:]) if len(s) > 1 else 0
        distinct = int((g["engagement"] > 0).sum())
        ms = 10 if distinct >= 2 else 0
        ae = round(best + 0.3 * others + ms, 2) if best > 0 else 0
        rows.append((acc_id, best, round(others, 2), distinct, ms, ae))

    acc_eng = pd.DataFrame(rows, columns=["account_id", "best_contact", "other_contacts", "distinct_engaged", "ms_bonus", "account_engagement"])
    broker_eng_map = dict(zip(acc_eng["account_id"], acc_eng["account_engagement"]))

    accounts = accounts.merge(acc_eng, on="account_id", how="left")
    for c in ["best_contact", "other_contacts", "distinct_engaged", "ms_bonus", "account_engagement"]:
        accounts[c] = accounts[c].fillna(0)

    accounts["broker_both"] = accounts.apply(
        lambda r: r["account_type"] == "Employer"
        and broker_eng_map.get(r["broker_id"], 0) >= 15
        and r["account_engagement"] >= 15,
        axis=1
    )
    accounts["broker_bonus"] = np.where(accounts["broker_both"], 15, 0)

    def fit_row(r):
        if r["account_type"] == "Broker":
            return 40
        d = (pd.to_datetime(r["renewal_date"]) - EVAL).days if pd.notna(r["renewal_date"]) else None
        if d is not None and d < 0:
            d += 365
        ren = 0 if d is None else (25 if d <= 30 else 20 if d <= 60 else 15 if d <= 90 else 5 if d <= 180 else 0)
        p = r["est_premium"]
        sz = 25 if p >= 2000000 else 20 if p >= 1000000 else 15 if p >= 500000 else 10 if p >= 100000 else 0
        ind = {"Target": 15, "Adjacent": 8, "Non-Target": 0}.get(r["industry_fit"], 0)
        car = 10 if r["carrier"] in ("Aetna", "UnitedHealth", "Cigna") else (5 if r["carrier"] else 0)
        brk = 15 if r["broker_id"] else 0
        return ren + sz + ind + car + brk

    accounts["fit_score"] = accounts.apply(fit_row, axis=1)
    accounts["days_to_renewal"] = accounts["renewal_date"].apply(lambda d: (pd.to_datetime(d) - EVAL).days if pd.notna(d) else np.nan)

    competitor_accs = set(contacts.loc[contacts["domain_type"] == "Competitor", "account_id"])
    accounts["disqualified"] = (accounts["account_type"] == "Employer") & ((accounts["lives"] < 50) | (accounts["account_id"].isin(competitor_accs)))
    accounts["composite_score"] = (0.4 * accounts["fit_score"] + 0.6 * accounts["account_engagement"] + 0.8 * accounts["broker_bonus"]).clip(1, 100)

    hi_ids = set(emails.loc[emails["link_category"] == "high_intent", "contact_id"])
    hi_accs = set(contacts.loc[contacts["contact_id"].isin(hi_ids), "account_id"])
    accounts["has_high_intent"] = accounts["account_id"].isin(hi_accs)

    def tier_row(r):
        if r["disqualified"]:
            return "Disqualified"
        if r["account_type"] == "Broker":
            return "Priority Broker" if r["composite_score"] >= 60 else ("Broker MQL" if r["composite_score"] >= 40 else "Responder")
        urgent = pd.notna(r["days_to_renewal"]) and r["days_to_renewal"] <= 90
        signal = urgent or r["distinct_engaged"] >= 2 or r["broker_both"]
        if r["composite_score"] >= 70 and r["est_premium"] >= 1000000 and r["account_engagement"] >= 15 and signal:
            return "Priority MQL"
        if r["composite_score"] >= 50:
            return "MQL"
        if r["account_engagement"] > 0:
            return "Responder"
        return "Disqualified"

    accounts["tier"] = accounts.apply(tier_row, axis=1)

    def arch_row(r):
        if r["account_type"] == "Broker":
            return "Broker"
        if r["disqualified"]:
            return "Detractor"
        fh = r["fit_score"] >= 60
        e = r["account_engagement"]
        multi = r["distinct_engaged"] >= 2
        if fh and e >= 40 and multi: return "Champion"
        if fh and e >= 40 and not multi: return "Achiever"
        if fh and 15 <= e < 40 and not r["has_high_intent"]: return "Watcher"
        if fh and e < 15: return "Dormant Giant"
        if 5 <= e < 15: return "Fader"
        return "Lurker"

    accounts["archetype"] = accounts.apply(arch_row, axis=1)

    q_summary = quotes.groupby("account_id").agg(quote_count=("quote_id", "count"), quoted_premium=("quoted_premium", "sum")).reset_index()
    p_summary = proposals.groupby("account_id").agg(proposal_count=("proposal_id", "count"), proposed_premium=("proposed_premium", "sum")).reset_index()

    accounts = accounts.merge(q_summary, on="account_id", how="left").merge(p_summary, on="account_id", how="left")
    for col in ["quote_count", "quoted_premium", "proposal_count", "proposed_premium"]:
        accounts[col] = accounts[col].fillna(0)

    accounts["engagement_score"] = accounts["account_engagement"]
    accounts["annual_premium_usd"] = accounts["est_premium"]
    accounts["sql_converted"] = np.where(accounts["proposal_count"] > 0, 1, 0)

    return accounts

# =============================================================================
# LOCAL FALLBACKS AND DATA SNAPSHOT HELPERS
# =============================================================================
@st.cache_data
def load_local_fallback_data():
    if os.path.exists("./data/accounts.csv"):
        accs = pd.read_csv("./data/accounts.csv")
        cont = pd.read_csv("./data/contacts.csv")
        em = pd.read_csv("./data/email_events.csv")
        web = pd.read_csv("./data/webinar_events.csv")
        quo = pd.read_csv("./data/quotes.csv")
        prop = pd.read_csv("./data/proposals.csv")
        return run_model_b_pipeline(accs, cont, em, web, quo, prop)
    return pd.DataFrame()

# =============================================================================
# RAG BACKEND
# =============================================================================
@st.cache_resource
def load_rag_backend():
    embedder = SentenceTransformer(EMBED_MODEL)
    chunks = []
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            chunks = json.load(f)
    else:
        chunks = [{"type": "fallback", "title": "General", "text": "No specialized lead scoring documents found locally."}]
    
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
    return embedder, embeddings, chunks, client

try:
    embedder, EMBEDDINGS, CHUNKS, groq_client = load_rag_backend()
    EMBEDDINGS_NORM = EMBEDDINGS / (np.linalg.norm(EMBEDDINGS, axis=1, keepdims=True) + 1e-9)
except Exception as e:
    st.error(f"RAG initialization failure: {e}")
    groq_client = None

# =============================================================================
# RAG UTILS
# =============================================================================
def retrieve(query, top_k=TOP_K):
    if not EMBEDDINGS_NORM is not None:
        return []
    q_emb = embedder.encode([query], convert_to_numpy=True)[0]
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    raw_sims = EMBEDDINGS_NORM @ q_emb
    top_idx = list(np.argsort(raw_sims)[::-1][:top_k])
    
    return [
        {
            "title": CHUNKS[i]["title"],
            "type": CHUNKS[i]["type"],
            "text": CHUNKS[i]["text"],
            "raw_score": float(raw_sims[i])
        }
        for i in top_idx
    ]

def build_dataset_summary_context(df):
    total_accts = len(df)
    avg_comp = df["composite_score"].mean() if "composite_score" in df.columns else 0.0
    total_prem = (df["annual_premium_usd"].sum() / 1e6) if "annual_premium_usd" in df.columns else 0.0
    sql_total = int(df["sql_converted"].sum()) if "sql_converted" in df.columns else 0
    return f"[Dataset Metrics] Total Accounts: {total_accts} | Avg Composite: {avg_comp:.1f} | Pipeline Premium: ${total_prem:.2f}M | Conversion SQLs: {sql_total}"

# =============================================================================
# STREAMLIT UI LAYOUT
# =============================================================================
st.title("🎯 RevOps Unified Lead Scoring Studio")
st.caption("Process complete relational pipeline datasets, track dynamic behaviors, and interact with the grounding bot.")

scored_df = load_local_fallback_data()
if not scored_df.empty:
    st.success("✅ Successfully loaded and processed dynamic RevOps pipeline!")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Accounts", f"{len(scored_df):,}")
    c2.metric("Avg Composite Score", f"{scored_df['composite_score'].mean():.1f}")
    c3.metric("Priority MQLs", f"{int((scored_df['tier'] == 'Priority MQL').sum()):,}")
    c4.metric("Total Annual Premium", f"${scored_df['annual_premium_usd'].sum() / 1e6:.2f}M")
    
    tab_table, tab_charts, tab_chat = st.tabs(["📋 Scored Accounts Table", "📊 Interactive Insights", "💬 RAG Assistant"])
    
    with tab_table:
        st.dataframe(scored_df.head(200), use_container_width=True)
        st.download_button(
            label="⬇️ Download Scored Accounts CSV",
            data=scored_df.to_csv(index=False).encode("utf-8"),
            file_name="scored_accounts_output.csv",
            mime="text/csv"
        )

    with tab_charts:
        st.subheader("Tier Distribution")
        tier_counts = scored_df["tier"].value_counts().reset_index()
        chart = alt.Chart(tier_counts).mark_bar().encode(
            x=alt.X("tier:N", title="Qualification Tier"),
            y=alt.Y("count:Q", title="Number of Accounts"),
            color="tier:N"
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

    with tab_chat:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_q := st.chat_input("Ask about MQL rules, account priority metrics, or custom scenarios..."):
            with st.chat_message("user"):
                st.markdown(user_q)
            st.session_state.messages.append({"role": "user", "content": user_q})
            
            if not groq_client:
                st.error("Chatbot is inactive due to missing GROQ_API_KEY.")
            else:
                retrieved = retrieve(user_q)
                retrieved_context = "\n\n".join([f"[{r['title']}]: {r['text']}" for r in retrieved])
                data_context = build_dataset_summary_context(scored_df)
                
                system_prompt = f"You are an artifact-grounded RevOps bot. Use the active data snapshot: {data_context} and context: {retrieved_context}. Cite your sources using [Chunk Title]. Keep replies succinct (2-4 sentences)."
                
                try:
                    response = groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_q}
                        ],
                        temperature=0.2
                    )
                    ans = response.choices[0].message.content
                    with st.chat_message("assistant"):
                        st.markdown(ans)
                        with st.expander("View Grounding Context"):
                            st.write(retrieved_context)
                    st.session_state.messages.append({"role": "assistant", "content": ans})
                except Exception as ex:
                    st.error(f"Inference Error: {ex}")
else:
    st.warning("⚠️ Waiting for pipeline execution data.")
