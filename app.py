%%writefile app.py
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
# STREAMLIT CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="B2B Insurance Scoring & RevOps Studio",
    page_icon="🎯",
    layout="wide"
)

# Safely resolve secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.warning("⚠️ **Missing API Key:** Set GROQ_API_KEY to ask questions in the RAG Chatbot.")

# Fallback paths for RAG chunks
CHUNKS_PATH = "./embeddings/chunks.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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
        (webinars["minutes"] // 10).clip(upper=5) +
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
    
    # Merge Quote / Proposal metrics
    q_summary = quotes.groupby("account_id").agg(quote_count=("quote_id", "count"), quoted_premium=("quoted_premium", "sum")).reset_index()
    p_summary = proposals.groupby("account_id").agg(proposal_count=("proposal_id", "count"), proposed_premium=("proposed_premium", "sum")).reset_index()
    
    accounts = accounts.merge(q_summary, on="account_id", how="left").merge(p_summary, on="account_id", how="left")
    for col in ["quote_count", "quoted_premium", "proposal_count", "proposed_premium"]:
        accounts[col] = accounts[col].fillna(0)
        
    # Metric mapping for UI compatibility
    accounts["engagement_score"] = accounts["account_engagement"]
    accounts["annual_premium_usd"] = accounts["est_premium"]
    accounts["sql_converted"] = np.where(accounts["proposal_count"] > 0, 1, 0)
    
    return accounts

# =============================================================================
# LOCAL FALLBACKS AND SYNTHETIC GENERATORS
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
# DIAGNOSTIC CHARTS & INTERPRETATIONS
# =============================================================================
def generate_chart_specs(df: pd.DataFrame) -> list:
    total = len(df)
    tier_order = ["Priority MQL", "MQL", "Responder", "Disqualified"]
    tier_counts = df["tier"].value_counts().reindex(tier_order, fill_value=0)
    tier_sql = df.groupby("tier")["sql_converted"].sum().reindex(tier_order, fill_value=0)
    arch_counts = df["archetype"].value_counts()
    
    return [
        {
            "key": "tier_dist",
            "title": "Account Volume by Qualification Tier",
            "x_label": "Qualification Tier",
            "y_label": "Number of Accounts",
            "chart_type": "bar",
            "data": pd.DataFrame({"Qualification Tier": tier_counts.index, "Number of Accounts": tier_counts.values}),
            "interpretation": f"The scoring model clusters accounts cleanly, with priority targets totaling {tier_counts.get('Priority MQL', 0)} Priority MQLs.",
            "recommendation": "Engage priority cohorts with custom executive briefing playbooks."
        },
        {
            "key": "sql_by_tier",
            "title": "Proposal SQL conversions by Tier",
            "x_label": "Qualification Tier",
            "y_label": "SQL Count",
            "chart_type": "bar",
            "data": pd.DataFrame({"Qualification Tier": tier_sql.index, "SQL Count": tier_sql.values}),
            "interpretation": f"Most opportunities converting to Proposal SQL stage are captured in high-priority tiers.",
            "recommendation": "Optimize response SLAs on high-potential accounts."
        },
        {
            "key": "archetype_dist",
            "title": "Behavioral Archetype Segments",
            "x_label": "Account Archetype",
            "y_label": "Number of Accounts",
            "chart_type": "bar",
            "data": pd.DataFrame({"Account Archetype": arch_counts.index, "Number of Accounts": arch_counts.values}),
            "interpretation": f"Your largest behavioral block is {arch_counts.idxmax() if not arch_counts.empty else 'Lurker'}.",
            "recommendation": "Activate high-fit segment groups with renewal alerts."
        }
    ]

# =============================================================================
# RAG CHATBOT SERVICE
# =============================================================================
@st.cache_resource
def load_embeddings_and_model():
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            chunks = json.load(f)
        embedder = SentenceTransformer(EMBED_MODEL)
        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode(texts, convert_to_numpy=True)
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        return embedder, embeddings_norm, chunks
    return None, None, []

embedder, EMBEDDINGS_NORM, CHUNKS = load_embeddings_and_model()

def get_grounded_response(question, history_context=""):
    if not GROQ_API_KEY:
        return "Groq API Key missing. Please provide GROQ_API_KEY in Streamlit secrets or environment.", []
    if not CHUNKS:
        return "No chunk artifacts found. Please verify chunks.json is generated.", []
        
    q_emb = embedder.encode([question], convert_to_numpy=True)[0]
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    similarities = EMBEDDINGS_NORM @ q_emb
    top_indices = np.argsort(similarities)[::-1][:3]
    
    sources = [CHUNKS[idx] for idx in top_indices if similarities[idx] > 0.1]
    context_str = "\n\n".join([f"Source: {s['title']}\n{s['text']}" for s in sources])
    
    client = Groq(api_key=GROQ_API_KEY)
    system_prompt = "You are an advanced B2B RevOps copilot. Use the context and dataset details to answer. Be factual and clear."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}"}
    ]
    
    try:
        res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages, temperature=0.2, max_tokens=800)
        return res.choices[0].message.content, sources
    except Exception as e:
        return f"Error generating response: {str(e)}", []

# =============================================================================
# STREAMLIT UI LAYOUT
# =============================================================================
st.title("🎯 RevOps Unified Lead Scoring Studio")
st.caption("Process complete relational pipeline datasets, track dynamic behaviors, and interact with the grounding bot.")

# File Upload Section
with st.sidebar:
    st.header("📂 Upload Pipeline Datasets")
    f_acc = st.file_uploader("accounts.csv", type=["csv"])
    f_con = st.file_uploader("contacts.csv", type=["csv"])
    f_em = st.file_uploader("email_events.csv", type=["csv"])
    f_web = st.file_uploader("webinar_events.csv", type=["csv"])
    f_q = st.file_uploader("quotes.csv", type=["csv"])
    f_p = st.file_uploader("proposals.csv", type=["csv"])

# Resolve Active DataFrame
scored_df = pd.DataFrame()
if all([f_acc, f_con, f_em, f_web, f_q, f_p]):
    try:
        scored_df = run_model_b_pipeline(
            pd.read_csv(f_acc), pd.read_csv(f_con), pd.read_csv(f_em), 
            pd.read_csv(f_web), pd.read_csv(f_q), pd.read_csv(f_p)
        )
        st.success("✅ Custom relational datasets parsed successfully.")
    except Exception as e:
        st.error(f"Error running custom pipeline: {e}. Falling back to default data.")
        scored_df = load_local_fallback_data()
else:
    scored_df = load_local_fallback_data()
    if not scored_df.empty:
        st.info("💡 Showing results computed from default datasets. Upload all files on the sidebar to update dynamically.")

if not scored_df.empty:
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Processed Accounts", f"{len(scored_df):,}")
    col2.metric("Average Score", f"{scored_df['composite_score'].mean():.1f}")
    col3.metric("Priority MQLs", f"{int((scored_df['tier']=='Priority MQL').sum()):,}")
    col4.metric("Total Premium Volume", f"${scored_df['est_premium'].sum() / 1e6:.2f}M")
    
    tab_diag, tab_tbl, tab_bot = st.tabs(["📈 Interactive Diagnostics", "📋 Scored Accounts List", "💬 Centralized Chatbot"])
    
    chart_specs = generate_chart_specs(scored_df)
    
    with tab_diag:
        st.subheader("Lead Quality Matrix & Breakdown")
        for spec in chart_specs:
            lc, rc = st.columns([2, 1])
            with lc:
                c_data = spec["data"]
                alt_chart = (
                    alt.Chart(c_data).mark_bar(color="#1f4e78", size=35)
                    .encode(
                        x=alt.X(f"{spec['x_label']}:N", sort=None),
                        y=alt.Y(f"{spec['y_label']}:Q"),
                        tooltip=[spec["x_label"], spec["y_label"]]
                    ).properties(height=260)
                )
                st.altair_chart(alt_chart, use_container_width=True)
            with rc:
                st.markdown(f"**{spec['title']} Insight**")
                st.info(spec["interpretation"])
                st.markdown("*AE Action Plan:*")
                st.caption(spec["recommendation"])
            st.divider()
            
    with tab_tbl:
        st.dataframe(scored_df[["account_id", "account_name", "account_type", "composite_score", "fit_score", "account_engagement", "tier", "archetype", "est_premium"]], use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export Scored Accounts Dataset", data=scored_df.to_csv(index=False).encode('utf-8'), file_name="scored_accounts.csv", mime="text/csv")
        
    with tab_bot:
        st.subheader("RevOps Grounded AI Assistant")
        st.write("Ask model-specific, factual questions regarding thresholds, tiers, and behavioral signals below.")
        
        # Initialize messages in session state if missing
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            
        # Persistent chatbot area container to prevent interface shifting
        chat_container = st.container(height=380, border=True)
        with chat_container:
            if not st.session_state.chat_history:
                st.caption("Enter a query below to initiate analysis...")
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
        if user_prompt := st.chat_input("Analyze Priority MQL conditions or dataset metrics..."):
            # Append user message
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
            
            # Generate response
            ans, refs = get_grounded_response(user_prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": ans})
            st.rerun()
else:
    st.warning("⚠️ Waiting for pipeline execution data. Please locate local dataset CSVs or prepare your files.")
