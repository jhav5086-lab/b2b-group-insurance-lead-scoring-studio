import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from groq import Groq

# Import our custom modular files
import scoring
import analytics
import storyteller
import pdf_generator

# =============================================================================
# CONFIG & API SETUP
# =============================================================================
st.set_page_config(page_title="B2B Lead Scoring Studio", page_icon="🎯", layout="wide")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 **Missing API Key:** add GROQ_API_KEY to Streamlit secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
LLM_MODEL = "llama-3.1-8b-instant"

REQUIRED_SCHEMAS = {
    "Accounts": ["account_id", "account_type"],
    "Contacts": ["contact_id", "account_id"],
    "Email Events": ["contact_id", "event_date"],
}


def validate_and_normalize_uploads(acc_df: pd.DataFrame, con_df: pd.DataFrame, em_df: pd.DataFrame):
    """Validate mandatory columns and normalize expected column data types."""
    errors = []
    for name, df, req_cols in [
        ("Accounts", acc_df, REQUIRED_SCHEMAS["Accounts"]),
        ("Contacts", con_df, REQUIRED_SCHEMAS["Contacts"]),
        ("Email Events", em_df, REQUIRED_SCHEMAS["Email Events"]),
    ]:
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            errors.append(f"**{name}** missing required columns: `{', '.join(missing)}`")

    if errors:
        return None, None, None, errors

    acc = acc_df.copy()
    con = con_df.copy()
    em = em_df.copy()

    # String IDs and categorical text columns
    for col in ["account_id", "account_type", "industry_fit", "carrier", "broker_id"]:
        if col in acc.columns:
            acc[col] = acc[col].fillna("").astype(str).str.strip()

    for col in ["contact_id", "account_id", "email_status"]:
        if col in con.columns:
            con[col] = con[col].fillna("").astype(str).str.strip()

    for col in ["contact_id", "link_category"]:
        if col in em.columns:
            em[col] = em[col].fillna("").astype(str).str.strip()

    # Numeric columns
    for col in ["lives", "est_premium"]:
        if col in acc.columns:
            acc[col] = pd.to_numeric(acc[col], errors="coerce").fillna(0)

    # Date columns
    if "renewal_date" in acc.columns:
        acc["renewal_date"] = pd.to_datetime(acc["renewal_date"], errors="coerce")

    if "event_date" in em.columns:
        em["event_date"] = pd.to_datetime(em["event_date"], errors="coerce")
        em = em.dropna(subset=["event_date"])

    return acc, con, em, []


# =============================================================================
# 1. DATA GENERATOR (Fallback Default Data)
# =============================================================================
@st.cache_data
def generate_default_data():
    np.random.seed(42)
    EVAL = pd.Timestamp("2024-09-15")
    N_EMP, N_BRK = 2375, 125

    inds = np.random.choice(
        ["Healthcare", "Technology", "Finance", "Manufacturing", "Retail", "Transportation", "Professional Services"],
        N_EMP,
        p=[0.20, 0.15, 0.15, 0.15, 0.10, 0.10, 0.15],
    )
    ind_fit = np.where(
        np.isin(inds, ["Healthcare", "Technology", "Finance", "Manufacturing", "Professional Services"]),
        "Target",
        np.where(np.isin(inds, ["Retail", "Transportation"]), "Adjacent", "Non-Target"),
    )
    lives = np.random.choice(
        [25, 40, 60, 80, 120, 180, 250, 400, 650, 850, 1200, 1800, 2500, 4200, 5000],
        N_EMP,
        p=[0.08, 0.07, 0.07, 0.07, 0.08, 0.08, 0.08, 0.08, 0.07, 0.07, 0.07, 0.06, 0.05, 0.04, 0.03],
    )
    premium = np.round(lives * np.random.uniform(800, 1500, N_EMP), -3)
    carriers = np.random.choice(["Aetna", "UnitedHealth", "Cigna", "Prudential", "Unknown"], N_EMP, p=[0.30, 0.25, 0.20, 0.15, 0.10])
    brk_ids = [f"BRK-{i:04d}" for i in range(1, N_BRK + 1)]
    assigned_brk = np.where(np.random.random(N_EMP) < 0.85, np.random.choice(brk_ids, N_EMP), "")
    renewal_dates = [EVAL + pd.Timedelta(days=int(d)) for d in np.random.randint(0, 365, N_EMP)]

    emp = pd.DataFrame({
        "account_id": [f"ACC-{i:05d}" for i in range(1, N_EMP + 1)],
        "account_type": "Employer",
        "industry_fit": ind_fit,
        "lives": lives,
        "est_premium": premium,
        "carrier": carriers,
        "broker_id": assigned_brk,
        "renewal_date": renewal_dates,
    })

    brk = pd.DataFrame({
        "account_id": brk_ids,
        "account_type": "Broker",
        "industry_fit": "",
        "lives": 0,
        "est_premium": 0,
        "carrier": "",
        "broker_id": "",
        "renewal_date": pd.NaT,
    })
    accounts = pd.concat([emp, brk], ignore_index=True)

    n_c_emp = np.random.choice([2, 3, 4], N_EMP, p=[0.5, 0.35, 0.15])
    acc_ids_rep = np.concatenate([np.repeat(emp["account_id"].values, n_c_emp), np.repeat(brk["account_id"].values, 1)])
    contacts = pd.DataFrame({
        "contact_id": [f"CON-{i:06d}" for i in range(1, len(acc_ids_rep) + 1)],
        "account_id": acc_ids_rep,
        "email_status": np.random.choice(["Active", "Unsubscribed", "Bounced"], len(acc_ids_rep), p=[0.95, 0.03, 0.02]),
    })

    n_ev = np.random.choice([0, 1, 2, 3, 5, 8, 12, 18, 25], len(contacts), p=[0.10, 0.10, 0.12, 0.12, 0.13, 0.13, 0.13, 0.10, 0.07])
    n_ev[contacts["email_status"].values != "Active"] = 0
    total_events = int(n_ev.sum())
    ev_cat = np.random.choice(["generic", "content", "product", "high_intent"], total_events, p=[0.20, 0.40, 0.25, 0.15])
    ev_days = np.random.randint(0, 180, total_events)
    emails = pd.DataFrame({
        "contact_id": np.repeat(contacts["contact_id"].values, n_ev),
        "link_category": ev_cat,
        "event_date": EVAL - pd.to_timedelta(ev_days, unit="D"),
    })

    return accounts, contacts, emails, EVAL


# =============================================================================
# 2. UI SETUP
# =============================================================================
st.title("🎯 B2B Lead Scoring Studio")
st.caption("Live RevOps simulation with email decay, dynamic charts, and LLM-generated PDF reports.")

default_acc, default_con, default_em, DEFAULT_EVAL = generate_default_data()

with st.sidebar:
    st.header("⚙️ Data & Settings")
    upload_acc = st.file_uploader("Upload Accounts CSV", type="csv")
    upload_con = st.file_uploader("Upload Contacts CSV", type="csv")
    upload_em = st.file_uploader("Upload Email Events CSV", type="csv")

    with st.expander("📋 Expected CSV Columns & Types"):
        st.markdown(
            "**Accounts:** `account_id` (str, req), `account_type` (str, req), `industry_fit` (str), "
            "`lives` (int), `est_premium` (float), `carrier` (str), `broker_id` (str), `renewal_date` (date)\n\n"
            "**Contacts:** `contact_id` (str, req), `account_id` (str, req), `email_status` (str)\n\n"
            "**Email Events:** `contact_id` (str, req), `event_date` (date, req), `link_category` (str)"
        )

    st.divider()
    eval_date = pd.Timestamp(st.date_input("Evaluation Date", DEFAULT_EVAL.date()))

    st.divider()
    apply_decay = st.checkbox("Apply Email Decay", value=True)
    decay_hl = st.slider("Decay Half-Life (Days)", 10, 90, 45) if apply_decay else 45

# =============================================================================
# 3. LOAD DATA & RUN PIPELINE
# =============================================================================
if upload_acc and upload_con and upload_em:
    raw_acc = pd.read_csv(upload_acc)
    raw_con = pd.read_csv(upload_con)
    raw_em = pd.read_csv(upload_em)
    df_acc, df_con, df_em, schema_errors = validate_and_normalize_uploads(raw_acc, raw_con, raw_em)
    if schema_errors:
        for err in schema_errors:
            st.error(err)
        st.info("Falling back to default synthetic dataset until all required columns are present.")
        df_acc, df_con, df_em = default_acc, default_con, default_em
    else:
        st.sidebar.success("✅ Using validated uploaded data")
else:
    if any([upload_acc, upload_con, upload_em]):
        st.sidebar.warning("⚠️ Upload all three CSVs (Accounts, Contacts, Email Events) to override defaults.")
    df_acc, df_con, df_em = default_acc, default_con, default_em
    st.sidebar.info("ℹ️ Using default synthetic dataset")

with st.spinner("Calculating live scores..."):
    scored_df = scoring.run_pipeline(df_acc, df_con, df_em, eval_date, apply_decay, decay_hl)
    no_decay_df = scoring.run_pipeline(df_acc, df_con, df_em, eval_date, False) if apply_decay else scored_df

# =============================================================================
# 4. RAW DATA & CHARTS
# =============================================================================
with st.expander("🔍 View Raw Data & Scored Accounts"):
    st.write("### Scored Accounts (Top 50)")
    st.dataframe(scored_df[["account_id", "tier", "archetype", "fit_score", "account_engagement", "total_score"]].head(50))
    st.write("### Raw Email Events (Top 20)")
    st.dataframe(df_em.head(20))

st.divider()

report = analytics.build_report(scored_df, no_decay_df, eval_date, apply_decay, decay_hl)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Tier Distribution")
    tier_counts = scored_df[scored_df["account_type"] == "Employer"]["tier"].value_counts().reset_index()
    tier_counts.columns = ["Tier", "Count"]

    c1 = (
        alt.Chart(tier_counts)
        .mark_bar()
        .encode(
            x=alt.X("Tier:N", sort="-y", title=""),
            y=alt.Y("Count:Q", title="Number of Accounts"),
            color=alt.Color("Tier:N", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["Tier", "Count"],
        )
        .properties(height=300)
    )
    st.altair_chart(c1, use_container_width=True)

    tier_data = report["charts"][0]
    polished_t = storyteller.polish(client, LLM_MODEL, "Tier Distribution", tier_data["insight"], tier_data["action"])
    st.info(f"**Insight:** {polished_t['insight']}\n\n**Action:** {polished_t['action']}")

with col2:
    st.subheader("Archetype Distribution")
    arch_counts = scored_df[scored_df["account_type"] == "Employer"]["archetype"].value_counts().reset_index()
    arch_counts.columns = ["Archetype", "Count"]

    c2 = (
        alt.Chart(arch_counts)
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color("Archetype:N", legend=alt.Legend(title="Archetype")),
            tooltip=["Archetype", "Count"],
        )
        .properties(height=300)
    )
    st.altair_chart(c2, use_container_width=True)

    arch_data = report["charts"][1]
    polished_a = storyteller.polish(client, LLM_MODEL, "Archetype Distribution", arch_data["insight"], arch_data["action"])
    st.info(f"**Insight:** {polished_a['insight']}\n\n**Action:** {polished_a['action']}")

# =============================================================================
# 5. PDF EXPORT
# =============================================================================
st.divider()
st.subheader("📄 Export Executive Report")

if st.button("Generate PDF Report"):
    with st.spinner("Generating multi-page PDF..."):
        try:
            meta = {
                "label": "Uploaded Data" if (upload_acc and upload_con and upload_em and not schema_errors) else "Synthetic Data",
                "eval_date": eval_date,
                "half_life": decay_hl,
                "decay": apply_decay,
            }
            ai_texts = {
                "tiers": polished_t,
                "archetypes": polished_a,
                "decay": polished_a,
            }
            pdf_bytes = pdf_generator.build_pdf(report, meta, ai_texts)
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_bytes,
                file_name="B2B_Lead_Scoring_Report.pdf",
                mime="application/pdf",
            )
            st.success("PDF generated successfully! Click the download button above.")
        except Exception as e:
            st.error(f"Failed to generate PDF: {e}")
