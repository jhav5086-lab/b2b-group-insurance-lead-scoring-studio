import os
import streamlit as st
import pandas as pd
from groq import Groq
import altair as alt

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
LLM_MODEL = st.secrets.get("GROQ_MODEL", "openai/gpt-oss-120b")

REQUIRED_SCHEMAS = {
    "Accounts": ["account_id", "account_type"],
    "Contacts": ["contact_id", "account_id"],
    "Email Events": ["contact_id", "event_date"],
    "Proposals": ["account_id"],
    "Quotes": ["account_id"],
    "Webinar Events": ["contact_id"],
}


@st.cache_data
def load_default_dataset():
    """Load the provided default CSV dataset from disk if present."""
    eval_date = pd.Timestamp("2024-09-15")
    files = {
        "accounts": "accounts.csv",
        "contacts": "contacts.csv",
        "emails": "email_events.csv",
        "proposals": "proposals.csv",
        "quotes": "quotes.csv",
        "webinars": "webinar_events.csv",
    }
    loaded = {}
    for key, fname in files.items():
        if os.path.exists(fname):
            loaded[key] = pd.read_csv(fname)
        else:
            loaded[key] = pd.DataFrame()
    return (
        loaded["accounts"],
        loaded["contacts"],
        loaded["emails"],
        loaded["proposals"],
        loaded["quotes"],
        loaded["webinars"],
        eval_date,
    )


def validate_table(name: str, df: pd.DataFrame, req_cols: list[str]) -> list[str]:
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        return [f"**{name}** missing required columns: `{', '.join(missing)}`"]
    return []


def normalize_tables(acc_df, con_df, em_df, prop_df, quo_df, web_df):
    acc = acc_df.copy()
    con = con_df.copy()
    em = em_df.copy()
    prop = prop_df.copy() if prop_df is not None and not prop_df.empty else pd.DataFrame()
    quo = quo_df.copy() if quo_df is not None and not quo_df.empty else pd.DataFrame()
    web = web_df.copy() if web_df is not None and not web_df.empty else pd.DataFrame()

    for col in ["account_id", "account_name", "account_type", "industry", "industry_fit", "carrier", "broker_id"]:
        if col in acc.columns:
            acc[col] = acc[col].fillna("").astype(str).str.strip()

    for col in ["contact_id", "account_id", "email", "first_name", "last_name", "persona", "email_status", "domain_type"]:
        if col in con.columns:
            con[col] = con[col].fillna("").astype(str).str.strip()

    for col in ["event_id", "contact_id", "link_category"]:
        if col in em.columns:
            em[col] = em[col].fillna("").astype(str).str.strip()

    for col in ["lives", "est_premium"]:
        if col in acc.columns:
            acc[col] = pd.to_numeric(acc[col], errors="coerce").fillna(0.0)

    if "renewal_date" in acc.columns:
        acc["renewal_date"] = pd.to_datetime(acc["renewal_date"], errors="coerce")

    if "event_date" in em.columns:
        em["event_date"] = pd.to_datetime(em["event_date"], errors="coerce")
        em = em.dropna(subset=["event_date"])

    if not prop.empty:
        if "account_id" in prop.columns:
            prop["account_id"] = prop["account_id"].fillna("").astype(str).str.strip()
        if "proposal_date" in prop.columns:
            prop["proposal_date"] = pd.to_datetime(prop["proposal_date"], errors="coerce")
        if "proposed_premium" in prop.columns:
            prop["proposed_premium"] = pd.to_numeric(prop["proposed_premium"], errors="coerce").fillna(0.0)

    if not quo.empty:
        if "account_id" in quo.columns:
            quo["account_id"] = quo["account_id"].fillna("").astype(str).str.strip()
        if "quote_date" in quo.columns:
            quo["quote_date"] = pd.to_datetime(quo["quote_date"], errors="coerce")
        if "quoted_premium" in quo.columns:
            quo["quoted_premium"] = pd.to_numeric(quo["quoted_premium"], errors="coerce").fillna(0.0)

    if not web.empty:
        if "contact_id" in web.columns:
            web["contact_id"] = web["contact_id"].fillna("").astype(str).str.strip()
        for wcol in ["attended", "minutes", "polls", "questions", "downloaded", "live"]:
            if wcol in web.columns:
                web[wcol] = pd.to_numeric(web[wcol], errors="coerce").fillna(0)

    return acc, con, em, prop, quo, web


# =============================================================================
# 1. UI SETUP
# =============================================================================
st.title("🎯 B2B Lead Scoring Studio")
st.caption("RevOps lead scoring with default 6-CSV dataset or custom uploads, decay = 0 support, before/after decay comparison, and PDF export.")

(
    def_acc,
    def_con,
    def_em,
    def_prop,
    def_quo,
    def_web,
    DEFAULT_EVAL,
) = load_default_dataset()

with st.sidebar:
    st.header("⚙️ Data & Settings")
    upload_acc = st.file_uploader("Upload Accounts CSV", type="csv")
    upload_con = st.file_uploader("Upload Contacts CSV", type="csv")
    upload_em = st.file_uploader("Upload Email Events CSV", type="csv")
    upload_prop = st.file_uploader("Upload Proposals CSV", type="csv")
    upload_quo = st.file_uploader("Upload Quotes CSV", type="csv")
    upload_web = st.file_uploader("Upload Webinar Events CSV", type="csv")

    with st.expander("📋 Expected CSV Columns & Types"):
        st.markdown(
            "**Accounts:** `account_id` (str, req), `account_type` (str, req), `account_name` (str), `industry` (str), "
            "`industry_fit` (str), `lives` (int), `est_premium` (float), `carrier` (str), `broker_id` (str), `renewal_date` (date)\n\n"
            "**Contacts:** `contact_id` (str, req), `account_id` (str, req), `email` (str), `first_name` (str), `last_name` (str), "
            "`persona` (str), `email_status` (str), `domain_type` (str)\n\n"
            "**Email Events:** `event_id` (str), `contact_id` (str, req), `link_category` (str), `event_date` (date, req)\n\n"
            "**Proposals:** `proposal_id` (str), `account_id` (str, req), `proposal_date` (date), `proposed_premium` (float)\n\n"
            "**Quotes:** `quote_id` (str), `account_id` (str, req), `product_code` (str), `quoted_premium` (float), "
            "`quote_date` (date), `status` (str)\n\n"
            "**Webinar Events:** `webinar_id` (str), `contact_id` (str, req), `attended` (int), `minutes` (int), "
            "`polls` (int), `questions` (int), `downloaded` (int), `live` (int)"
        )

    st.divider()
    eval_date = pd.Timestamp(st.date_input("Evaluation Date", DEFAULT_EVAL.date()))

    st.divider()
    apply_decay = st.checkbox("Apply Email Decay", value=True)
    decay_hl = st.slider("Decay Half-Life (Days, 0 = No Decay)", min_value=0, max_value=90, value=45)

# =============================================================================
# 2. LOAD DATA & RUN PIPELINE
# =============================================================================
raw_acc = pd.read_csv(upload_acc) if upload_acc else def_acc
raw_con = pd.read_csv(upload_con) if upload_con else def_con
raw_em = pd.read_csv(upload_em) if upload_em else def_em
raw_prop = pd.read_csv(upload_prop) if upload_prop else def_prop
raw_quo = pd.read_csv(upload_quo) if upload_quo else def_quo
raw_web = pd.read_csv(upload_web) if upload_web else def_web

any_upload = any([upload_acc, upload_con, upload_em, upload_prop, upload_quo, upload_web])

schema_errors = []
for label, df, req in [
    ("Accounts", raw_acc, REQUIRED_SCHEMAS["Accounts"]),
    ("Contacts", raw_con, REQUIRED_SCHEMAS["Contacts"]),
    ("Email Events", raw_em, REQUIRED_SCHEMAS["Email Events"]),
]:
    if not df.empty:
        schema_errors.extend(validate_table(label, df, req))

if schema_errors:
    for err in schema_errors:
        st.error(err)
    st.warning("Reverting to default dataset files due to missing required columns.")
    raw_acc, raw_con, raw_em, raw_prop, raw_quo, raw_web = def_acc, def_con, def_em, def_prop, def_quo, def_web
    dataset_label = "Default Dataset (Fallback)"
else:
    dataset_label = "Uploaded CSV Dataset" if any_upload else "Default Provided Dataset"

st.sidebar.info(f"ℹ️ Active source: **{dataset_label}**")

df_acc, df_con, df_em, df_prop, df_quo, df_web = normalize_tables(
    raw_acc, raw_con, raw_em, raw_prop, raw_quo, raw_web
)

effective_decay = bool(apply_decay and decay_hl > 0)

with st.spinner("Calculating live scores..."):
    before_decay_df = scoring.run_pipeline(
        df_acc, df_con, df_em, eval_date,
        apply_decay=False, decay_halflife=0,
        proposals=df_prop, quotes=df_quo, webinars=df_web,
    )
    scored_df = scoring.run_pipeline(
        df_acc, df_con, df_em, eval_date,
        apply_decay=effective_decay, decay_halflife=decay_hl,
        proposals=df_prop, quotes=df_quo, webinars=df_web,
    )

# =============================================================================
# 3. RAW DATA & CHARTS
# =============================================================================
with st.expander("🔍 View Raw Data & Scored Accounts"):
    st.write("### Scored Accounts (Top 50)")
    show_cols = [c for c in ["account_id", "tier", "archetype", "fit_score", "account_engagement", "total_score"] if c in scored_df.columns]
    st.dataframe(scored_df[show_cols].head(50))
    st.write("### Raw Email Events (Top 20)")
    st.dataframe(df_em.head(20))

st.divider()

report = analytics.build_report(scored_df, before_decay_df, eval_date, effective_decay, decay_hl)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Tier Distribution")
    emp_view = scored_df[scored_df["account_type"] == "Employer"] if "account_type" in scored_df.columns else scored_df
    tier_counts = emp_view["tier"].value_counts().reset_index() if len(emp_view) else pd.DataFrame(columns=["Tier", "Count"])
    if not tier_counts.empty:
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
    arch_counts = emp_view["archetype"].value_counts().reset_index() if len(emp_view) else pd.DataFrame(columns=["Archetype", "Count"])
    if not arch_counts.empty:
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
# 4. BEFORE VS AFTER DECAY COMPARISON & RECOMMENDATIONS
# =============================================================================
st.divider()
st.subheader("⚖️ Before vs. After Decay Comparison")
st.caption(report.get("decay_summary", ""))
st.dataframe(report["decay_comparison"], use_container_width=True)

st.markdown("### 💡 Difference-Based Recommendations")
for rec in report.get("decay_recommendations", []):
    st.write(f"- {rec}")

# =============================================================================
# 5. PDF EXPORT
# =============================================================================
st.divider()
st.subheader("📄 Export Executive Report")

if "pdf_bytes" not in st.session_state:
    st.session_state["pdf_bytes"] = None

if st.button("Generate PDF Report"):
    with st.spinner("Generating multi-page PDF..."):
        try:
            meta = {
                "label": dataset_label,
                "eval_date": eval_date,
                "half_life": decay_hl,
                "decay": effective_decay,
            }
            ai_texts = {
                "tiers": polished_t,
                "archetypes": polished_a,
                "decay": polished_a,
            }
            st.session_state["pdf_bytes"] = pdf_generator.build_pdf(report, meta, ai_texts)
            st.success("PDF generated successfully! Click the download button below.")
        except Exception as e:
            st.session_state["pdf_bytes"] = None
            st.error(f"Failed to generate PDF: {e}")

if st.session_state["pdf_bytes"]:
    st.download_button(
        label="⬇️ Download PDF",
        data=st.session_state["pdf_bytes"],
        file_name="B2B_Lead_Scoring_Report.pdf",
        mime="application/pdf",
    )
