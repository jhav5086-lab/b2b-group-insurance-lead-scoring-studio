from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import analytics
import data_io
import pdf_generator
import scoring
import storyteller

try:
    from groq import Groq
except Exception:
    Groq = None

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(page_title="B2B Lead Scoring Studio", page_icon="🎯", layout="wide")

APP_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL = pd.Timestamp("2024-09-15")

# The LLM is optional: without a key the app shows the Python-written insights.
GROQ_API_KEY = ""
try:
    GROQ_API_KEY = str(st.secrets.get("GROQ_API_KEY", "")).strip()
    LLM_MODEL = str(st.secrets.get("GROQ_MODEL", "openai/gpt-oss-120b"))
except Exception:
    LLM_MODEL = "openai/gpt-oss-120b"
client = Groq(api_key=GROQ_API_KEY) if (GROQ_API_KEY and Groq is not None) else None


@st.cache_data(show_spinner=False)
def cached_defaults():
    return data_io.load_defaults(APP_DIR)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_polish(title, insight, action, model):
    """One LLM call per distinct text (cached), so moving a slider does not spend tokens again."""
    return storyteller.polish(client, model, title, insight, action)


def story_for(title, chart):
    if client is None:
        return {"insight": chart["insight"], "action": chart["action"], "status": "computed", "note": "no API key"}
    return cached_polish(title, chart["insight"], chart["action"], LLM_MODEL)


# =============================================================================
# SIDEBAR
# =============================================================================
st.title("🎯 B2B Lead Scoring Studio")
st.caption("Live scoring on the default dataset or your own CSVs, before/after email-decay comparison, and an executive PDF.")

default_tables, found_files, missing_files = cached_defaults()

with st.sidebar:
    st.header("⚙️ Data & Settings")
    st.caption("Core files (all three together):")
    up = {
        "accounts": st.file_uploader("Accounts CSV", type="csv"),
        "contacts": st.file_uploader("Contacts CSV", type="csv"),
        "emails": st.file_uploader("Email Events CSV", type="csv"),
    }
    st.caption("Optional files:")
    up["proposals"] = st.file_uploader("Proposals CSV", type="csv")
    up["quotes"] = st.file_uploader("Quotes CSV", type="csv")
    up["webinars"] = st.file_uploader("Webinar Events CSV", type="csv")
    st.info("🔒 Files are processed in memory and not stored. Don't upload confidential data to this public demo.")

    with st.expander("📋 Expected CSV Columns & Types"):
        st.markdown(
            "**Accounts:** `account_id` (req), `account_type` (req: Employer or Broker), `account_name`, `industry`, "
            "`industry_fit` (Target / Adjacent / Non-Target), `lives`, `est_premium`, `carrier`, `broker_id`, `renewal_date`\n\n"
            "**Contacts:** `contact_id` (req), `account_id` (req), `email_status` (Active / Unsubscribed / Bounced), `email`, "
            "`first_name`, `last_name`, `persona`, `domain_type`\n\n"
            "**Email Events:** `contact_id` (req), `event_date` (req), `link_category` (generic / content / product / high_intent), `event_id`\n\n"
            "**Proposals:** `account_id` (req), `proposal_id`, `proposal_date`, `proposed_premium`\n\n"
            "**Quotes:** `account_id` (req), `quote_id`, `product_code`, `quoted_premium`, `quote_date`, `status`\n\n"
            "**Webinar Events:** `contact_id` (req), `webinar_id`, `attended`, `minutes`, `polls`, `questions`, `downloaded`, `live`"
        )

    st.divider()
    eval_date = pd.Timestamp(st.date_input("Evaluation Date", DEFAULT_EVAL.date()))
    st.divider()
    apply_decay = st.checkbox("Apply Email Decay", value=True)
    decay_hl = st.slider("Decay Half-Life (Days, 0 = No Decay)", min_value=0, max_value=90, value=45)

# =============================================================================
# LOAD, VALIDATE, CHECK
# =============================================================================
if "accounts.csv" not in found_files and not any(up[k] is not None for k in data_io.CORE):
    st.error("No default dataset found. Add accounts.csv, contacts.csv and email_events.csv (and optionally proposals.csv, "
             "quotes.csv, webinar_events.csv) to the repo's `data/` folder, or upload your own files in the sidebar.")
    st.stop()

src = data_io.resolve_sources(up, default_tables)
for msg in src["errors"]:
    st.error(msg)
tables = data_io.normalize_tables(src["tables"])
checks = data_io.data_checks(tables)
dataset_label = src["label"]
st.sidebar.success(f"Active source: **{dataset_label}**")

with st.expander("🩺 Data health check", expanded=bool(checks["problems"] or src["warnings"])):
    st.write("**Rows loaded**")
    st.dataframe(pd.DataFrame([checks["rows"]]), hide_index=True)
    c1_, c2_, c3_ = st.columns(3)
    c1_.metric("Employer accounts", checks["employers"])
    c2_.metric("Contacts matched to accounts", f"{checks.get('contacts_matched_pct', 0):.0f}%")
    c3_.metric("Email events matched to contacts", f"{checks.get('events_matched_pct', 0):.0f}%")
    for w in src["warnings"] + checks["notes"]:
        st.warning(w)
    for p in checks["problems"]:
        st.error(p)

if checks["employers"] == 0 or checks["problems"]:
    st.error("The data has problems that make the results unreliable (see the health check above), so scoring is paused.")
    st.stop()

# =============================================================================
# SCORING (all math in pandas)
# =============================================================================
effective_decay = bool(apply_decay and decay_hl > 0)
args = dict(proposals=tables["proposals"], quotes=tables["quotes"], webinars=tables["webinars"])

with st.spinner("Calculating live scores..."):
    before_decay_df = scoring.run_pipeline(tables["accounts"], tables["contacts"], tables["emails"], eval_date,
                                           apply_decay=False, decay_halflife=0, **args)
    scored_df = scoring.run_pipeline(tables["accounts"], tables["contacts"], tables["emails"], eval_date,
                                     apply_decay=effective_decay, decay_halflife=decay_hl, **args)

with st.expander("🔍 View Raw Data & Scored Accounts"):
    st.write("### Scored Accounts (Top 50)")
    show_cols = [c for c in ["account_id", "account_type", "tier", "archetype", "fit_score", "account_engagement", "total_score"] if c in scored_df.columns]
    st.dataframe(scored_df[show_cols].head(50))
    for key, title in (("emails", "Raw Email Events (Top 20)"), ("contacts", "Raw Contacts (Top 20)"), ("accounts", "Raw Accounts (Top 20)")):
        st.write(f"### {title}")
        st.dataframe(tables[key].head(20))
    st.download_button("⬇️ Download all scored accounts (CSV)", scored_df.to_csv(index=False).encode("utf-8"),
                       file_name="scored_accounts.csv", mime="text/csv")

st.divider()

report = analytics.build_report(scored_df, before_decay_df, eval_date, effective_decay, decay_hl)
emp_view = scored_df[scored_df["account_type"] == "Employer"]

col1, col2 = st.columns(2)
with col1:
    st.subheader("Tier Distribution")
    tier_counts = emp_view["tier"].value_counts().reset_index()
    tier_counts.columns = ["Tier", "Count"]
    st.altair_chart(
        alt.Chart(tier_counts).mark_bar().encode(
            x=alt.X("Tier:N", sort="-y", title=""), y=alt.Y("Count:Q", title="Number of Accounts"),
            color=alt.Color("Tier:N", scale=alt.Scale(scheme="blues"), legend=None), tooltip=["Tier", "Count"]).properties(height=300),
        use_container_width=True)
    polished_t = story_for("Tier Distribution", report["charts"][0])
    st.info(f"**Insight:** {polished_t['insight']}\n\n**Action:** {polished_t['action']}")

with col2:
    st.subheader("Archetype Distribution")
    arch_counts = emp_view["archetype"].value_counts().reset_index()
    arch_counts.columns = ["Archetype", "Count"]
    st.altair_chart(
        alt.Chart(arch_counts).mark_arc(innerRadius=50).encode(
            theta=alt.Theta("Count:Q"), color=alt.Color("Archetype:N", legend=alt.Legend(title="Archetype")),
            tooltip=["Archetype", "Count"]).properties(height=300),
        use_container_width=True)
    polished_a = story_for("Archetype Distribution", report["charts"][1])
    st.info(f"**Insight:** {polished_a['insight']}\n\n**Action:** {polished_a['action']}")

st.divider()
st.subheader("⚖️ Before vs. After Decay Comparison")
st.caption(report.get("decay_summary", ""))
st.dataframe(report["decay_comparison"], use_container_width=True)
st.markdown("### 💡 Difference-Based Recommendations")
for rec in report.get("decay_recommendations", []):
    st.write(f"- {rec}")

# =============================================================================
# PDF
# =============================================================================
st.divider()
st.subheader("📄 Export Executive Report")
if "pdf_bytes" not in st.session_state:
    st.session_state["pdf_bytes"] = None

if st.button("Generate PDF Report"):
    with st.spinner("Generating multi-page PDF..."):
        try:
            meta = {"label": dataset_label, "eval_date": eval_date, "half_life": decay_hl, "decay": effective_decay}
            ai_texts = {"tiers": polished_t, "archetypes": polished_a}  # the decay page keeps its Python-computed text
            st.session_state["pdf_bytes"] = pdf_generator.build_pdf(report, meta, ai_texts)
            st.success("PDF generated successfully! Click the download button below.")
        except Exception as e:
            st.session_state["pdf_bytes"] = None
            st.error(f"Failed to generate PDF: {e}")

if st.session_state["pdf_bytes"]:
    st.download_button("⬇️ Download PDF", data=st.session_state["pdf_bytes"],
                       file_name="B2B_Lead_Scoring_Report.pdf", mime="application/pdf")
