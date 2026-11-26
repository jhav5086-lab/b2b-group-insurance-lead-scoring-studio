import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import json
import re
import os
import time
import io
from datetime import datetime
from groq import Groq
from fpdf import FPDF
import vl_convert as vlc

# =============================================================================
# CONFIG & API SETUP
# =============================================================================
st.set_page_config(page_title="B2B Lead Scoring Engine", page_icon="🎯", layout="wide")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 **Missing API Key:** add GROQ_API_KEY to Streamlit secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
# Using 8b for insights to avoid daily limits and run instantly
LLM_MODEL = "llama-3.1-8b-instant" 

# =============================================================================
# 1. DEFAULT DATA GENERATOR
# =============================================================================
@st.cache_data
def generate_default_data():
    np.random.seed(42)
    EVAL = pd.Timestamp("2024-09-15")
    N_EMP, N_BRK = 2375, 125
    
    inds = np.random.choice(["Healthcare","Technology","Finance","Manufacturing","Retail","Transportation","Professional Services"], N_EMP, p=[0.20,0.15,0.15,0.15,0.10,0.10,0.15])
    ind_fit = np.where(np.isin(inds, ["Healthcare","Technology","Finance","Manufacturing","Professional Services"]), "Target", np.where(np.isin(inds, ["Retail","Transportation"]), "Adjacent", "Non-Target"))
    lives = np.random.choice([25,40,60,80,120,180,250,400,650,850,1200,1800,2500,4200,5000], N_EMP, p=[0.08,0.07,0.07,0.07,0.08,0.08,0.08,0.08,0.07,0.07,0.07,0.06,0.05,0.04,0.03])
    premium = np.round(lives * np.random.uniform(800,1500,N_EMP), -3)
    carriers = np.random.choice(["Aetna","UnitedHealth","Cigna","Prudential","Unknown"], N_EMP, p=[0.30,0.25,0.20,0.15,0.10])
    brk_ids = [f"BRK-{i:04d}" for i in range(1, N_BRK+1)]
    assigned_brk = np.where(np.random.random(N_EMP) < 0.85, np.random.choice(brk_ids, N_EMP), "")
    renewal_dates = [EVAL + pd.Timedelta(days=int(d)) for d in np.random.randint(0,365,N_EMP)]

    emp = pd.DataFrame({
        "account_id":[f"ACC-{i:05d}" for i in range(1,N_EMP+1)], "account_type":"Employer",
        "industry_fit":ind_fit, "lives":lives, "est_premium":premium, "carrier":carriers,
        "broker_id":assigned_brk, "renewal_date":renewal_dates})

    brk = pd.DataFrame({
        "account_id":brk_ids, "account_type":"Broker", "industry_fit":"", "lives":0,
        "est_premium":0, "carrier":"", "broker_id":"", "renewal_date":pd.NaT})
    accounts = pd.concat([emp, brk], ignore_index=True)

    n_c_emp = np.random.choice([2,3,4], N_EMP, p=[0.5,0.35,0.15])
    acc_ids_rep = np.concatenate([np.repeat(emp["account_id"].values, n_c_emp), np.repeat(brk["account_id"].values, 1)])
    contacts = pd.DataFrame({
        "contact_id":[f"CON-{i:06d}" for i in range(1,len(acc_ids_rep)+1)],
        "account_id":acc_ids_rep,
        "email_status":np.random.choice(["Active","Unsubscribed","Bounced"], len(acc_ids_rep), p=[0.95,0.03,0.02])})

    n_ev = np.random.choice([0,1,2,3,5,8,12,18,25], len(contacts), p=[0.10,0.10,0.12,0.12,0.13,0.13,0.13,0.10,0.07])
    n_ev[contacts["email_status"].values != "Active"] = 0
    total_events = int(n_ev.sum())
    ev_cat = np.random.choice(["generic","content","product","high_intent"], total_events, p=[0.20,0.40,0.25,0.15])
    ev_days = np.random.randint(0, 180, total_events)
    emails = pd.DataFrame({"contact_id":np.repeat(contacts["contact_id"].values, n_ev), "link_category":ev_cat, "event_date":EVAL - pd.to_timedelta(ev_days, unit="D")})

    return accounts, contacts, emails, EVAL

# =============================================================================
# 2. CORE MATH & SCORING ENGINE
# =============================================================================
def run_pipeline(accounts, contacts, emails, eval_date, apply_decay, decay_halflife=45):
    acc = accounts.copy()
    con = contacts.copy()
    em = emails.copy()

    TIER_POINTS = {"generic":2,"content":5,"product":8,"high_intent":20}
    em["points"] = em["link_category"].map(TIER_POINTS).fillna(0)
    em["days_since"] = (eval_date - pd.to_datetime(em["event_date"])).dt.days.clip(lower=0)
    
    if apply_decay:
        em["decayed_points"] = em["points"] * (0.5 ** (em["days_since"] / decay_halflife))
    else:
        em["decayed_points"] = em["points"]

    email_s = em.groupby("contact_id")["decayed_points"].sum().clip(upper=50).rename("email_score")

    con = con.merge(email_s, on="contact_id", how="left").fillna({"email_score":0})
    con.loc[con["email_status"] != "Active", "email_score"] = 0
    con["engagement"] = con["email_score"]

    rows = []
    for acc_id, g in con.groupby("account_id"):
        s = sorted(g["engagement"].tolist(), reverse=True)
        best = s[0] if s else 0
        others = sum(s[1:]) if len(s)>1 else 0
        distinct = int((g["engagement"]>0).sum())
        ms = 10 if distinct>=2 else 0
        ae = round(best + 0.3*others + ms, 2) if best>0 else 0
        rows.append((acc_id, distinct, ae))
    
    acc_eng = pd.DataFrame(rows, columns=["account_id","distinct_engaged","account_engagement"])
    acc = acc.merge(acc_eng, on="account_id", how="left").fillna({"distinct_engaged":0, "account_engagement":0})

    def fit_score(r):
        if r["account_type"]=="Broker": return 40
        d = (pd.to_datetime(r["renewal_date"]) - eval_date).days if pd.notna(r["renewal_date"]) else 999
        if d < 0: d += 365
        ren = 25 if d<=30 else 20 if d<=60 else 15 if d<=90 else 5 if d<=180 else 0
        p = r["est_premium"]
        sz  = 25 if p>=2_000_000 else 20 if p>=1_000_000 else 15 if p>=500_000 else 10 if p>=100_000 else 0
        ind = {"Target":15,"Adjacent":8,"Non-Target":0}.get(r["industry_fit"],0)
        car = 10 if r["carrier"] in ("Aetna","UnitedHealth","Cigna") else (5 if r["carrier"] else 0)
        brk = 15 if r["broker_id"] else 0
        return ren+sz+ind+car+brk

    acc["fit_score"] = acc.apply(fit_score, axis=1)
    acc["total_score"] = (0.4*acc["fit_score"] + 0.6*acc["account_engagement"]).clip(1,100)

    def tier_row(r):
        if r["account_type"]=="Broker": return "Broker"
        if r["total_score"]>=70 and r["est_premium"]>=1_000_000 and r["account_engagement"]>=15: return "Priority MQL"
        if r["total_score"]>=50: return "MQL"
        if r["account_engagement"]>0: return "Responder"
        return "Disqualified"
    
    acc["tier"] = acc.apply(tier_row, axis=1)

    def arch_row(r):
        if r["account_type"]=="Broker": return "Broker"
        if r["tier"]=="Disqualified": return "Detractor"
        fh = r["fit_score"]>=60
        e  = r["account_engagement"]
        multi = r["distinct_engaged"]>=2
        if fh and e>=40 and multi: return "Champion"
        if fh and e>=40 and not multi: return "Achiever"
        if fh and 15<=e<40: return "Watcher"
        if fh and e<15: return "Dormant Giant"
        if 5<=e<15: return "Fader"
        return "Lurker"
    
    acc["archetype"] = acc.apply(arch_row, axis=1)
    return acc

# =============================================================================
# 3. INSIGHT GENERATION & PDF GENERATOR
# =============================================================================
def generate_insight(chart_title, data_summary):
    prompt = f"""
    You are a RevOps data analyst. Here is the data for the chart '{chart_title}':
    {data_summary}
    
    Write exactly 2 sentences:
    1. An insight explaining the biggest takeaway from this data (use exact numbers).
    2. A recommendation on what the Sales team should do based on this data.
    
    Do not write anything else. Do not say "Here is the insight".
    """
    try:
        res = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return "Could not generate insight due to rate limits."

def create_pdf(chart1, insight1, chart2, insight2, summary_text):
    """Generates a clean, formatted PDF from Altair charts and text."""
    pdf = FPDF()
    pdf.add_page()
    
    # 1. Title Page
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 20, "B2B Lead Scoring Executive Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.ln(10)
    
    # 2. Summary
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Executive Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 8, summary_text)
    pdf.ln(10)
    
    # 3. Charts & Insights
    # Convert Altair to PNG using vl-convert
    png1 = vlc.altair_to_png(chart1)
    png2 = vlc.altair_to_png(chart2)
    
    with open("temp_chart1.png", "wb") as f: f.write(png1)
    with open("temp_chart2.png", "wb") as f: f.write(png2)
    
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Tier Distribution", ln=True)
    pdf.image("temp_chart1.png", w=170)
    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 11)
    pdf.multi_cell(0, 8, f"Insight: {insight1}")
    
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Archetype Distribution", ln=True)
    pdf.image("temp_chart2.png", w=170)
    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 11)
    pdf.multi_cell(0, 8, f"Insight: {insight2}")
    
    # Cleanup temp files
    os.remove("temp_chart1.png")
    os.remove("temp_chart2.png")
    
    return bytes(pdf.output())

# =============================================================================
# 4. UI SETUP & DATA UPLOAD
# =============================================================================
st.title("🎯 B2B Lead Scoring Analytics Engine")
st.caption("Upload your data to run live analytics, or use our default synthetic dataset.")

with st.sidebar:
    st.header("⚙️ Data & Settings")
    upload_acc = st.file_uploader("Upload Accounts CSV", type="csv")
    upload_con = st.file_uploader("Upload Contacts CSV", type="csv")
    upload_em = st.file_uploader("Upload Email Events CSV", type="csv")
    
    st.divider()
    default_acc, default_con, default_em, DEFAULT_EVAL = generate_default_data()
    
    eval_date = st.date_input("Evaluation Date", DEFAULT_EVAL.date())
    eval_date = pd.Timestamp(eval_date)
    
    st.divider()
    apply_decay = st.checkbox("Apply Email Decay", value=True)
    decay_hl = 45
    if apply_decay:
        decay_hl = st.slider("Decay Half-Life (Days)", 10, 90, 45)

# Load Data
if upload_acc and upload_con and upload_em:
    df_acc = pd.read_csv(upload_acc)
    df_con = pd.read_csv(upload_con)
    df_em = pd.read_csv(upload_em)
    st.sidebar.success("✅ Using uploaded data")
else:
    df_acc, df_con, df_em = default_acc, default_con, default_em
    st.sidebar.info("ℹ️ Using default synthetic dataset")

# Run Math Pipeline
with st.spinner("Calculating scores..."):
    scored_df = run_pipeline(df_acc, df_con, df_em, eval_date, apply_decay, decay_hl)

# =============================================================================
# 5. MAIN DASHBOARD (Raw Data & Insights)
# =============================================================================
with st.expander("🔍 View Raw Data & Scored Accounts"):
    st.write("### Scored Accounts")
    st.dataframe(scored_df[["account_id", "tier", "archetype", "fit_score", "account_engagement", "total_score"]].head(50))
    st.write("### Raw Email Events")
    st.dataframe(df_em.head(20))

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Tier Distribution")
    tier_counts = scored_df[scored_df["account_type"]=="Employer"]["tier"].value_counts().reset_index()
    tier_counts.columns = ["Tier", "Count"]
    
    chart = alt.Chart(tier_counts).mark_bar().encode(
        x=alt.X("Tier:N", sort="-y", title=""),
        y=alt.Y("Count:Q", title="Number of Accounts"),
        color=alt.Color("Tier:N", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=["Tier", "Count"]
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)
    
    data_str = ", ".join([f"{row['Tier']}: {row['Count']}" for _, row in tier_counts.iterrows()])
    insight1 = generate_insight("Tier Distribution", data_str)
    st.info(insight1)

with col2:
    st.subheader("Archetype Distribution")
    arch_counts = scored_df[scored_df["account_type"]=="Employer"]["archetype"].value_counts().reset_index()
    arch_counts.columns = ["Archetype", "Count"]
    
    chart2 = alt.Chart(arch_counts).mark_arc(innerRadius=50).encode(
        theta=alt.Theta("Count:Q"),
        color=alt.Color("Archetype:N", legend=alt.Legend(title="Archetype")),
        tooltip=["Archetype", "Count"]
    ).properties(height=300)
    st.altair_chart(chart2, use_container_width=True)
    
    data_str2 = ", ".join([f"{row['Archetype']}: {row['Count']}" for _, row in arch_counts.iterrows()])
    insight2 = generate_insight("Archetype Distribution", data_str2)
    st.info(insight2)

st.subheader("Impact of Email Decay")
no_decay_df = run_pipeline(df_acc, df_con, df_em, eval_date, False)
decay_comparison = pd.DataFrame({
    "Metric": ["Avg Engagement", "MQL Count", "Priority MQL Count"],
    "Without Decay": [no_decay_df["account_engagement"].mean(), len(no_decay_df[no_decay_df["tier"]=="MQL"]), len(no_decay_df[no_decay_df["tier"]=="Priority MQL"])],
    "With Decay": [scored_df["account_engagement"].mean(), len(scored_df[scored_df["tier"]=="MQL"]), len(scored_df[scored_df["tier"]=="Priority MQL"])]
})
st.dataframe(decay_comparison, hide_index=True)

# =============================================================================
# 6. PDF EXPORT BUTTON
# =============================================================================
st.divider()
st.subheader("📄 Export Report")
if st.button("Generate Executive PDF"):
    with st.spinner("Generating PDF..."):
        try:
            # Create a short summary for the PDF
            summary_prompt = f"Write a 3-sentence executive summary of this data: {tier_counts.to_dict('records')} and {arch_counts.to_dict('records')}. The evaluation date is {eval_date.date()} and decay is {'on' if apply_decay else 'off'}."
            res = client.chat.completions.create(model=LLM_MODEL, messages=[{"role": "user", "content": summary_prompt}], temperature=0.2)
            summary = res.choices[0].message.content.strip()
            
            pdf_bytes = create_pdf(chart, insight1, chart2, insight2, summary)
            
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_bytes,
                file_name="B2B_Lead_Scoring_Report.pdf",
                mime="application/pdf"
            )
            st.success("PDF generated successfully! Click the download button above.")
        except Exception as e:
            st.error(f"Failed to generate PDF: {e}")
