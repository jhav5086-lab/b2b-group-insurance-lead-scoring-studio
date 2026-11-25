import os
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer

# =============================================================================
# 1. SETUP & SYNTHETIC DATA GENERATION
# =============================================================================
np.random.seed(42)
EVAL = pd.Timestamp("2024-09-15")
DATA_DIR = "/content/data"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("./embeddings", exist_ok=True)

print("Generating 2,500 synthetic employer & broker relational accounts...")
N_EMP, N_BRK = 2375, 125

inds = np.random.choice(
    ["Healthcare","Technology","Finance","Manufacturing","Retail","Transportation","Professional Services"],
    N_EMP, p=[0.20,0.15,0.15,0.15,0.10,0.10,0.15]
)
ind_fit = np.where(np.isin(inds, ["Healthcare","Technology","Finance","Manufacturing","Professional Services"]),
                   "Target", np.where(np.isin(inds, ["Retail","Transportation"]), "Adjacent", "Non-Target"))

lives = np.random.choice(
    [25,40,60,80,120,180,250,400,650,850,1200,1800,2500,4200,5000], N_EMP,
    p=[0.08,0.07,0.07,0.07,0.08,0.08,0.08,0.08,0.07,0.07,0.07,0.06,0.05,0.04,0.03]
)
premium = np.round(lives * np.random.uniform(800,1500,N_EMP), -3)
carriers = np.random.choice(	["Aetna","UnitedHealth","Cigna","Prudential","Unknown"], N_EMP,
    p=[0.30,0.25,0.20,0.15,0.10])
brk_ids = [f"BRK-{i:04d}" for i in range(1, N_BRK+1)]
assigned_brk = np.where(np.random.random(N_EMP) < 0.85, np.random.choice(brk_ids, N_EMP), "")
renewal_dates = [EVAL + pd.Timedelta(days=int(d)) for d in np.random.randint(0,365,N_EMP)]

emp = pd.DataFrame({
    "account_id": [f"ACC-{i:05d}" for i in range(1, N_EMP+1)],
    "account_name": [f"Company_{i:05d}" for i in range(1, N_EMP+1)],
    "account_type": "Employer", "industry": inds, "industry_fit": ind_fit,
    "lives": lives, "est_premium": premium, "carrier": carriers,
    "broker_id": assigned_brk, "renewal_date": renewal_dates
})

brk = pd.DataFrame({
    "account_id": brk_ids, "account_name": [f"Broker_{i:04d}" for i in range(1, N_BRK+1)],
    "account_type": "Broker", "industry": "Brokerage", "industry_fit": "",
    "lives": [0]*N_BRK, "est_premium": [0]*N_BRK, "carrier": [""]*N_BRK,
    "broker_id": [""]*N_BRK, "renewal_date": [pd.NaT]*N_BRK
})

accounts = pd.concat([emp, brk], ignore_index=True)
accounts.to_csv(f"{DATA_DIR}/accounts.csv", index=False)

n_c_emp = np.random.choice([2,3,4], N_EMP, p=[0.5,0.35,0.15])
n_c_brk = np.random.choice([1,2], N_BRK, p=[0.7,0.3])
acc_ids_rep = np.concatenate([np.repeat(emp["account_id"].values, n_c_emp), np.repeat(brk["account_id"].values, n_c_brk)])
personas = np.random.choice(["HR","Finance","Executive"], len(acc_ids_rep), p=[0.55,0.30,0.15])
is_broker_contact = np.isin(acc_ids_rep, brk["account_id"].values)
personas[is_broker_contact] = "Broker"
email_status = np.random.choice(["Active","Unsubscribed","Bounced"], len(acc_ids_rep), p=[0.95,0.03,0.02])
domain_type = np.random.choice(["Corporate","Competitor"], len(acc_ids_rep), p=[0.98,0.02])

contacts = pd.DataFrame({
    "contact_id": [f"CON-{i:06d}" for i in range(1, len(acc_ids_rep)+1)],
    "account_id": acc_ids_rep,
    "first_name": np.random.choice(["Julia","Mike","Sarah","David","Maria","Raj","Linda","Tom","Elena","Chris"], len(acc_ids_rep)),
    "last_name": np.random.choice(["Rivera","Chen","Patel","Okoro","Diaz","Kumar","Nguyen","Lee","Singh","Ross"], len(acc_ids_rep)),
    "persona": personas, "email_status": email_status, "domain_type": domain_type
})
contacts["email"] = (contacts["first_name"].str.lower() + "_" + contacts["last_name"].str.lower() + "_" + contacts["contact_id"].str[-4:] + "@corp.com")
contacts = contacts[["contact_id","account_id","email","first_name","last_name","persona","email_status","domain_type"]]
contacts.to_csv(f"{DATA_DIR}/contacts.csv", index=False)

n_ev = np.random.choice([0,1,2,3,5,8,12,18,25], len(contacts), p=[0.10,0.10,0.12,0.12,0.13,0.13,0.13,0.10,0.07])
n_ev[contacts["email_status"].values != "Active"] = 0
total_events = int(n_ev.sum())
eve_contact = np.repeat(contacts["contact_id"].values, n_ev)
eve_cat = np.random.choice(["generic","content","product","high_intent"], total_events, p=[0.20,0.40,0.25,0.15])
eve_days = np.random.randint(0, 180, total_events)
email_events = pd.DataFrame({
    "event_id": [f"E-{i:07d}" for i in range(1, total_events+1)],
    "contact_id": eve_contact, "link_category": eve_cat,
    "event_date": EVAL - pd.to_timedelta(eve_days, unit="D")
})
email_events.to_csv(f"{DATA_DIR}/email_events.csv", index=False)

n_w = np.random.choice([0,1,2,3], len(contacts), p=[0.70,0.15,0.10,0.05])
n_w[contacts["email_status"].values != "Active"] = 0
total_w = int(n_w.sum())
w_contact = np.repeat(contacts["contact_id"].values, n_w)
webinar_events = pd.DataFrame({
    "webinar_id": np.random.choice(["WEB-1","WEB-2","WEB-3","WEB-4"], total_w),
    "contact_id": w_contact,
    "attended": (np.random.random(total_w) < 0.60).astype(int),
    "minutes": np.random.choice([0,10,20,30,40,50,60], total_w),
    "polls": np.random.choice([0,1,2,3], total_w),
    "questions": np.random.choice([0,0,0,1,1,2,3], total_w),
    "downloaded": (np.random.random(total_w) < 0.40).astype(int),
    "live": (np.random.random(total_w) < 0.50).astype(int)
})
webinar_events.loc[webinar_events["attended"] == 0, ["minutes","polls","questions","downloaded","live"]] = 0
webinar_events.to_csv(f"{DATA_DIR}/webinar_events.csv", index=False)

contact_ev = contacts[["contact_id","account_id"]].merge(email_events.groupby("contact_id").size().rename("n_ev").reset_index(), on="contact_id", how="left").fillna({"n_ev":0})
first_ev = email_events.groupby("contact_id")["event_date"].min().reset_index()
contact_ev = contact_ev.merge(first_ev, on="contact_id", how="left")
acc_ev = contact_ev.groupby("account_id").agg(proxy_ev=("n_ev","sum"), first_ev=("event_date","min")).reset_index()
emp = emp.merge(acc_ev, on="account_id", how="left").fillna({"proxy_ev":0})

emp["quote_prob"] = (0.06 + 0.22 * (emp["est_premium"] >= 1000000) + 0.20 * (emp["proxy_ev"].fillna(0) >= 5) + 0.27 * (emp["proxy_ev"].fillna(0) >= 15)).clip(0, 0.80)
quote_mask = np.random.random(len(emp)) < emp["quote_prob"].values
q_emp = emp[quote_mask].copy()
q_emp["quote_date"] = (q_emp["first_ev"] + pd.to_timedelta(np.random.randint(10,60,len(q_emp)), unit="D")).fillna(EVAL - pd.Timedelta(days=30)).clip(upper=EVAL)
q_emp["quoted_premium"] = (q_emp["est_premium"] * np.random.uniform(0.85, 1.05, len(q_emp))).round(-3)
q_emp["status"] = np.random.choice(["Issued","Requested"], len(q_emp), p=[0.88, 0.12])

quotes = pd.DataFrame({
    "quote_id": [f"Q-{i:05d}" for i in range(1, len(q_emp)+1)],
    "account_id": q_emp["account_id"].values, "product_code": "FW-Wellness",
    "quoted_premium": q_emp["quoted_premium"].values, "quote_date": q_emp["quote_date"].values, "status": q_emp["status"].values
})
quotes.to_csv(f"{DATA_DIR}/quotes.csv", index=False)

issued = quotes[quotes["status"] == "Issued"].copy()
issued["prop_prob"] = (0.12 + 0.30 * (issued["quoted_premium"] >= 1000000) + 0.30 * (issued["quoted_premium"] >= 2000000)).clip(0, 0.85)
prop_mask = np.random.random(len(issued)) < issued["prop_prob"].values
p_emp = issued[prop_mask].copy()
p_emp["proposal_date"] = (p_emp["quote_date"] + pd.to_timedelta(np.random.randint(10,45,len(p_emp)), unit="D")).clip(upper=EVAL)
p_emp["proposed_premium"] = (p_emp["quoted_premium"] * np.random.uniform(0.85, 1.0, len(p_emp))).round(-3)

proposals = pd.DataFrame({
    "proposal_id": [f"P-{i:05d}" for i in range(1, len(p_emp)+1)],
    "account_id": p_emp["account_id"].values, "proposal_date": p_emp["proposal_date"].values, "proposed_premium": p_emp["proposed_premium"].values
})
proposals.to_csv(f"{DATA_DIR}/proposals.csv", index=False)

# =============================================================================
# 2. RUN MODEL B PIPELINE SCORING
# =============================================================================
print("Processing lead scoring rules...")
TIER_POINTS = {"generic": 2, "content": 5, "product": 8, "high_intent": 20}
TIER_HL = {"generic": 60, "content": 60, "product": 45, "high_intent": 30}

email_events["points"] = email_events["link_category"].map(TIER_POINTS).fillna(0)
email_events["hl"] = email_events["link_category"].map(TIER_HL).fillna(60)
email_events["days_since"] = (EVAL - email_events["event_date"]).dt.days.clip(lower=0)
email_events["decayed"] = email_events["points"] * (0.5 ** (email_events["days_since"] / email_events["hl"]))
email_score_agg = email_events.groupby("contact_id")["decayed"].sum().clip(upper=50).rename("email_score")

webinar_events["wpts"] = (
    webinar_events["attended"] * 5 +
    (webinar_events["minutes"] // 10).clip(upper=5) +
    webinar_events["polls"] * 2 +
    webinar_events["questions"] * 5 +
    webinar_events["downloaded"] * 5 +
    webinar_events["live"] * 3
)
web_score_agg = webinar_events.groupby("contact_id")["wpts"].sum().clip(upper=30).rename("web_score")

contacts = contacts.merge(email_score_agg, on="contact_id", how="left").fillna({"email_score": 0})
contacts = contacts.merge(web_score_agg, on="contact_id", how="left").fillna({"web_score": 0})
contacts["engagement"] = contacts["email_score"] + contacts["web_score"]
contacts.loc[contacts["email_status"] != "Active", "engagement"] = 0

eng_rows = []
for acc_id, g in contacts.groupby("account_id"):
    s = sorted(g["engagement"].tolist(), reverse=True)
    best = s[0] if s else 0
    others = sum(s[1:]) if len(s) > 1 else 0
    distinct = int((g["engagement"] > 0).sum())
    ms = 10 if distinct >= 2 else 0
    ae = round(best + 0.3 * others + ms, 2) if best > 0 else 0
    eng_rows.append((acc_id, best, round(others, 2), distinct, ms, ae))

acc_eng = pd.DataFrame(eng_rows, columns=["account_id","best_contact","other_contacts","distinct_engaged","ms_bonus","account_engagement"])
broker_eng_map = dict(zip(acc_eng["account_id"], acc_eng["account_engagement"]))
accounts = accounts.merge(acc_eng, on="account_id", how="left")
for col in ["best_contact","other_contacts","distinct_engaged","ms_bonus","account_engagement"]: 
    accounts[col] = accounts[col].fillna(0)

accounts["broker_both"] = accounts.apply(lambda r: r["account_type"] == "Employer" and broker_eng_map.get(r["broker_id"], 0) >= 15 and r["account_engagement"] >= 15, axis=1)
accounts["broker_bonus"] = np.where(accounts["broker_both"], 15, 0)

def calculate_fit_score(r):
    if r["account_type"] == "Broker": return 40
    d = (pd.to_datetime(r["renewal_date"]) - EVAL).days if pd.notna(r["renewal_date"]) else None
    if d is not None and d < 0: d += 365
    ren_pts = 0 if d is None else (25 if d <= 30 else 20 if d <= 60 else 15 if d <= 90 else 5 if d <= 180 else 0)
    prem = r["est_premium"]
    size_pts = 25 if prem >= 2000000 else 20 if prem >= 1000000 else 15 if prem >= 500000 else 10 if prem >= 100000 else 0
    ind_pts = {"Target": 15, "Adjacent": 8, "Non-Target": 0}.get(r["industry_fit"], 0)
    car_pts = 10 if r["carrier"] in ("Aetna", "UnitedHealth", "Cigna") else (5 if r["carrier"] else 0)
    brk_pts = 15 if r["broker_id"] else 0
    return ren_pts + size_pts + ind_pts + car_pts + brk_pts

accounts["fit_score"] = accounts.apply(calculate_fit_score, axis=1)
accounts["days_to_renewal"] = accounts["renewal_date"].apply(lambda d: (pd.to_datetime(d) - EVAL).days if pd.notna(d) else np.nan)

competitor_accounts = set(contacts.loc[contacts["domain_type"] == "Competitor", "account_id"])
accounts["disqualified"] = (accounts["account_type"] == "Employer") & ((accounts["lives"] < 50) | (accounts["account_id"].isin(competitor_accounts)))
accounts["composite_score"] = (0.4 * accounts["fit_score"] + 0.6 * accounts["account_engagement"] + 0.8 * accounts["broker_bonus"]).clip(1, 100)

hi_ids = set(email_events.loc[email_events["link_category"] == "high_intent", "contact_id"])
hi_accs = set(contacts.loc[contacts["contact_id"].isin(hi_ids), "account_id"])
accounts["has_high_intent"] = accounts["account_id"].isin(hi_accs)

def calculate_tier_row(r):
    if r["disqualified"]: return "Disqualified"
    if r["account_type"] == "Broker": return "Priority Broker" if r["composite_score"] >= 60 else ("Broker MQL" if r["composite_score"] >= 40 else "Responder")
    urgent = pd.notna(r["days_to_renewal"]) and r["days_to_renewal"] <= 90
    signal = urgent or r["distinct_engaged"] >= 2 or r["broker_both"]
    if r["composite_score"] >= 70 and r["est_premium"] >= 1000000 and r["account_engagement"] >= 15 and signal: return "Priority MQL"
    if r["composite_score"] >= 50: return "MQL"
    if r["account_engagement"] > 0: return "Responder"
    return "Disqualified"

accounts["tier"] = accounts.apply(calculate_tier_row, axis=1)

def calculate_archetype_row(r):
    if r["account_type"] == "Broker": return "Broker"
    if r["disqualified"]: return "Detractor"
    fh = r["fit_score"] >= 60
    e = r["account_engagement"]
    multi = r["distinct_engaged"] >= 2
    if fh and e >= 40 and multi: return "Champion"
    if fh and e >= 40 and not multi: return "Achiever"
    if fh and 15 <= e < 40 and not r["has_high_intent"]: return "Watcher"
    if fh and e < 15: return "Dormant Giant"
    if 5 <= e < 15: return "Fader"
    return "Lurker"

accounts["archetype"] = accounts.apply(calculate_archetype_row, axis=1)

# =============================================================================
# 3. CREATE RAG CHUNKS, EMBED AND EXPORT EMBEDDINGS
# =============================================================================
print("Generating RAG chunks & computing SentenceTransformer embeddings...")
chunks = [
    {"type": "model_summary", "title": "Lead Scoring Model Overview", "text": "The B2B Lead Scoring Model evaluates 2,375 employers and 125 brokers. It uses a composite score (1-100) calculated as 0.4 * fit_score + 0.6 * account_engagement + 0.8 * broker_bonus. The model outputs a Tier (Priority MQL, MQL, Responder, Disqualified) and an Archetype (Champion, Achiever, Watcher, Dormant Giant, Fader, Lurker, Detractor)."},
    {"type": "methodology", "title": "Fit Score Calculation Rules", "text": "Fit score is based on: Renewal proximity (25 pts if <=30 days, 20 if <=60, 15 if <=90, 5 if <=180), Premium size (25 pts if >=$2M, 20 if >=$1M, 15 if >=$500K, 10 if >=$100K), Industry fit (15 pts Target, 8 Adjacent, 0 Non-Target), Carrier presence (10 pts for Aetna/UnitedHealth/Cigna), and Broker presence (15 pts)."},
    {"type": "methodology", "title": "Engagement Score Rules", "text": "Engagement is calculated from email and webinar activities. Email points range from 2 (generic) to 20 (high_intent) with a half-life decay. Webinar points include attendance (5), minutes (up to 5), polls (2), questions (5), and downloads (5). Account engagement = Best contact score + 0.3 * (other contacts) + 10 pt multi-stakeholder bonus if >=2 contacts are engaged."},
    {"type": "recommendations", "title": "Tiering Definitions", "text": "Priority MQL: Total score >=70, premium >=$1M, engagement >=15, and active signal (urgent renewal, multi-threaded, or broker engaged). MQL: Total score >=50. Responder: Score <50 but has engagement. Disqualified: Lives <50 or competitor domain."}
]

with open("./embeddings/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
texts = [c["text"] for c in chunks]
embeddings = embedder.encode(texts, convert_to_numpy=True)
np.save("./embeddings/embeddings.npy", embeddings)

with zipfile.ZipFile("/content/embeddings.zip", "w") as zf:
    zf.write("./embeddings/chunks.json", "embeddings/chunks.json")
    zf.write("./embeddings/embeddings.npy", "embeddings/embeddings.npy")

# =============================================================================
# 4. EXPORT UNIFIED APP TO APP.PY (CLEAN STREAMLINED APP)
# =============================================================================
print("Writing consolidated streamlit dashboard app to /content/app.py...")

with open("/content/app.py", "w", encoding="utf-8") as app_file:
    app_file.write("""
import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from sentence_transformers import SentenceTransformer
from groq import Groq

# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="B2B Insurance Scoring & RevOps Studio",
    page_icon="┐",
    layout="wide"
)

# Safely resolve secrets with fallbacks
GROQ_API_KEY = ""
try:
    if "GROQ_API_KEY" in st.secrets:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not GROQ_API_KEY:
    try:
        from google.colab import userdata
        GROQ_API_KEY = userdata.get("GROQ_API_KEY")
    except Exception:
        try:
            GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
        except Exception:
            pass

if not GROQ_API_KEY:
    st.warning("☑☑ **Missing API Key:** Set GROQ_API_KEY to enable Chatbot capabilities.")

CHUNKS_PATH = "./embeddings/chunks.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

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
    if os.path.exists("/content/data/accounts.csv"):
        accs = pd.read_csv("/content/data/accounts.csv")
        cont = pd.read_csv("/content/data/contacts.csv")
        em = pd.read_csv("/content/data/email_events.csv")
        web = pd.read_csv("/content/data/webinar_events.csv")
        quo = pd.read_csv("/content/data/quotes.csv")
        prop = pd.read_csv("/content/data/proposals.csv")
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
    if EMBEDDINGS_NORM is None:
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
st.title("┐ RevOps Unified Lead Scoring Studio")
st.caption("Process complete relational pipeline datasets, track dynamic behaviors, and interact with the grounding bot.")

scored_df = load_local_fallback_data()
if not scored_df.empty:
    st.success("☑ Successfully loaded and processed dynamic RevOps pipeline!")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Accounts", f"{len(scored_df):,}")
    c2.metric("Avg Composite Score", f"{scored_df[‘composite_score’].mean():.1f}")
    c3.metric("Priority MQLs", f"{int((scored_df[‘tier’] == ‘Priority MQL’).sum()):,}")
    c4.metric("Total Annual Premium", f"${scored_df[‘annual_premium_usd’].sum() / 1e6:.2f}M")

    tab_table, tab_charts, tab_chat = st.tabs(["┒ Scored Accounts Table", "│ Interactive Insights", "┤ RAG Assistant"])

    with tab_table:
        st.dataframe(scored_df.head(200), use_container_width=True)
        st.download_button(
            label="↓│ Download Scored Accounts CSV",
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
                retrieved_context = "\\n\\n".join([f"[{r[‘title’]}]: {r[‘text’]}" for r in retrieved])
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
    st.warning("☑ Waiting for pipeline execution data.")
""")

print("‰ Consolidated pipeline completed! Model outputs, embeddings, and app.py are fully assembled.")
