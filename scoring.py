import pandas as pd
import numpy as np

# Required by pdf_generator.py
RULES_TABLE = [
    ("Renewal Proximity", "25 pts <=30 days, 20 pts <=60, 15 pts <=90, 5 pts <=180", "Assumed"),
    ("Premium Size", "25 pts >=$2M, 20 pts >=$1M, 15 pts >=$500K, 10 pts >=$100K", "Assumed"),
    ("Industry Fit", "15 pts Target, 8 pts Adjacent, 0 pts Non-Target", "Assumed"),
    ("Carrier", "10 pts for Aetna/UnitedHealth/Cigna, 5 pts otherwise", "Assumed"),
    ("Broker", "15 pts if broker is assigned", "Assumed"),
    ("Total Score", "0.4 * Fit Score + 0.6 * Account Engagement", "Assumed"),
    ("Priority MQL", "Total >=70, Premium >=$1M, Engagement >=15", "Assumed"),
    ("MQL", "Total >=50", "Assumed"),
]

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
    acc["days_to_renewal"] = acc["renewal_date"].apply(lambda d: (pd.to_datetime(d)-eval_date).days if pd.notna(d) else 999)

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
