import pandas as pd
import numpy as np

# Required by pdf_generator.py
RULES_TABLE = [
    ("Renewal Proximity", "25 pts <=30 days, 20 pts <=60, 15 pts <=90, 5 pts <=180", "Assumed"),
    ("Premium Size", "25 pts >=$2M, 20 pts >=$1M, 15 pts >=$500K, 10 pts >=$100K", "Assumed"),
    ("Industry Fit", "15 pts Target, 8 pts Adjacent, 0 pts Non-Target", "Assumed"),
    ("Carrier", "10 pts for Aetna/UnitedHealth/Cigna, 5 pts other known carrier, 0 if unknown", "Assumed"),
    ("Broker", "15 pts if broker is assigned", "Assumed"),
    ("Engagement Signals", "Email decay + optional webinar, quote, and proposal signals", "Assumed"),
    ("Total Score", "0.4 * Fit Score + 0.6 * Account Engagement (clipped 1..100)", "Assumed"),
    ("Priority MQL", "Total >=70, Premium >=$1M, Engagement >=15", "Assumed"),
    ("MQL", "Total >=50", "Assumed"),
]

_NO_CARRIER = {"", "unknown", "nan", "none"}


def _empty_scores(key_col, value_col):
    """An empty score table that still has the key column, so merges never fail on missing data."""
    return pd.DataFrame({key_col: pd.Series(dtype="object"), value_col: pd.Series(dtype="float64")})


def run_pipeline(
    accounts,
    contacts,
    emails,
    eval_date,
    apply_decay,
    decay_halflife=45,
    proposals=None,
    quotes=None,
    webinars=None,
):
    acc = accounts.copy() if accounts is not None else pd.DataFrame()
    con = contacts.copy() if contacts is not None else pd.DataFrame()
    em = emails.copy() if emails is not None else pd.DataFrame()
    prop = proposals.copy() if proposals is not None else pd.DataFrame()
    quo = quotes.copy() if quotes is not None else pd.DataFrame()
    web = webinars.copy() if webinars is not None else pd.DataFrame()

    eval_ts = pd.Timestamp(eval_date)

    # ---- email engagement per contact (always a table with a contact_id column) ----
    if not em.empty and "contact_id" in em.columns:
        if "link_category" not in em.columns:
            em["link_category"] = ""
        if "event_date" not in em.columns:
            em["event_date"] = eval_ts
        TIER_POINTS = {"generic": 2, "content": 5, "product": 8, "high_intent": 20}
        em["points"] = em["link_category"].fillna("").astype(str).str.strip().map(TIER_POINTS).fillna(0)
        em["days_since"] = (eval_ts - pd.to_datetime(em["event_date"], errors="coerce")).dt.days.clip(lower=0).fillna(0)

        # Safe handling when decay is disabled or half-life is 0
        if apply_decay and decay_halflife and decay_halflife > 0:
            em["decayed_points"] = em["points"] * (0.5 ** (em["days_since"] / float(decay_halflife)))
        else:
            em["decayed_points"] = em["points"]

        email_s = em.groupby("contact_id")["decayed_points"].sum().clip(upper=50).rename("email_score").reset_index()
    else:
        email_s = _empty_scores("contact_id", "email_score")

    if not con.empty:
        if "email_status" not in con.columns:
            con["email_status"] = "Active"
        con = con.merge(email_s, on="contact_id", how="left").fillna({"email_score": 0})
        con.loc[con["email_status"] != "Active", "email_score"] = 0

        # Optional webinar engagement per contact
        if not web.empty and "contact_id" in web.columns:
            for wcol in ["attended", "minutes", "polls", "questions", "downloaded", "live"]:
                if wcol not in web.columns:
                    web[wcol] = 0
            web_pts = (
                web["attended"].fillna(0).clip(lower=0) * 5
                + (web["minutes"].fillna(0).clip(lower=0) / 15.0).clip(upper=4) * 2
                + web["polls"].fillna(0).clip(lower=0) * 1.5
                + web["questions"].fillna(0).clip(lower=0) * 2.5
                + web["downloaded"].fillna(0).clip(lower=0) * 3
                + web["live"].fillna(0).clip(lower=0) * 2
            )
            web_s = web.assign(_wpts=web_pts).groupby("contact_id")["_wpts"].sum().clip(upper=20).rename("webinar_score").reset_index()
            con = con.merge(web_s, on="contact_id", how="left").fillna({"webinar_score": 0})
        else:
            con["webinar_score"] = 0.0

        con["engagement"] = con["email_score"] + con["webinar_score"]

        rows = []
        for acc_id, g in con.groupby("account_id"):
            s = sorted(g["engagement"].tolist(), reverse=True)
            best = s[0] if s else 0.0
            others = sum(s[1:]) if len(s) > 1 else 0.0
            distinct = int((g["engagement"] > 0).sum())
            ms = 10.0 if distinct >= 2 else 0.0
            ae = round(best + 0.3 * others + ms, 2) if best > 0 else 0.0
            rows.append((acc_id, distinct, ae))

        acc_eng = pd.DataFrame(rows, columns=["account_id", "distinct_engaged", "account_engagement"])
    else:
        acc_eng = pd.DataFrame(columns=["account_id", "distinct_engaged", "account_engagement"])

    if acc.empty:
        return acc

    if "account_type" not in acc.columns:
        acc["account_type"] = "Employer"
    for col, default_val in [
        ("renewal_date", pd.NaT),
        ("est_premium", 0.0),
        ("industry_fit", ""),
        ("carrier", ""),
        ("broker_id", ""),
    ]:
        if col not in acc.columns:
            acc[col] = default_val
    for col in ("industry_fit", "carrier", "broker_id"):
        acc[col] = acc[col].fillna("").astype(str).str.strip()

    acc["est_premium"] = pd.to_numeric(acc["est_premium"], errors="coerce").fillna(0.0)
    acc["renewal_date"] = pd.to_datetime(acc["renewal_date"], errors="coerce")
    acc = acc.merge(acc_eng, on="account_id", how="left").fillna({"distinct_engaged": 0, "account_engagement": 0.0})

    # Optional account-level signals from quotes and proposals
    quote_accounts = set()
    if not quo.empty and "account_id" in quo.columns:
        quote_accounts = set(quo["account_id"].dropna().astype(str))
    proposal_accounts = set()
    if not prop.empty and "account_id" in prop.columns:
        proposal_accounts = set(prop["account_id"].dropna().astype(str))

    acc["has_quote"] = acc["account_id"].astype(str).isin(quote_accounts)
    acc["has_proposal"] = acc["account_id"].astype(str).isin(proposal_accounts)
    acc["account_engagement"] = (
        acc["account_engagement"]
        + np.where(acc["has_quote"], 5.0, 0.0)
        + np.where(acc["has_proposal"], 8.0, 0.0)
    ).round(2)

    def fit_score(r):
        if r["account_type"] == "Broker":
            return 40
        d = (pd.to_datetime(r["renewal_date"]) - eval_ts).days if pd.notna(r["renewal_date"]) else 999
        if d < 0:
            d += 365
        ren = 25 if d <= 30 else 20 if d <= 60 else 15 if d <= 90 else 5 if d <= 180 else 0
        p = r["est_premium"]
        sz = 25 if p >= 2_000_000 else 20 if p >= 1_000_000 else 15 if p >= 500_000 else 10 if p >= 100_000 else 0
        ind = {"Target": 15, "Adjacent": 8, "Non-Target": 0}.get(str(r["industry_fit"]), 0)
        carrier = str(r["carrier"]).strip()
        car = 10 if carrier in ("Aetna", "UnitedHealth", "Cigna") else (0 if carrier.lower() in _NO_CARRIER else 5)
        brk = 15 if str(r["broker_id"]).strip() else 0
        return ren + sz + ind + car + brk

    acc["fit_score"] = acc.apply(fit_score, axis=1)
    acc["total_score"] = (0.4 * acc["fit_score"] + 0.6 * acc["account_engagement"]).clip(1, 100)
    acc["days_to_renewal"] = acc["renewal_date"].apply(
        lambda d: (pd.to_datetime(d) - eval_ts).days if pd.notna(d) else 999
    )

    def tier_row(r):
        if r["account_type"] == "Broker":
            return "Broker"
        if r["total_score"] >= 70 and r["est_premium"] >= 1_000_000 and r["account_engagement"] >= 15:
            return "Priority MQL"
        if r["total_score"] >= 50:
            return "MQL"
        if r["account_engagement"] > 0:
            return "Responder"
        return "Disqualified"

    acc["tier"] = acc.apply(tier_row, axis=1)

    def arch_row(r):
        if r["account_type"] == "Broker":
            return "Broker"
        if r["tier"] == "Disqualified":
            return "Detractor"
        fh = r["fit_score"] >= 60
        e = r["account_engagement"]
        multi = r["distinct_engaged"] >= 2
        if fh and e >= 40 and multi:
            return "Champion"
        if fh and e >= 40 and not multi:
            return "Achiever"
        if fh and 15 <= e < 40:
            return "Watcher"
        if fh and e < 15:
            return "Dormant Giant"
        if 5 <= e < 15:
            return "Fader"
        return "Lurker"

    acc["archetype"] = acc.apply(arch_row, axis=1)
    return acc
