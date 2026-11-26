"""Safe data loading for the Streamlit app: default files, uploads, validation and health checks.

Rules that make the app trustworthy on any dataset:
  * Default data comes from the repo's data/ folder; if it is missing the app says so instead of showing zeros.
  * The three core files (accounts, contacts, email events) must be uploaded TOGETHER. A partial upload never
    gets mixed with default tables.
  * Optional tables (proposals, quotes, webinars) are taken from uploads only when you upload your own core data,
    so default rows can never leak into someone else's accounts.
  * data_checks() reports row counts and how many IDs match, so a wrong file is visible immediately.
"""
from pathlib import Path

import pandas as pd

FILES = {"accounts": "accounts.csv", "contacts": "contacts.csv", "emails": "email_events.csv",
         "proposals": "proposals.csv", "quotes": "quotes.csv", "webinars": "webinar_events.csv"}
CORE = ["accounts", "contacts", "emails"]
OPTIONAL = ["proposals", "quotes", "webinars"]
LABELS = {"accounts": "Accounts", "contacts": "Contacts", "emails": "Email Events",
          "proposals": "Proposals", "quotes": "Quotes", "webinars": "Webinar Events"}
REQUIRED = {"accounts": ["account_id", "account_type"], "contacts": ["contact_id", "account_id"],
            "emails": ["contact_id", "event_date"], "proposals": ["account_id"], "quotes": ["account_id"], "webinars": ["contact_id"]}
RECOMMENDED = {"accounts": ["est_premium", "renewal_date", "industry_fit", "carrier", "broker_id"],
               "contacts": ["email_status"], "emails": ["link_category"]}
ID_COLS = ["account_id", "contact_id", "event_id", "webinar_id", "quote_id", "proposal_id", "broker_id"]


def read_csv_safe(source):
    """Reads a path or an uploaded file. IDs stay text so '00123' or 'ACC-1' never turn into numbers."""
    return pd.read_csv(source, dtype={c: str for c in ID_COLS})


def default_dir(app_dir):
    for d in (Path(app_dir) / "data", Path(app_dir)):
        if (d / FILES["accounts"]).exists():
            return d
    return None


def load_defaults(app_dir):
    """Returns (tables, found_files, missing_files). tables always has all six keys."""
    d = default_dir(app_dir)
    tables, found, missing = {}, [], []
    for key, fname in FILES.items():
        path = d / fname if d else None
        if path is not None and path.exists():
            tables[key] = read_csv_safe(path)
            found.append(fname)
        else:
            tables[key] = pd.DataFrame()
            missing.append(fname)
    return tables, found, missing


def _missing_cols(name, df):
    return [c for c in REQUIRED[name] if c not in df.columns]


def resolve_sources(uploads, defaults):
    """uploads: {key: file-like or None}. Returns dict(tables, label, is_upload, errors, warnings)."""
    errors, warnings = [], []
    uploaded_core = [k for k in CORE if uploads.get(k) is not None]
    use_default = True
    tables = None

    if uploaded_core:
        if len(uploaded_core) < len(CORE):
            miss = [LABELS[k] for k in CORE if k not in uploaded_core]
            errors.append(f"Upload incomplete: please also upload {', '.join(miss)} (all three core files are needed together). "
                          "Showing the default data instead.")
        else:
            try:
                tables = {k: read_csv_safe(uploads[k]) for k in CORE}
                for k in OPTIONAL:
                    tables[k] = read_csv_safe(uploads[k]) if uploads.get(k) is not None else pd.DataFrame()
                bad = [f"{LABELS[k]} is missing required column(s): {', '.join(_missing_cols(k, tables[k]))}"
                       for k in CORE if _missing_cols(k, tables[k])]
                bad += [f"{LABELS[k]} is missing required column(s): {', '.join(_missing_cols(k, tables[k]))}"
                        for k in OPTIONAL if not tables[k].empty and _missing_cols(k, tables[k])]
                if bad:
                    errors += bad + ["Showing the default data instead."]
                    tables = None
                elif any(tables[k].empty for k in CORE):
                    errors.append("One of the core files has no rows. Showing the default data instead.")
                    tables = None
                else:
                    use_default = False
                    for k in OPTIONAL:
                        if uploads.get(k) is None:
                            warnings.append(f"No {LABELS[k]} file uploaded: scoring runs without those signals.")
                    for k in CORE:
                        for c in RECOMMENDED.get(k, []):
                            if c not in tables[k].columns:
                                warnings.append(f"{LABELS[k]}: column '{c}' is missing, so it is treated as empty.")
            except Exception as e:  # unreadable or malformed CSV
                errors.append(f"Could not read an uploaded file: {e}. Showing the default data instead.")
                tables = None

    if use_default:
        tables = {k: v.copy() for k, v in defaults.items()}
        label = "Default dataset" if not errors else "Default dataset (upload not used)"
    else:
        label = "Uploaded dataset"
    return {"tables": tables, "label": label, "is_upload": not use_default, "errors": errors, "warnings": warnings}


def normalize_tables(tables):
    """Cleans types and labels. Works on any combination of tables (missing ones become empty frames)."""
    out = {k: (tables.get(k) if tables.get(k) is not None else pd.DataFrame()).copy() for k in FILES}
    acc, con, em, prop, quo, web = (out[k] for k in ("accounts", "contacts", "emails", "proposals", "quotes", "webinars"))

    def txt(df, cols):
        for c in cols:
            if c in df.columns:
                df[c] = df[c].fillna("").astype(str).str.strip().replace({"nan": "", "NaN": "", "None": ""})

    txt(acc, ["account_id", "account_name", "account_type", "industry", "industry_fit", "carrier", "broker_id"])
    txt(con, ["contact_id", "account_id", "email", "first_name", "last_name", "persona", "email_status", "domain_type"])
    txt(em, ["event_id", "contact_id", "link_category"])
    for df, cols in ((prop, ["account_id"]), (quo, ["account_id"]), (web, ["contact_id"])):
        txt(df, cols)

    if "account_type" in acc.columns:
        acc["account_type"] = acc["account_type"].str.title()
    if "industry_fit" in acc.columns:
        acc["industry_fit"] = acc["industry_fit"].str.title().replace({"Non-target": "Non-Target", "Nontarget": "Non-Target"})
    if "email_status" in con.columns:
        con["email_status"] = con["email_status"].str.title()
    if "link_category" in em.columns:
        em["link_category"] = em["link_category"].str.lower()
    for col in ("lives", "est_premium"):
        if col in acc.columns:
            acc[col] = pd.to_numeric(acc[col], errors="coerce").fillna(0.0)
    if "renewal_date" in acc.columns:
        acc["renewal_date"] = pd.to_datetime(acc["renewal_date"], errors="coerce")
    if "event_date" in em.columns:
        em["event_date"] = pd.to_datetime(em["event_date"], errors="coerce")
        em = em.dropna(subset=["event_date"])
    for df, col, kind in ((prop, "proposal_date", "date"), (prop, "proposed_premium", "num"),
                          (quo, "quote_date", "date"), (quo, "quoted_premium", "num")):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce") if kind == "date" else pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in ("attended", "minutes", "polls", "questions", "downloaded", "live"):
        if col in web.columns:
            web[col] = pd.to_numeric(web[col], errors="coerce").fillna(0)
    return {"accounts": acc, "contacts": con, "emails": em, "proposals": prop, "quotes": quo, "webinars": web}


def data_checks(t):
    """Health report for the normalized tables: counts, ID match rates and plain-language problems."""
    acc, con, em = t["accounts"], t["contacts"], t["emails"]
    checks = {"rows": {LABELS[k]: int(len(t[k])) for k in FILES}, "problems": [], "notes": []}
    types = acc["account_type"].value_counts().to_dict() if "account_type" in acc.columns else {}
    n_emp = int(types.get("Employer", 0))
    checks["employers"], checks["brokers"] = n_emp, int(types.get("Broker", 0))
    if n_emp == 0:
        checks["problems"].append(f"No accounts with account_type = Employer. Values found in account_type: {sorted(types) or 'none'}.")
    if "account_id" in acc.columns and acc["account_id"].duplicated().any():
        checks["problems"].append(f"{int(acc['account_id'].duplicated().sum())} duplicate account_id value(s) in Accounts.")
    if len(con) and len(acc):
        m = float(con["account_id"].isin(set(acc["account_id"])).mean() * 100)
        checks["contacts_matched_pct"] = round(m, 1)
        if m < 50:
            checks["problems"].append(f"Only {m:.0f}% of contacts point to an account_id that exists in Accounts. Are these files from the same dataset?")
        elif m < 100:
            checks["notes"].append(f"{100 - m:.1f}% of contacts have an account_id that is not in Accounts and are ignored.")
    if len(em) and len(con):
        m = float(em["contact_id"].isin(set(con["contact_id"])).mean() * 100)
        checks["events_matched_pct"] = round(m, 1)
        if m < 50:
            checks["problems"].append(f"Only {m:.0f}% of email events point to a contact_id that exists in Contacts. Are these files from the same dataset?")
        elif m < 100:
            checks["notes"].append(f"{100 - m:.1f}% of email events have a contact_id that is not in Contacts and are ignored.")
    if len(em) == 0:
        checks["notes"].append("There are no email events, so engagement is 0 for every account.")
    return checks
