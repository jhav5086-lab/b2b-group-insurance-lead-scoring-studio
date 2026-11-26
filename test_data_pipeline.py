"""Run with:  python -m pytest tests -q     (or: python tests/test_data_pipeline.py)

Self-contained: builds its own six-table datasets in the app's schema, so it needs no data files.
"""
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data_io  # noqa: E402
import scoring  # noqa: E402

EVAL = pd.Timestamp("2024-09-15")


def make_tables(n_emp=300, n_brk=20, seed=1, id_prefix="ACC"):
    rng = np.random.default_rng(seed)
    brk_ids = [f"BRK-{i:03d}" for i in range(1, n_brk + 1)]
    fit = rng.choice(["Target", "Adjacent", "Non-Target"], n_emp, p=[0.6, 0.25, 0.15])
    prem = np.round(rng.choice([50, 150, 600, 1200, 2500], n_emp, p=[.2, .3, .25, .15, .1]) * 1000 * rng.uniform(.8, 1.5, n_emp), -3)
    emp = pd.DataFrame({
        "account_id": [f"{id_prefix}-{i:05d}" for i in range(1, n_emp + 1)], "account_name": [f"Company {i}" for i in range(n_emp)],
        "account_type": "Employer", "industry": "Healthcare", "industry_fit": fit, "lives": (prem / 1000).astype(int),
        "est_premium": prem, "carrier": rng.choice(["Aetna", "Cigna", "UnitedHealth", "Prudential", ""], n_emp),
        "broker_id": np.where(rng.random(n_emp) < 0.8, rng.choice(brk_ids, n_emp), ""),
        "renewal_date": (EVAL + pd.to_timedelta(rng.integers(0, 365, n_emp), unit="D")).strftime("%Y-%m-%d")})
    brk = pd.DataFrame({"account_id": brk_ids, "account_name": brk_ids, "account_type": "Broker", "industry": "", "industry_fit": "",
                        "lives": 0, "est_premium": 0, "carrier": "", "broker_id": "", "renewal_date": ""})
    accounts = pd.concat([emp, brk], ignore_index=True)
    n_c = rng.choice([2, 3, 4], n_emp)
    cids = [f"CON-{i:06d}" for i in range(int(n_c.sum()))]
    contacts = pd.DataFrame({"contact_id": cids, "account_id": np.repeat(emp["account_id"], n_c).to_numpy(),
                             "email": [f"p{i}@x.com" for i in range(len(cids))], "first_name": "A", "last_name": "B", "persona": "HR",
                             "email_status": rng.choice(["Active", "Unsubscribed"], len(cids), p=[.95, .05]), "domain_type": "Customer"})
    n_ev = rng.poisson(4, len(cids))
    emails = pd.DataFrame({"event_id": [f"E{i}" for i in range(int(n_ev.sum()))], "contact_id": np.repeat(cids, n_ev),
                           "link_category": rng.choice(["generic", "content", "product", "high_intent"], int(n_ev.sum())),
                           "event_date": (EVAL - pd.to_timedelta(rng.integers(0, 180, int(n_ev.sum())), unit="D")).strftime("%Y-%m-%d")})
    wc = rng.choice(cids, max(len(cids) // 5, 1), replace=False)
    webinars = pd.DataFrame({"webinar_id": [f"W{i}" for i in range(len(wc))], "contact_id": wc, "attended": 1,
                             "minutes": rng.integers(5, 60, len(wc)), "polls": rng.integers(0, 3, len(wc)),
                             "questions": rng.integers(0, 2, len(wc)), "downloaded": rng.integers(0, 2, len(wc)), "live": 1})
    qa = rng.choice(emp["account_id"], max(n_emp // 6, 1), replace=False)
    quotes = pd.DataFrame({"quote_id": [f"Q{i}" for i in range(len(qa))], "account_id": qa, "product_code": "GL",
                           "quoted_premium": 1000.0, "quote_date": "2024-08-01", "status": "Open"})
    proposals = pd.DataFrame({"proposal_id": [f"P{i}" for i in range(len(qa) // 2)], "account_id": qa[: len(qa) // 2],
                              "proposal_date": "2024-08-15", "proposed_premium": 1000.0})
    return {"accounts": accounts, "contacts": contacts, "emails": emails, "proposals": proposals, "quotes": quotes, "webinars": webinars}


def as_uploads(tables, keys=None):
    out = {}
    for k in data_io.FILES:
        if keys is not None and k not in keys:
            out[k] = None
        else:
            out[k] = io.BytesIO(tables[k].to_csv(index=False).encode())
    return out


def run(tables, decay=True, hl=45):
    t = data_io.normalize_tables(tables)
    return scoring.run_pipeline(t["accounts"], t["contacts"], t["emails"], EVAL, apply_decay=decay, decay_halflife=hl,
                                proposals=t["proposals"], quotes=t["quotes"], webinars=t["webinars"]), t


DEFAULTS = make_tables(seed=99, id_prefix="DEF")


def test_full_upload_runs_and_counts_employers():
    tb = make_tables(300, seed=1)
    src = data_io.resolve_sources(as_uploads(tb), DEFAULTS)
    assert src["is_upload"] and not src["errors"]
    scored, t = run(src["tables"])
    emp = scored[scored["account_type"] == "Employer"]
    assert len(emp) == 300 and set(emp["tier"]) <= {"Priority MQL", "MQL", "Responder", "Disqualified"}
    assert data_io.data_checks(t)["problems"] == []


def test_partial_upload_is_rejected_and_defaults_are_used_not_mixed():          # the bug you hit
    tb = make_tables(300, seed=2)
    src = data_io.resolve_sources(as_uploads(tb, keys={"accounts", "contacts"}), DEFAULTS)
    assert not src["is_upload"] and src["errors"] and "Email Events" in src["errors"][0]
    assert src["tables"]["accounts"]["account_id"].str.startswith("DEF").all()


def test_empty_email_table_no_longer_crashes():                                  # the KeyError at scoring.py line 61
    tb = make_tables(100, seed=3)
    tb["emails"] = tb["emails"].iloc[0:0]
    scored, _ = run(tb)
    assert (scored.loc[scored["account_type"] == "Employer", "account_engagement"] <= 13).all()   # only quote/proposal bumps remain


def test_email_file_with_no_rows_falls_back_with_message():
    tb = make_tables(100, seed=4)
    tb["emails"] = tb["emails"].iloc[0:0]
    src = data_io.resolve_sources(as_uploads(tb), DEFAULTS)
    assert not src["is_upload"] and any("no rows" in e for e in src["errors"])


def test_messy_labels_are_normalised():
    tb = make_tables(120, seed=5)
    tb["accounts"]["account_type"] = tb["accounts"]["account_type"].str.lower() + " "
    tb["accounts"]["industry_fit"] = tb["accounts"]["industry_fit"].str.upper()
    tb["contacts"]["email_status"] = tb["contacts"]["email_status"].str.upper()
    scored, t = run(tb)
    assert data_io.data_checks(t)["employers"] == 120
    assert (scored["engagement"] if "engagement" in scored else scored["account_engagement"]).max() > 0


def test_missing_optional_columns_and_tables_still_score():
    tb = make_tables(100, seed=6)
    tb["accounts"] = tb["accounts"].drop(columns=["carrier", "broker_id", "renewal_date", "est_premium"])
    for k in ("proposals", "quotes", "webinars"):
        tb[k] = pd.DataFrame()
    src = data_io.resolve_sources(as_uploads(tb, keys={"accounts", "contacts", "emails"}), DEFAULTS)
    assert src["is_upload"] and any("treated as empty" in w for w in src["warnings"])
    scored, _ = run(src["tables"])
    assert len(scored) == 120 and scored["total_score"].notna().all()


def test_blank_and_garbage_values():
    tb = make_tables(100, seed=7)
    tb["accounts"].loc[:20, "carrier"] = None
    tb["accounts"].loc[10:30, "broker_id"] = None
    tb["accounts"].loc[:15, "renewal_date"] = "not a date"
    tb["accounts"].loc[5:9, "est_premium"] = "n/a"
    tb["emails"].loc[:10, "event_date"] = "garbage"
    scored, _ = run(tb)
    assert len(scored) == 120 and scored["fit_score"].notna().all()


def test_column_order_and_extra_columns_do_not_matter():
    tb = make_tables(100, seed=8)
    tb["accounts"] = tb["accounts"][list(reversed(tb["accounts"].columns))].assign(extra_note="x")
    a, _ = run(tb)
    b, _ = run(make_tables(100, seed=8))
    assert a.sort_values("account_id")["tier"].tolist() == b.sort_values("account_id")["tier"].tolist()


def test_numeric_ids_keep_matching():
    tb = make_tables(80, seed=9)
    m = {a: str(1000 + i) for i, a in enumerate(tb["accounts"]["account_id"])}
    tb["accounts"]["account_id"] = tb["accounts"]["account_id"].map(m)
    tb["contacts"]["account_id"] = tb["contacts"]["account_id"].map(m)
    tb["quotes"]["account_id"] = tb["quotes"]["account_id"].map(m)
    tb["proposals"]["account_id"] = tb["proposals"]["account_id"].map(m)
    src = data_io.resolve_sources(as_uploads(tb), DEFAULTS)
    t = data_io.normalize_tables(src["tables"])
    assert data_io.data_checks(t).get("contacts_matched_pct") == 100.0


def test_files_from_different_datasets_are_flagged():
    a, b = make_tables(100, seed=10, id_prefix="AAA"), make_tables(100, seed=11, id_prefix="BBB")
    mixed = {**a, "contacts": b["contacts"]}
    t = data_io.normalize_tables(data_io.resolve_sources(as_uploads(mixed), DEFAULTS)["tables"])
    assert any("same dataset" in p for p in data_io.data_checks(t)["problems"])


def test_no_employers_and_duplicates_are_flagged():
    tb = make_tables(50, seed=12)
    tb["accounts"]["account_type"] = "Broker"
    assert data_io.data_checks(data_io.normalize_tables(tb))["problems"]
    tb = make_tables(50, seed=13)
    tb["accounts"] = pd.concat([tb["accounts"], tb["accounts"].head(3)], ignore_index=True)
    assert any("duplicate" in p for p in data_io.data_checks(data_io.normalize_tables(tb))["problems"])


def test_missing_required_columns_fall_back_with_a_clear_message():
    tb = make_tables(60, seed=14)
    tb["accounts"] = tb["accounts"].drop(columns=["account_type"])
    src = data_io.resolve_sources(as_uploads(tb), DEFAULTS)
    assert not src["is_upload"] and any("account_type" in e for e in src["errors"])


def test_optional_default_tables_never_leak_into_uploads():
    tb = make_tables(60, seed=15)
    src = data_io.resolve_sources(as_uploads(tb, keys={"accounts", "contacts", "emails"}), DEFAULTS)
    assert src["is_upload"] and all(src["tables"][k].empty for k in data_io.OPTIONAL)


def test_missing_default_folder_is_reported(tmp_path=Path("/tmp/no_such_dir_xyz")):
    tables, found, missing = data_io.load_defaults(tmp_path)
    assert found == [] and len(missing) == 6 and all(df.empty for df in tables.values())


def test_decay_lowers_engagement_and_priority():
    tb = make_tables(400, seed=16)
    on, _ = run(tb, decay=True, hl=30)
    off, _ = run(tb, decay=False, hl=0)
    assert on["account_engagement"].mean() < off["account_engagement"].mean()
    assert (on["tier"] == "Priority MQL").sum() <= (off["tier"] == "Priority MQL").sum()


def test_same_inputs_same_outputs():
    tb = make_tables(200, seed=17)
    a, _ = run(tb)
    b, _ = run(tb)
    assert a["total_score"].tolist() == b["total_score"].tolist() and a["tier"].tolist() == b["tier"].tolist()


def test_scale_20k_accounts_finishes_in_reasonable_time():
    tb = make_tables(20000, n_brk=100, seed=18)
    t0 = time.time()
    scored, _ = run(tb)
    assert len(scored) == 20100
    assert time.time() - t0 < 60


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS  {name}  ({time.time() - t0:.1f}s)")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
