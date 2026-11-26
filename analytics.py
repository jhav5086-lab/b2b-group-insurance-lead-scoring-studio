import pandas as pd
import altair as alt
import vl_convert as vlc


def _chart_to_png(chart: alt.Chart, scale: int = 2) -> bytes:
    """Convert an Altair chart into PNG bytes via Vega-Lite."""
    return vlc.vegalite_to_png(vl_spec=chart.to_json(), scale=scale)


def _employers(df: pd.DataFrame) -> pd.DataFrame:
    """Employer accounts only. Every metric in the report must use this same filter."""
    if df is not None and "account_type" in df.columns:
        return df[df["account_type"] == "Employer"]
    return df if df is not None else pd.DataFrame()


def _mean(df: pd.DataFrame, col: str) -> float:
    return float(df[col].mean()) if len(df) and col in df.columns else 0.0


def _count_tier(df: pd.DataFrame, tier: str) -> int:
    return int((df["tier"] == tier).sum()) if len(df) and "tier" in df.columns else 0


def build_decay_comparison(scored_df: pd.DataFrame, baseline_df: pd.DataFrame):
    """Build side-by-side before/after decay metrics, deltas, and difference-driven recommendations."""
    emp_after = _employers(scored_df)
    emp_before = _employers(baseline_df)

    avg_eng_before = _mean(emp_before, "account_engagement")
    avg_eng_after = _mean(emp_after, "account_engagement")

    avg_score_before = _mean(emp_before, "total_score")
    avg_score_after = _mean(emp_after, "total_score")

    prio_before = _count_tier(emp_before, "Priority MQL")
    prio_after = _count_tier(emp_after, "Priority MQL")

    mql_before = _count_tier(emp_before, "MQL")
    mql_after = _count_tier(emp_after, "MQL")

    resp_before = _count_tier(emp_before, "Responder")
    resp_after = _count_tier(emp_after, "Responder")

    comp_df = pd.DataFrame([
        {
            "Metric": "Avg Account Engagement",
            "Before Decay": round(avg_eng_before, 2),
            "After Decay": round(avg_eng_after, 2),
            "Difference": round(avg_eng_after - avg_eng_before, 2),
        },
        {
            "Metric": "Avg Total Score",
            "Before Decay": round(avg_score_before, 2),
            "After Decay": round(avg_score_after, 2),
            "Difference": round(avg_score_after - avg_score_before, 2),
        },
        {
            "Metric": "Priority MQL Accounts",
            "Before Decay": prio_before,
            "After Decay": prio_after,
            "Difference": prio_after - prio_before,
        },
        {
            "Metric": "MQL Accounts",
            "Before Decay": mql_before,
            "After Decay": mql_after,
            "Difference": mql_after - mql_before,
        },
        {
            "Metric": "Responder Accounts",
            "Before Decay": resp_before,
            "After Decay": resp_after,
            "Difference": resp_after - resp_before,
        },
    ])

    prio_diff = prio_after - prio_before
    mql_diff = mql_after - mql_before
    eng_diff = avg_eng_after - avg_eng_before

    recommendations = []
    if prio_diff < 0:
        recommendations.append(
            f"Priority MQL count dropped by {abs(prio_diff):,} accounts after decay: focus immediate outbound reps on the {prio_after:,} accounts that remain Priority MQLs due to recent engagement."
        )
    elif prio_diff > 0:
        recommendations.append(
            f"Priority MQL count increased by {prio_diff:,} accounts after decay: verify recent activity timestamps and route newly qualified accounts quickly."
        )
    else:
        recommendations.append(
            f"Priority MQL count is unchanged ({prio_after:,} accounts) between before and after decay: high-intent accounts hold strong recent signals or strong fit/quote/proposal anchors."
        )

    if mql_diff < 0:
        recommendations.append(
            f"MQL count changed by {mql_diff:+,} ({mql_before:,} -> {mql_after:,}): move accounts that lost MQL status from direct sales queues into re-engagement email or webinar nurture."
        )
    else:
        recommendations.append(
            f"MQL count changed by {mql_diff:+,} ({mql_before:,} -> {mql_after:,}): review accounts near the 50-point MQL threshold for targeted multi-contact outreach."
        )

    if eng_diff < 0:
        recommendations.append(
            f"Average account engagement shifted by {eng_diff:.2f} ({avg_eng_before:.2f} -> {avg_eng_after:.2f}): accounts with older email clicks are discounting stale touchpoints, so prioritize fresh outreach over historical opens."
        )
    else:
        recommendations.append(
            f"Average account engagement shifted by {eng_diff:+.2f} ({avg_eng_before:.2f} -> {avg_eng_after:.2f}): recent activity or non-decaying signals (quotes/proposals/webinars) keep engagement steady."
        )

    summary = (
        f"Before vs. After Decay: Avg Engagement {avg_eng_before:.2f} -> {avg_eng_after:.2f} ({eng_diff:+.2f}), "
        f"Avg Score {avg_score_before:.2f} -> {avg_score_after:.2f} ({avg_score_after - avg_score_before:+.2f}), "
        f"Priority MQLs {prio_before:,} -> {prio_after:,} ({prio_diff:+,})."
    )

    return comp_df, summary, recommendations


def build_report(scored_df, no_decay_df, eval_date, apply_decay, decay_hl):
    report = {
        "kpis": {},
        "charts": [],
        "warnings": [],
        "top10": pd.DataFrame(),
        "decay_comparison": pd.DataFrame(),
        "decay_summary": "",
        "decay_recommendations": [],
    }

    # Guard: nothing to report (e.g. run_pipeline returned an empty frame with no columns)
    if scored_df is None or scored_df.empty or "tier" not in scored_df.columns:
        report["kpis"] = {"employers": 0, "priority": 0, "mql": 0, "avg_score": 0.0}
        report["warnings"].append("No scored accounts available. Check that Accounts, Contacts and Email Events were loaded.")
        return report

    emp = _employers(scored_df)
    emp_before = _employers(no_decay_df)

    report["kpis"] = {
        "employers": len(emp),
        "priority": _count_tier(emp, "Priority MQL"),
        "mql": _count_tier(emp, "MQL"),
        "avg_score": _mean(emp, "total_score"),
    }

    # 1. Tier Chart
    t_counts = emp["tier"].value_counts().reset_index() if len(emp) else pd.DataFrame(columns=["Tier", "Count"])
    if not t_counts.empty:
        t_counts.columns = ["Tier", "Count"]
    c1 = (
        alt.Chart(t_counts)
        .mark_bar()
        .encode(
            x=alt.X("Tier:N", sort="-y"),
            y="Count:Q",
            color="Tier:N",
            tooltip=["Tier", "Count"],
        )
        .properties(title="Tier Distribution", width=480, height=220)
    )
    png1 = _chart_to_png(c1)

    if not t_counts.empty:
        top_tier = t_counts.iloc[0]
        t_insight = f"The {top_tier['Tier']} segment is the largest, containing {int(top_tier['Count']):,} accounts."
        t_action = f"Direct sales and marketing execution toward the {top_tier['Tier']} segment based on account volume."
    else:
        t_insight = "No employer tier data available."
        t_action = "Verify account inputs and scoring rules."

    report["charts"].append({
        "key": "tiers",
        "title": "Tier Distribution",
        "png": png1,
        "how": "Shows the count of employer accounts in each lead tier.",
        "insight": t_insight,
        "action": t_action,
    })

    # 2. Archetype Chart
    a_counts = emp["archetype"].value_counts().reset_index() if len(emp) else pd.DataFrame(columns=["Archetype", "Count"])
    if not a_counts.empty:
        a_counts.columns = ["Archetype", "Count"]
    c2 = (
        alt.Chart(a_counts)
        .mark_arc(innerRadius=45)
        .encode(
            theta="Count:Q",
            color="Archetype:N",
            tooltip=["Archetype", "Count"],
        )
        .properties(title="Archetype Distribution", width=360, height=220)
    )
    png2 = _chart_to_png(c2)

    if not a_counts.empty:
        top_arch = a_counts.iloc[0]
        a_insight = f"The most common archetype is {top_arch['Archetype']} with {int(top_arch['Count']):,} accounts."
        a_action = f"Tailor outreach messaging to address the {top_arch['Archetype']} segment."
    else:
        a_insight = "No employer archetype data available."
        a_action = "Check contact and engagement event inputs."

    report["charts"].append({
        "key": "archetypes",
        "title": "Archetype Distribution",
        "png": png2,
        "how": "Shows the distribution of employer behavioral archetypes.",
        "insight": a_insight,
        "action": a_action,
    })

    # 3. Decay Simulation Chart (Comparison)
    # FIX: both sides now use Employer accounts only, matching build_decay_comparison and the KPIs.
    before_eng = _mean(emp_before, "account_engagement")
    after_eng = _mean(emp, "account_engagement")
    before_prio = _count_tier(emp_before, "Priority MQL")
    after_prio = _count_tier(emp, "Priority MQL")

    comp = pd.DataFrame({
        "Metric": ["Avg Engagement", "Priority MQLs"],
        "Without Decay": [before_eng, before_prio],
        "With Decay": [after_eng, after_prio],
    })
    comp_melt = comp.melt("Metric", var_name="Condition", value_name="Value")
    c3 = (
        alt.Chart(comp_melt)
        .mark_bar()
        .encode(
            x="Metric:N",
            y="Value:Q",
            color="Condition:N",
            xOffset="Condition:N",
        )
        .properties(title="Decay Simulation Impact", width=480, height=220)
    )
    png3 = _chart_to_png(c3)

    if not apply_decay or decay_hl == 0:
        sim_sentence = f"Decay is 0 / off, so average engagement remains {after_eng:.1f} and Priority MQLs remain {after_prio:,}."
    else:
        sim_sentence = (
            f"Applying a {decay_hl}-day half-life decay changes average engagement from {before_eng:.1f} to {after_eng:.1f} "
            f"and Priority MQLs from {before_prio:,} to {after_prio:,}."
        )

    report["simulation"] = comp
    report["sim_sentence"] = sim_sentence
    report["charts"].append({
        "key": "decay",
        "title": "Decay Simulation",
        "png": png3,
        "how": "Compares metrics before decay vs. after applying the current decay setting.",
        "insight": sim_sentence,
        "action": "Use time decay to prioritize accounts with recent engagement over stale activity.",
    })

    comp_df, summary, recs = build_decay_comparison(scored_df, no_decay_df)
    report["decay_comparison"] = comp_df
    report["decay_summary"] = summary
    report["decay_recommendations"] = recs

    # Top 10: highest score first, ties broken by estimated premium (as the PDF states)
    if len(emp):
        report["top10"] = (
            emp[emp["tier"] == "Priority MQL"]
            .sort_values(["total_score", "est_premium"], ascending=[False, False])
            .head(10)
        )

    return report
