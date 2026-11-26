import pandas as pd
import altair as alt
import vl_convert as vlc


def _chart_to_png(chart: alt.Chart, scale: int = 2) -> bytes:
    """Convert an Altair chart into PNG bytes via Vega-Lite."""
    return vlc.vegalite_to_png(vl_spec=chart.to_json(), scale=scale)


def build_report(scored_df, no_decay_df, eval_date, apply_decay, decay_hl):
    report = {"kpis": {}, "charts": [], "warnings": [], "top10": pd.DataFrame()}
    emp = scored_df[scored_df["account_type"] == "Employer"]

    # KPIs
    report["kpis"] = {
        "employers": len(emp),
        "priority": len(emp[emp["tier"] == "Priority MQL"]),
        "mql": len(emp[emp["tier"] == "MQL"]),
        "avg_score": emp["total_score"].mean(),
    }

    # 1. Tier Chart
    t_counts = emp["tier"].value_counts().reset_index()
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
        .properties(title="Tier Distribution")
    )
    png1 = _chart_to_png(c1)

    if not t_counts.empty:
        top_tier = t_counts.iloc[0]
        t_insight = f"The {top_tier['Tier']} segment is the largest, containing {top_tier['Count']} accounts."
        t_action = f"Recommendation: Direct sales efforts towards the {top_tier['Tier']} segment to maximize potential volume."
    else:
        t_insight = "No employer tier data available."
        t_action = "Recommendation: Verify input data and scoring rules."

    report["charts"].append({
        "key": "tiers",
        "title": "Tier Distribution",
        "png": png1,
        "how": "Shows the count of accounts in each MQL tier.",
        "insight": t_insight,
        "action": t_action,
    })

    # 2. Archetype Chart
    a_counts = emp["archetype"].value_counts().reset_index()
    a_counts.columns = ["Archetype", "Count"]
    c2 = (
        alt.Chart(a_counts)
        .mark_arc(innerRadius=50)
        .encode(
            theta="Count:Q",
            color="Archetype:N",
            tooltip=["Archetype", "Count"],
        )
        .properties(title="Archetype Distribution")
    )
    png2 = _chart_to_png(c2)

    if not a_counts.empty:
        top_arch = a_counts.iloc[0]
        a_insight = f"The most common archetype is {top_arch['Archetype']} with {top_arch['Count']} accounts."
        a_action = f"Recommendation: Investigate why {top_arch['Archetype']} is so prevalent and tailor outreach accordingly."
    else:
        a_insight = "No employer archetype data available."
        a_action = "Recommendation: Check contact and event inputs."

    report["charts"].append({
        "key": "archetypes",
        "title": "Archetype Distribution",
        "png": png2,
        "how": "Shows the distribution of behavioral archetypes.",
        "insight": a_insight,
        "action": a_action,
    })

    # 3. Decay Simulation Chart (Comparison)
    comp = pd.DataFrame({
        "Metric": ["Avg Engagement", "Priority MQLs"],
        "Without Decay": [
            no_decay_df["account_engagement"].mean(),
            len(no_decay_df[no_decay_df["tier"] == "Priority MQL"]),
        ],
        "With Decay": [
            scored_df["account_engagement"].mean(),
            len(scored_df[scored_df["tier"] == "Priority MQL"]),
        ],
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
        .properties(title="Decay Simulation Impact")
    )
    png3 = _chart_to_png(c3)

    report["simulation"] = comp
    report["sim_sentence"] = (
        f"Applying a {decay_hl}-day half-life decay reduces average engagement "
        f"from {comp.iloc[0]['Without Decay']:.1f} to {comp.iloc[0]['With Decay']:.1f}."
    )
    report["charts"].append({
        "key": "decay",
        "title": "Decay Simulation",
        "png": png3,
        "how": "Compares metrics with and without decay applied.",
        "insight": report["sim_sentence"],
        "action": "Recommendation: Use decay to prioritize accounts with recent intent.",
    })

    # Top 10
    report["top10"] = (
        emp[emp["tier"] == "Priority MQL"]
        .sort_values("total_score", ascending=False)
        .head(10)
    )

    return report
