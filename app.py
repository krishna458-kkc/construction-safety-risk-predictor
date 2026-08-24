"""Construction Safety Intelligence Platform.

Phase 3A — Real Company Safety Intelligence.
A unified predictive safety intelligence and risk management platform.
"""

from datetime import date, datetime, timezone
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.company_analytics import (
    calculate_activity_risk_metrics,
    calculate_ppe_analysis,
    calculate_project_comparison,
    calculate_site_kpis,
    filter_by_date_range,
    safe_parse_datetime,
)
from src.data_storage import (
    INCIDENT_RECORD_COLUMNS,
    INCIDENT_RECORDS_PATH,
    PREDICTION_HISTORY_COLUMNS,
    PREDICTION_HISTORY_PATH,
    PROJECTS_PATH,
    ensure_company_data_stores,
    record_prediction,
)
from src.predictor import predict
from src.recommendations import get_recommendations
from src.toolbox_talk import get_toolbox_topics
from src.ui.charts import (
    MONO_ACCENT_SCALE,
    RISK_COLOR_DISCRETE_MAP,
    SAFETY_ALERT_SCALE,
    style_mercury_chart,
)
from src.ui.components import (
    PAGES,
    render_empty_state,
    render_footer,
    render_global_header,
    render_metric_card,
    render_page_hero,
    render_risk_badge,
    render_section_heading,
    render_top_navigation,
)
from src.ui.theme import (
    COLOR_ACCENT_COBALT,
    COLOR_BORDER_LIGHT,
    COLOR_SURFACE,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    SAFETY_COLORS,
    SAFETY_COLORS_SOFT,
    inject_mercury_css,
)

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Construction Safety Intelligence",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize company CSV data stores
ensure_company_data_stores()

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

if "current_page" not in st.session_state:
    st.session_state.current_page = "Overview"

if "selected_project" not in st.session_state:
    st.session_state.selected_project = "Metro Tower Expansion — Phase 2"

if "last_prediction_result" not in st.session_state:
    st.session_state.last_prediction_result = None

# Filter state persistence for Records
if "records_filter_reset" not in st.session_state:
    st.session_state.records_filter_reset = 0

# ==============================================================================
# DATA LOADING HELPERS
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "incidents.csv"


@st.cache_data
def load_historical_incidents() -> pd.DataFrame:
    """Load benchmark historical incident dataset."""
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_prediction_history() -> pd.DataFrame:
    """Load company prediction history from data/prediction_history.csv."""
    if not PREDICTION_HISTORY_PATH.exists():
        return pd.DataFrame(columns=PREDICTION_HISTORY_COLUMNS)
    try:
        df = pd.read_csv(PREDICTION_HISTORY_PATH)
        return df
    except Exception:
        return pd.DataFrame(columns=PREDICTION_HISTORY_COLUMNS)


def load_actual_incident_records() -> pd.DataFrame:
    """Load actual company incident records from data/incident_records.csv."""
    if not INCIDENT_RECORDS_PATH.exists():
        return pd.DataFrame(columns=INCIDENT_RECORD_COLUMNS)
    try:
        df = pd.read_csv(INCIDENT_RECORDS_PATH)
        return df
    except Exception:
        return pd.DataFrame(columns=INCIDENT_RECORD_COLUMNS)


def load_projects_data() -> pd.DataFrame:
    """Load projects registry from data/projects.csv."""
    if not PROJECTS_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(PROJECTS_PATH)
        return df
    except Exception:
        return pd.DataFrame()


try:
    incidents_df = load_historical_incidents()
    data_loaded = not incidents_df.empty
    data_error = None
except Exception as err:
    incidents_df = pd.DataFrame()
    data_loaded = False
    data_error = str(err)


# ==============================================================================
# REUSABLE UI WIDGETS
# ==============================================================================

def render_date_filter_control(key_prefix: str = "df") -> Tuple[str, Optional[date], Optional[date]]:
    """Render a clean Mercury-styled date filter selector with optional custom range pickers."""
    col_preset, col_custom = st.columns([1.2, 2.8])
    with col_preset:
        period = st.selectbox(
            "Date Range Filter:",
            ["All time", "Last 7 days", "Last 30 days", "This month", "Previous month", "Custom range"],
            index=0,
            key=f"{key_prefix}_period_select",
        )

    start_date = None
    end_date = None
    if period == "Custom range":
        with col_custom:
            c1, c2 = st.columns(2)
            with c1:
                start_date = st.date_input("Start Date", value=datetime.now(timezone.utc).date(), key=f"{key_prefix}_start")
            with c2:
                end_date = st.date_input("End Date", value=datetime.now(timezone.utc).date(), key=f"{key_prefix}_end")

    return period, start_date, end_date


# ==============================================================================
# GLOBAL APPLICATION SHELL
# ==============================================================================

inject_mercury_css()
render_global_header(current_project=st.session_state.selected_project)
current_page = render_top_navigation()


# ==============================================================================
# 1. OVERVIEW
# ==============================================================================

if current_page == "Overview":
    render_page_hero(
        title="Construction Safety Intelligence",
        subtitle="Real-time predictive risk analytics and historical safety benchmarking for jobsites.",
        tagline="SAFETY INTELLIGENCE PLATFORM",
    )

    overview_perspective = st.radio(
        "Overview Perspective",
        ["Site Safety", "Historical Benchmark"],
        horizontal=True,
        label_visibility="collapsed",
        key="overview_perspective_radio",
    )

    if overview_perspective == "Site Safety":
        render_section_heading(
            "Site Safety Operations",
            "Company-specific safety assessments recorded at active construction sites.",
        )

        site_history_raw = load_prediction_history()

        if site_history_raw.empty:
            render_empty_state(
                title="No site activity recorded yet",
                message=(
                    "Run a risk assessment in Risk Predictor to begin building your site's "
                    "safety intelligence and operational risk history."
                ),
                icon="🦺",
            )
        else:
            # Date Filter Control
            period, s_date, e_date = render_date_filter_control("ov_site")
            site_history = filter_by_date_range(site_history_raw, period, s_date, e_date)

            if site_history.empty:
                render_empty_state(
                    title="No assessments found for selected period",
                    message="No site evaluations match the current date filter. Select 'All time' or widen the date range.",
                    icon="📅",
                )
            else:
                kpis = calculate_site_kpis(site_history)

                # Primary KPI row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    render_metric_card(
                        "Total Assessments",
                        kpis["total_assessments"],
                        subtitle="Recorded site evaluations",
                        variant="accent",
                    )
                with col2:
                    render_metric_card(
                        "High / Critical Risks",
                        kpis["high_critical_count"],
                        subtitle=f"{kpis['high_critical_pct']:.1f}% of assessments",
                        variant="high",
                    )
                with col3:
                    render_metric_card(
                        "Critical Alerts",
                        kpis["critical_risk_count"],
                        subtitle="Immediate stop-work hazard",
                        variant="critical",
                    )
                with col4:
                    render_metric_card(
                        "Observed Avg PPE",
                        f"{kpis['avg_ppe_compliance']:.1f}%",
                        subtitle="Observed crew compliance",
                        variant="medium",
                    )

                # Secondary KPI row
                st.markdown("<div style='margin: 0.75rem 0;'></div>", unsafe_allow_html=True)
                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1:
                    render_metric_card(
                        "Average Risk Score",
                        f"{kpis['avg_risk_score']:.1f} / 100",
                        subtitle="Cohort average score",
                        variant="default",
                    )
                with sc2:
                    render_metric_card(
                        "Model Confidence",
                        f"{kpis['avg_confidence']:.1f}%",
                        subtitle="Average classifier certainty",
                        variant="default",
                    )
                with sc3:
                    render_metric_card(
                        "Primary Activity",
                        kpis["most_assessed_activity"],
                        subtitle="Most frequent assessment",
                        variant="default",
                    )
                with sc4:
                    render_metric_card(
                        "Latest Assessment",
                        kpis["latest_assessment_date"],
                        subtitle="Most recent timestamp",
                        variant="default",
                    )

                st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

                # Visual charts
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    render_section_heading("Site Risk Distribution", "By predicted risk level")
                    risk_counts_df = pd.DataFrame(
                        {
                            "Risk Level": list(kpis["risk_counts"].keys()),
                            "Assessments": list(kpis["risk_counts"].values()),
                        }
                    )
                    fig_site_risk = px.bar(
                        risk_counts_df,
                        x="Risk Level",
                        y="Assessments",
                        color="Risk Level",
                        color_discrete_map=RISK_COLOR_DISCRETE_MAP,
                        text="Assessments",
                    )
                    fig_site_risk.update_traces(textposition="outside")
                    style_mercury_chart(fig_site_risk)
                    st.plotly_chart(fig_site_risk, use_container_width=True)

                with col_chart2:
                    render_section_heading("Site Risk Score Timeline", "Risk evaluations over time")
                    dt_series = safe_parse_datetime(site_history["timestamp"])
                    valid_time_mask = dt_series.notna()
                    if valid_time_mask.any() and "risk_score" in site_history.columns:
                        timeline_df = site_history[valid_time_mask].copy()
                        timeline_df["parsed_time"] = dt_series[valid_time_mask]
                        timeline_df = timeline_df.sort_values("parsed_time")

                        fig_trend = px.line(
                            timeline_df,
                            x="parsed_time",
                            y="risk_score",
                            markers=True,
                            hover_data=["activity_type", "predicted_risk_level", "project/site"],
                        )
                        fig_trend.update_traces(
                            line_color=COLOR_ACCENT_COBALT,
                            marker_color=SAFETY_COLORS["HIGH"],
                            marker_size=7,
                        )
                        fig_trend.update_layout(xaxis_title="Assessment Date & Time", yaxis_title="Risk Score (0-100)")
                        style_mercury_chart(fig_trend)
                        st.plotly_chart(fig_trend, use_container_width=True)
                    else:
                        st.info("Insufficient timestamp data to generate timeline trend.")

                render_section_heading(
                    "Recent Safety Assessments",
                    "Latest jobsite risk evaluations logged from Risk Predictor.",
                )
                display_cols = [
                    "timestamp", "project/site", "activity_type", "location_type",
                    "predicted_risk_level", "risk_score", "model_confidence", "ppe_compliance_pct", "description"
                ]
                avail_cols = [c for c in display_cols if c in site_history.columns]
                display_df = site_history[avail_cols].tail(10).sort_values("timestamp", ascending=False)
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                )

    else:
        render_section_heading(
            "Historical Incident Benchmark",
            "500-incident industry benchmark dataset. Used for reference and baseline training.",
        )

        if not data_loaded:
            st.error("Historical incident dataset could not be loaded.")
            if data_error:
                st.code(data_error)
        else:
            total_incidents = len(incidents_df)
            crit_cnt = int((incidents_df["risk_level"] == "CRITICAL").sum())
            high_cnt = int((incidents_df["risk_level"] == "HIGH").sum())
            med_cnt = int((incidents_df["risk_level"] == "MEDIUM").sum())
            low_cnt = int((incidents_df["risk_level"] == "LOW").sum())

            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            with kpi1:
                render_metric_card("Total Baseline", total_incidents, "Historical records", variant="accent")
            with kpi2:
                render_metric_card("Critical Risk", crit_cnt, f"{(crit_cnt / total_incidents * 100):.1f}% share", variant="critical")
            with kpi3:
                render_metric_card("High Risk", high_cnt, f"{(high_cnt / total_incidents * 100):.1f}% share", variant="high")
            with kpi4:
                render_metric_card("Medium Risk", med_cnt, f"{(med_cnt / total_incidents * 100):.1f}% share", variant="medium")
            with kpi5:
                render_metric_card("Low Risk", low_cnt, f"{(low_cnt / total_incidents * 100):.1f}% share", variant="low")

            st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

            col_b1, col_b2 = st.columns(2)
            with col_b1:
                render_section_heading("Benchmark Risk Distribution", "Incidents by category")
                b_risk_counts = (
                    incidents_df["risk_level"]
                    .value_counts()
                    .reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0)
                    .reset_index()
                )
                b_risk_counts.columns = ["Risk Level", "Incidents"]
                fig_b_risk = px.bar(
                    b_risk_counts,
                    x="Risk Level",
                    y="Incidents",
                    color="Risk Level",
                    color_discrete_map=RISK_COLOR_DISCRETE_MAP,
                    text="Incidents",
                )
                fig_b_risk.update_traces(textposition="outside")
                style_mercury_chart(fig_b_risk)
                st.plotly_chart(fig_b_risk, use_container_width=True)

            with col_b2:
                render_section_heading("Severity Breakdown", "Incident consequence severity")
                sev_counts = incidents_df["severity"].value_counts().reset_index()
                sev_counts.columns = ["Severity", "Incidents"]
                fig_sev = px.pie(
                    sev_counts,
                    names="Severity",
                    values="Incidents",
                    hole=0.45,
                    color_discrete_sequence=[
                        SAFETY_COLORS["LOW"],
                        SAFETY_COLORS["MEDIUM"],
                        SAFETY_COLORS["HIGH"],
                        SAFETY_COLORS["CRITICAL"],
                        "#70707d",
                    ],
                )
                style_mercury_chart(fig_sev)
                st.plotly_chart(fig_sev, use_container_width=True)

            col_b3, col_b4 = st.columns(2)
            with col_b3:
                render_section_heading("Incidents by Construction Activity", "Historical frequency")
                act_counts = incidents_df["activity_type"].value_counts().reset_index()
                act_counts.columns = ["Activity", "Incidents"]
                fig_act = px.bar(
                    act_counts,
                    x="Incidents",
                    y="Activity",
                    orientation="h",
                    color="Incidents",
                    color_continuous_scale=MONO_ACCENT_SCALE,
                )
                fig_act.update_layout(coloraxis_showscale=False)
                style_mercury_chart(fig_act)
                st.plotly_chart(fig_act, use_container_width=True)

            with col_b4:
                render_section_heading("Daily Incident Timeline", "Historical event series")
                daily_df = (
                    incidents_df.dropna(subset=["date"])
                    .groupby("date")
                    .size()
                    .reset_index(name="Incidents")
                    .sort_values("date")
                )
                fig_time = px.line(
                    daily_df,
                    x="date",
                    y="Incidents",
                    markers=True,
                )
                fig_time.update_traces(
                    line_color=COLOR_ACCENT_COBALT,
                    marker_color=SAFETY_COLORS["MEDIUM"],
                    marker_size=5,
                )
                style_mercury_chart(fig_time)
                st.plotly_chart(fig_time, use_container_width=True)


# ==============================================================================
# 2. RISK PREDICTOR
# ==============================================================================

elif current_page == "Risk Predictor":
    render_page_hero(
        title="Construction Risk Predictor",
        subtitle="Evaluate planned construction tasks before execution using our trained risk model.",
        tagline="PREDICTIVE HAZARD ENGINE",
    )

    if data_loaded:
        activity_options = sorted(incidents_df["activity_type"].dropna().unique().tolist())
        location_options = sorted(incidents_df["location_type"].dropna().unique().tolist())
        weather_options = sorted(incidents_df["weather"].dropna().unique().tolist())
        shift_options = sorted(incidents_df["shift"].dropna().unique().tolist())
    else:
        activity_options = [
            "Working at Height", "Lifting", "Scaffolding", "Excavation",
            "Electrical Work", "Material Handling", "Welding",
            "Confined Space", "Vehicle Movement", "Housekeeping",
        ]
        location_options = ["Roof", "Electrical Room", "Excavation Area", "Warehouse", "Loading Area"]
        weather_options = ["Clear", "Rain", "Adverse"]
        shift_options = ["Day", "Night"]

    st.markdown('<div class="cs-card">', unsafe_allow_html=True)
    render_section_heading("Planned Activity Parameters", "Input task details for pre-work risk calculation")

    form_col1, form_col2 = st.columns(2)

    with form_col1:
        st.caption("1. ACTIVITY & LOCATION")
        activity_type = st.selectbox("Activity Type", activity_options, index=0)
        location_type = st.selectbox("Location Type", location_options, index=0)

        st.caption("2. ENVIRONMENTAL CONDITIONS")
        weather = st.selectbox("Weather Condition", weather_options, index=0)
        shift = st.selectbox("Operational Shift", shift_options, index=0)

    with form_col2:
        st.caption("3. CREW DYNAMICS & COMPLIANCE")
        crew_size = st.number_input("Crew Size (Workers)", min_value=1, max_value=120, value=6, step=1)
        ppe_compliance_pct = st.slider("Observed PPE Compliance (%)", min_value=0, max_value=100, value=85, step=1)
        previous_incidents_30d = st.number_input("Site Incidents in Last 30 Days", min_value=0, max_value=20, value=0, step=1)

        st.caption("4. PROJECT CONTEXT & HAZARDS")
        project_site = st.text_input(
            "Project / Jobsite Name",
            value=st.session_state.selected_project,
            placeholder="e.g. Metro Tower — Level 14",
        )

    description = st.text_area(
        "Hazard & Task Description",
        value="",
        placeholder="e.g., Installing exterior curtain-wall panels on perimeter edge without temporary guardrails.",
        height=80,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    predict_btn = st.button("◈ Run Safety Risk Assessment", type="primary", use_container_width=True)

    if predict_btn:
        activity_data = {
            "activity_type": activity_type,
            "location_type": location_type,
            "weather": weather,
            "shift": shift,
            "ppe_compliance_pct": ppe_compliance_pct,
            "previous_incidents_30d": previous_incidents_30d,
            "crew_size": crew_size,
            "description": description,
        }

        try:
            prediction_result = predict(activity_data, strict=True)
            record_prediction(project_site, activity_data, prediction_result)
            st.session_state.last_prediction_result = {
                "activity_data": activity_data,
                "result": prediction_result,
                "project_site": project_site,
            }
        except Exception as pred_err:
            st.error(f"Prediction could not be completed: {pred_err}")

    if st.session_state.last_prediction_result:
        saved = st.session_state.last_prediction_result
        res = saved["result"]
        act_d = saved["activity_data"]
        r_level = res["risk_level"]
        r_score = res["risk_score"]
        conf = res["confidence"]
        conf_pct = conf * 100 if conf <= 1.0 else conf

        st.markdown(
            f'<div class="cs-risk-assessment {r_level.lower()}">',
            unsafe_allow_html=True,
        )

        render_section_heading("Risk Intelligence Assessment Result", f"Assessed for: {act_d['activity_type']} at {act_d['location_type']}")

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            render_metric_card("Predicted Risk Level", r_level, subtitle="Machine learning classification", variant=r_level.lower())
        with res_col2:
            render_metric_card("Risk Score", f"{r_score:.1f} / 100", subtitle="Probability-weighted risk index", variant=r_level.lower())
        with res_col3:
            render_metric_card("Model Confidence", f"{conf_pct:.1f}%", subtitle="Classifier certainty", variant="accent")

        if r_level == "CRITICAL":
            banner_msg = "CRITICAL RISK — Immediate intervention required. Work must NOT proceed until verified engineering controls and supervisor permits are issued."
        elif r_level == "HIGH":
            banner_msg = "HIGH RISK — Elevated hazard exposure. Rigorous control verification and active supervision required before starting task."
        elif r_level == "MEDIUM":
            banner_msg = "MEDIUM RISK — Standard operational hazards present. Verify routine safety controls and conduct pre-task briefing."
        else:
            banner_msg = "LOW RISK — Normal jobsite risk profile. Proceed with standard safety protocols and monitoring."

        st.markdown(
            f'<div class="cs-risk-banner {r_level.lower()}">{banner_msg}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
        why_col, prob_col = st.columns([1.1, 0.9])

        with why_col:
            render_section_heading("Why this risk?", "Key contributing factors from model weights")
            top_factors = res.get("top_factors", [])
            if top_factors:
                for factor in top_factors:
                    if isinstance(factor, dict):
                        f_name = factor.get("label", "Factor")
                        f_contrib = factor.get("contribution")
                        f_dir = factor.get("direction", "contributes to risk")
                        contrib_str = f"magnitude: {f_contrib:.4f}" if f_contrib is not None else ""
                        st.markdown(
                            f"""
                            <div class="cs-factor-item">
                                <div>
                                    <span class="cs-factor-name">{f_name}</span>
                                    <span style="color: {COLOR_TEXT_SECONDARY}; font-size: 0.82rem; margin-left: 0.4rem;">({f_dir})</span>
                                </div>
                                <span class="cs-factor-meta">{contrib_str}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="cs-factor-item"><span class="cs-factor-name">{factor}</span></div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.caption("No specific outlier factors identified for this profile.")

        with prob_col:
            render_section_heading("Risk Probability Distribution", "Class probability breakdown")
            probs = res.get("probabilities", {})
            if probs:
                prob_df = pd.DataFrame(
                    {
                        "Risk Level": list(probs.keys()),
                        "Probability (%)": [v * 100 for v in probs.values()],
                    }
                )
                fig_p = px.bar(
                    prob_df,
                    x="Risk Level",
                    y="Probability (%)",
                    text="Probability (%)",
                    color="Risk Level",
                    color_discrete_map=RISK_COLOR_DISCRETE_MAP,
                )
                fig_p.update_traces(
                    texttemplate="%{text:.1f}%",
                    textposition="outside",
                )
                fig_p.update_layout(showlegend=False)
                style_mercury_chart(fig_p, height=260)
                st.plotly_chart(fig_p, use_container_width=True)

        st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
        rec_col, tool_col = st.columns(2)

        with rec_col:
            render_section_heading("Recommended Preventive Actions", f"Controls for {act_d['activity_type']}")
            recs = get_recommendations(act_d["activity_type"])
            for rec in recs:
                st.markdown(
                    f"""
                    <div class="cs-action-item">
                        <span class="cs-action-icon">✓</span>
                        <span>{rec}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with tool_col:
            render_section_heading("Targeted Toolbox Talk Briefing", "Recommended pre-shift discussion points")
            topics = get_toolbox_topics(act_d["activity_type"])
            for topic in topics:
                st.markdown(
                    f"""
                    <div class="cs-action-item">
                        <span class="cs-action-icon" style="color:{COLOR_ACCENT_COBALT};">🗣</span>
                        <span>{topic}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# 3. ANALYTICS
# ==============================================================================

elif current_page == "Analytics":
    render_page_hero(
        title="Safety Risk Analytics",
        subtitle="Multi-dimensional risk patterns across activities, environmental conditions, and compliance.",
        tagline="INTELLIGENCE DASHBOARD",
    )

    analytics_perspective = st.radio(
        "Analytics Perspective",
        ["Site Analytics", "Historical Benchmark"],
        horizontal=True,
        label_visibility="collapsed",
        key="analytics_perspective_radio",
    )

    if analytics_perspective == "Site Analytics":
        site_history_raw = load_prediction_history()
        if site_history_raw.empty:
            render_empty_state(
                title="No site analytics data available",
                message="Site analytics are generated dynamically from your recorded predictions. Submit assessments in Risk Predictor to populate this dashboard.",
                icon="▥",
            )
        else:
            # Reusable Date Filter Bar
            period, s_date, e_date = render_date_filter_control("an_site")
            site_history = filter_by_date_range(site_history_raw, period, s_date, e_date)

            if site_history.empty:
                render_empty_state(
                    title="No analytics for selected date period",
                    message="No company predictions match the selected date range. Select 'All time' or widen your filter range.",
                    icon="📅",
                )
            else:
                kpis = calculate_site_kpis(site_history)

                # KPI Summary Banner
                render_section_heading("Site Safety Intelligence Summary", f"Analytics for {kpis['total_assessments']} recorded site evaluations")
                k1, k2, k3, k4, k5 = st.columns(5)
                with k1:
                    render_metric_card("Assessments", kpis["total_assessments"], variant="accent")
                with k2:
                    render_metric_card("Avg Risk Score", f"{kpis['avg_risk_score']:.1f}", variant="default")
                with k3:
                    render_metric_card("High/Critical %", f"{kpis['high_critical_pct']:.1f}%", variant="high")
                with k4:
                    render_metric_card("Avg PPE %", f"{kpis['avg_ppe_compliance']:.1f}%", variant="medium")
                with k5:
                    render_metric_card("Confidence", f"{kpis['avg_confidence']:.1f}%", variant="default")

                st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

                # Row 1: Distribution & Trend
                col1, col2 = st.columns(2)
                with col1:
                    render_section_heading("Site Risk Distribution", "Breakdown by predicted risk class")
                    risk_counts_df = pd.DataFrame(
                        {
                            "Risk Level": list(kpis["risk_counts"].keys()),
                            "Assessments": list(kpis["risk_counts"].values()),
                        }
                    )
                    fig_site_risk = px.bar(
                        risk_counts_df,
                        x="Risk Level",
                        y="Assessments",
                        color="Risk Level",
                        color_discrete_map=RISK_COLOR_DISCRETE_MAP,
                        text="Assessments",
                    )
                    fig_site_risk.update_traces(textposition="outside")
                    style_mercury_chart(fig_site_risk)
                    st.plotly_chart(fig_site_risk, use_container_width=True)

                with col2:
                    render_section_heading("Risk Score Timeline & Trend", "Evaluations across timeline")
                    dt_series = safe_parse_datetime(site_history["timestamp"])
                    valid_time = dt_series.notna()
                    if valid_time.any() and "risk_score" in site_history.columns:
                        trend_df = site_history[valid_time].copy()
                        trend_df["parsed_time"] = dt_series[valid_time]
                        trend_df = trend_df.sort_values("parsed_time")

                        fig_trend = px.line(
                            trend_df,
                            x="parsed_time",
                            y="risk_score",
                            markers=True,
                            hover_data=["activity_type", "predicted_risk_level"],
                        )
                        fig_trend.update_traces(
                            line_color=COLOR_ACCENT_COBALT,
                            marker_color=SAFETY_COLORS["HIGH"],
                            marker_size=6,
                        )
                        fig_trend.update_layout(xaxis_title="Date / Time", yaxis_title="Risk Score (0-100)")
                        style_mercury_chart(fig_trend)
                        st.plotly_chart(fig_trend, use_container_width=True)
                    else:
                        st.info("Insufficient timestamp records to plot trend.")

                st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

                # Row 2: Activity Risk & PPE Analysis
                col3, col4 = st.columns(2)
                with col3:
                    render_section_heading("Activity-Specific Risk Breakdown", "Risk metrics per construction task")
                    act_metrics_df = calculate_activity_risk_metrics(site_history)
                    if not act_metrics_df.empty:
                        fig_act_risk = px.bar(
                            act_metrics_df,
                            x="Avg Risk Score",
                            y="Activity",
                            orientation="h",
                            color="Avg Risk Score",
                            color_continuous_scale=SAFETY_ALERT_SCALE,
                            text="Avg Risk Score",
                        )
                        fig_act_risk.update_traces(textposition="outside")
                        fig_act_risk.update_layout(coloraxis_showscale=False)
                        style_mercury_chart(fig_act_risk)
                        st.plotly_chart(fig_act_risk, use_container_width=True)
                        st.dataframe(act_metrics_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No activity records available.")

                with col4:
                    render_section_heading("Observed PPE Compliance Analysis", "Observed relationship with evaluated risk")
                    ppe_data = calculate_ppe_analysis(site_history)
                    ppe_risk_df = ppe_data["avg_ppe_by_risk"]
                    if not ppe_risk_df.empty:
                        fig_ppe_risk = px.bar(
                            ppe_risk_df,
                            x="Risk Level",
                            y="Avg PPE %",
                            color="Risk Level",
                            color_discrete_map=RISK_COLOR_DISCRETE_MAP,
                            text="Avg PPE %",
                        )
                        fig_ppe_risk.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                        fig_ppe_risk.update_layout(showlegend=False)
                        style_mercury_chart(fig_ppe_risk)
                        st.plotly_chart(fig_ppe_risk, use_container_width=True)
                    else:
                        st.info("No PPE compliance data available.")

                    st.caption("Note: Represents observed empirical relationship across jobsite evaluations. Correlation does not imply sole causation.")

                st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

                # Row 3: Project / Site Comparison
                render_section_heading("Project / Jobsite Comparison", "Safety metrics across registered sites")
                proj_comp_df = calculate_project_comparison(site_history)
                if not proj_comp_df.empty:
                    if len(proj_comp_df) > 1:
                        c_p1, c_p2 = st.columns([1.2, 0.8])
                        with c_p1:
                            fig_proj = px.bar(
                                proj_comp_df,
                                x="Project / Site",
                                y="Assessments",
                                color="Avg Risk Score",
                                color_continuous_scale=SAFETY_ALERT_SCALE,
                                text="Assessments",
                            )
                            fig_proj.update_traces(textposition="outside")
                            fig_proj.update_layout(coloraxis_showscale=False)
                            style_mercury_chart(fig_proj)
                            st.plotly_chart(fig_proj, use_container_width=True)
                        with c_p2:
                            st.dataframe(proj_comp_df, use_container_width=True, hide_index=True)
                    else:
                        single_p = proj_comp_df.iloc[0]
                        sp_col1, sp_col2, sp_col3, sp_col4 = st.columns(4)
                        with sp_col1:
                            render_metric_card("Active Project", single_p["Project / Site"], variant="accent")
                        with sp_col2:
                            render_metric_card("Site Assessments", single_p["Assessments"], variant="default")
                        with sp_col3:
                            render_metric_card("Site Avg Risk", f"{single_p['Avg Risk Score']:.1f}", variant="high" if single_p["Avg Risk Score"] >= 50 else "medium")
                        with sp_col4:
                            render_metric_card("Site Avg PPE", f"{single_p['Avg PPE %']:.1f}%", variant="medium")

    else:
        render_section_heading("Historical Benchmark Risk Patterns", "500-record benchmark intelligence")
        if not data_loaded:
            st.error("Historical incident data is not loaded.")
        else:
            high_risk_df = incidents_df[
                incidents_df["risk_level"].isin(["HIGH", "CRITICAL"])
            ].copy()

            col_a1, col_a2 = st.columns(2)
            with col_a1:
                render_section_heading("High-Risk Activity Patterns", "HIGH & CRITICAL incident volume")
                act_counts = high_risk_df["activity_type"].value_counts().reset_index()
                act_counts.columns = ["Activity", "High/Critical Incidents"]
                fig_h_act = px.bar(
                    act_counts,
                    x="High/Critical Incidents",
                    y="Activity",
                    orientation="h",
                    color="High/Critical Incidents",
                    color_continuous_scale=SAFETY_ALERT_SCALE,
                )
                fig_h_act.update_layout(coloraxis_showscale=False)
                style_mercury_chart(fig_h_act)
                st.plotly_chart(fig_h_act, use_container_width=True)

            with col_a2:
                render_section_heading("High-Risk Location Patterns", "Locations with elevated severe hazards")
                loc_counts = high_risk_df["location_type"].value_counts().reset_index()
                loc_counts.columns = ["Location", "High/Critical Incidents"]
                fig_h_loc = px.bar(
                    loc_counts,
                    x="High/Critical Incidents",
                    y="Location",
                    orientation="h",
                    color="High/Critical Incidents",
                    color_continuous_scale=MONO_ACCENT_SCALE,
                )
                fig_h_loc.update_layout(coloraxis_showscale=False)
                style_mercury_chart(fig_h_loc)
                st.plotly_chart(fig_h_loc, use_container_width=True)

            st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
            col_a3, col_a4 = st.columns(2)

            with col_a3:
                render_section_heading("Weather vs Risk Level Matrix", "Cross-tabulated benchmark frequency")
                weather_crosstab = pd.crosstab(
                    incidents_df["weather"],
                    incidents_df["risk_level"],
                ).reindex(columns=["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0)
                st.dataframe(weather_crosstab, use_container_width=True)

            with col_a4:
                render_section_heading("PPE Compliance by Risk Level", "Observed historical average")
                ppe_summary = (
                    incidents_df.groupby("risk_level")["ppe_compliance_pct"]
                    .mean()
                    .reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"])
                    .reset_index()
                )
                ppe_summary.columns = ["Risk Level", "Avg PPE Compliance (%)"]
                fig_ppe_bar = px.bar(
                    ppe_summary,
                    x="Risk Level",
                    y="Avg PPE Compliance (%)",
                    color="Risk Level",
                    color_discrete_map=RISK_COLOR_DISCRETE_MAP,
                    text="Avg PPE Compliance (%)",
                )
                fig_ppe_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_ppe_bar.update_layout(showlegend=False)
                style_mercury_chart(fig_ppe_bar)
                st.plotly_chart(fig_ppe_bar, use_container_width=True)


# ==============================================================================
# 4. RECORDS
# ==============================================================================

elif current_page == "Records":
    render_page_hero(
        title="Safety Data & Records",
        subtitle="Search, filter, inspect, and audit company risk predictions and historical incident logs.",
        tagline="DATA MANAGEMENT",
    )

    records_tab = st.radio(
        "Records View",
        ["Risk Predictions", "Actual Incidents"],
        horizontal=True,
        label_visibility="collapsed",
        key="records_tab_radio",
    )

    if records_tab == "Risk Predictions":
        render_section_heading("Company Risk Predictions", "Logged assessments from data/prediction_history.csv")
        pred_df = load_prediction_history()

        if pred_df.empty:
            render_empty_state(
                title="No risk prediction records found",
                message="Predictions created in the Risk Predictor workspace are saved to company records and will display here.",
                icon="▤",
            )
        else:
            # Multi-Filter Panel
            st.markdown('<div class="cs-card-flat">', unsafe_allow_html=True)
            st.caption("FILTER ASSESSMENTS")

            rf_r = st.session_state.records_filter_reset

            col_f1, col_f2, col_f3, col_f4 = st.columns(4)
            with col_f1:
                period_filter = st.selectbox(
                    "Date Range Preset",
                    ["All time", "Last 7 days", "Last 30 days", "This month", "Previous month", "Custom range"],
                    index=0,
                    key=f"rec_p_period_{rf_r}",
                )
            with col_f2:
                proj_opts = ["All"] + sorted([p for p in pred_df["project/site"].dropna().unique().tolist() if str(p).strip()])
                sel_proj = st.selectbox("Project / Site", proj_opts, index=0, key=f"rec_p_proj_{rf_r}")
            with col_f3:
                act_opts = ["All"] + sorted(pred_df["activity_type"].dropna().unique().tolist())
                sel_act = st.selectbox("Activity Type", act_opts, index=0, key=f"rec_p_act_{rf_r}")
            with col_f4:
                risk_opts = ["All", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
                sel_risk = st.selectbox("Risk Level", risk_opts, index=0, key=f"rec_p_risk_{rf_r}")

            col_f5, col_f6, col_f7, col_f8 = st.columns(4)
            with col_f5:
                loc_opts = ["All"] + sorted(pred_df["location_type"].dropna().unique().tolist()) if "location_type" in pred_df.columns else ["All"]
                sel_loc = st.selectbox("Location Type", loc_opts, index=0, key=f"rec_p_loc_{rf_r}")
            with col_f6:
                w_opts = ["All"] + sorted(pred_df["weather"].dropna().unique().tolist()) if "weather" in pred_df.columns else ["All"]
                sel_w = st.selectbox("Weather", w_opts, index=0, key=f"rec_p_w_{rf_r}")
            with col_f7:
                sh_opts = ["All"] + sorted(pred_df["shift"].dropna().unique().tolist()) if "shift" in pred_df.columns else ["All"]
                sel_sh = st.selectbox("Shift", sh_opts, index=0, key=f"rec_p_sh_{rf_r}")
            with col_f8:
                sort_choice = st.selectbox(
                    "Sort By",
                    [
                        "Timestamp (Newest first)",
                        "Timestamp (Oldest first)",
                        "Risk Score (High to Low)",
                        "Risk Score (Low to High)",
                        "PPE Compliance (High to Low)",
                    ],
                    index=0,
                    key=f"rec_p_sort_{rf_r}",
                )

            custom_s = None
            custom_e = None
            if period_filter == "Custom range":
                c_c1, c_c2 = st.columns(2)
                with c_c1:
                    custom_s = st.date_input("Start Date", value=datetime.now(timezone.utc).date(), key=f"rec_p_cs_{rf_r}")
                with c_c2:
                    custom_e = st.date_input("End Date", value=datetime.now(timezone.utc).date(), key=f"rec_p_ce_{rf_r}")

            col_s1, col_s2 = st.columns([3.5, 0.5])
            with col_s1:
                search_query = st.text_input(
                    "Search Task / Hazard Description",
                    placeholder="Search by keywords, description, or project name...",
                    key=f"rec_p_search_{rf_r}",
                )
            with col_s2:
                st.markdown("<div style='margin-top: 1.85rem;'></div>", unsafe_allow_html=True)
                if st.button("Reset", key="rec_p_reset_btn", use_container_width=True):
                    st.session_state.records_filter_reset += 1
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

            # Apply Filters
            filtered_pred = pred_df.copy()
            filtered_pred = filter_by_date_range(filtered_pred, period_filter, custom_s, custom_e)

            if sel_proj != "All":
                filtered_pred = filtered_pred[filtered_pred["project/site"] == sel_proj]
            if sel_act != "All":
                filtered_pred = filtered_pred[filtered_pred["activity_type"] == sel_act]
            if sel_risk != "All":
                filtered_pred = filtered_pred[filtered_pred["predicted_risk_level"] == sel_risk]
            if sel_loc != "All" and "location_type" in filtered_pred.columns:
                filtered_pred = filtered_pred[filtered_pred["location_type"] == sel_loc]
            if sel_w != "All" and "weather" in filtered_pred.columns:
                filtered_pred = filtered_pred[filtered_pred["weather"] == sel_w]
            if sel_sh != "All" and "shift" in filtered_pred.columns:
                filtered_pred = filtered_pred[filtered_pred["shift"] == sel_sh]

            if search_query.strip():
                q = search_query.strip().lower()
                filtered_pred = filtered_pred[
                    filtered_pred["description"].fillna("").str.lower().str.contains(q)
                    | filtered_pred["project/site"].fillna("").str.lower().str.contains(q)
                    | filtered_pred["activity_type"].fillna("").str.lower().str.contains(q)
                ]

            # Apply Sorting
            if sort_choice == "Timestamp (Newest first)":
                filtered_pred = filtered_pred.sort_values("timestamp", ascending=False)
            elif sort_choice == "Timestamp (Oldest first)":
                filtered_pred = filtered_pred.sort_values("timestamp", ascending=True)
            elif sort_choice == "Risk Score (High to Low)":
                filtered_pred = filtered_pred.sort_values("risk_score", ascending=False)
            elif sort_choice == "Risk Score (Low to High)":
                filtered_pred = filtered_pred.sort_values("risk_score", ascending=True)
            elif sort_choice == "PPE Compliance (High to Low)":
                filtered_pred = filtered_pred.sort_values("ppe_compliance_pct", ascending=False)

            # Record metrics & count badge
            kpis_f = calculate_site_kpis(filtered_pred)
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin: 1rem 0;">
                    <span class="cs-chip active">Showing {len(filtered_pred)} of {len(pred_df)} assessments</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                render_metric_card("Matching Assessments", len(filtered_pred), variant="accent")
            with col_m2:
                render_metric_card("High / Critical Count", kpis_f["high_critical_count"], variant="high")
            with col_m3:
                render_metric_card("Avg Risk Score", f"{kpis_f['avg_risk_score']:.1f}", variant="default")
            with col_m4:
                render_metric_card("Avg PPE Compliance", f"{kpis_f['avg_ppe_compliance']:.1f}%", variant="medium")

            # Data Table
            st.dataframe(
                filtered_pred,
                use_container_width=True,
                hide_index=True,
            )

            # Detailed Row Inspector
            if not filtered_pred.empty:
                st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
                with st.expander("🔍 Inspect Specific Assessment Details", expanded=False):
                    row_labels = [
                        f"#{idx+1} — {row['activity_type']} ({row.get('predicted_risk_level', 'N/A')}) at {row.get('project/site', 'Unassigned')} [{str(row.get('timestamp', ''))[:19]}]"
                        for idx, row in filtered_pred.reset_index().iterrows()
                    ]
                    selected_idx = st.selectbox(
                        "Select an assessment record to inspect:",
                        range(len(row_labels)),
                        format_func=lambda i: row_labels[i],
                    )

                    chosen = filtered_pred.iloc[selected_idx]
                    c_risk = str(chosen.get("predicted_risk_level", "LOW")).upper()
                    c_score = float(chosen.get("risk_score", 0.0))
                    c_conf = float(chosen.get("model_confidence", 0.0))
                    c_conf_pct = c_conf * 100 if c_conf <= 1.0 else c_conf

                    st.markdown(
                        f"""
                        <div class="cs-card-flat" style="border-left: 4px solid {SAFETY_COLORS.get(c_risk, '#5266eb')};">
                            <div class="cs-card-header">
                                <div>
                                    <strong style="color: #ededf3; font-size: 1.1rem;">{chosen.get('activity_type', 'Activity')}</strong>
                                    <span class="cs-badge cs-badge-{c_risk.lower()}" style="margin-left: 0.6rem;">{c_risk} RISK</span>
                                </div>
                                <span style="font-family: monospace; font-size: 0.82rem; color: #c3c3cc;">{chosen.get('timestamp', '')}</span>
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1rem 0; font-size: 0.88rem; color: #c3c3cc;">
                                <div>🏢 Project / Site: <strong style="color: #ededf3;">{chosen.get('project/site', 'Unspecified')}</strong></div>
                                <div>📍 Location: <strong style="color: #ededf3;">{chosen.get('location_type', 'N/A')}</strong></div>
                                <div>🌤 Weather: <strong style="color: #ededf3;">{chosen.get('weather', 'N/A')}</strong></div>
                                <div>⏱ Shift: <strong style="color: #ededf3;">{chosen.get('shift', 'N/A')}</strong></div>
                                <div>👥 Crew Size: <strong style="color: #ededf3;">{chosen.get('crew_size', 'N/A')} workers</strong></div>
                                <div>🦺 PPE Compliance: <strong style="color: #ededf3;">{chosen.get('ppe_compliance_pct', 'N/A')}%</strong></div>
                                <div>⚠️ 30-Day Incidents: <strong style="color: #ededf3;">{chosen.get('previous_incidents_30d', '0')}</strong></div>
                                <div>◈ Risk Score: <strong style="color: #ededf3;">{c_score:.1f} / 100</strong></div>
                                <div>✦ Model Confidence: <strong style="color: #ededf3;">{c_conf_pct:.1f}%</strong></div>
                            </div>
                            <div style="background: #272735; padding: 0.75rem 1rem; border-radius: 8px; font-size: 0.88rem; margin-top: 0.5rem;">
                                <span style="color: #8e8e9c; font-size: 0.78rem; text-transform: uppercase; display: block; margin-bottom: 0.25rem;">Task / Hazard Description</span>
                                <span style="color: #ededf3;">{chosen.get('description', 'No specific description logged.') or 'No specific description logged.'}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Recommendations for selected row
                    act_name = str(chosen.get("activity_type", ""))
                    if act_name:
                        st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)
                        st.caption(f"PREVENTIVE ACTIONS FOR {act_name.upper()}")
                        rec_list = get_recommendations(act_name)
                        for r_item in rec_list:
                            st.markdown(f"✓ **{r_item}**")

            csv_data = filtered_pred.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Export Filtered Predictions (CSV)",
                data=csv_data,
                file_name="site_prediction_history.csv",
                mime="text/csv",
                type="secondary",
            )

    else:
        render_section_heading("Actual Jobsite Incident Records", "Company incident log from data/incident_records.csv")
        st.markdown(
            """
            <div class="cs-card-flat" style="border-left: 3px solid #3b82f6;">
                <div style="font-weight: 600; color: #ededf3; margin-bottom: 0.25rem;">
                    ℹ️ ACTUAL INCIDENTS vs RISK PREDICTIONS
                </div>
                <div style="color: #c3c3cc; font-size: 0.88rem; line-height: 1.5;">
                    Actual incident records represent logged safety events, injuries, and near-miss occurrences on site.
                    They are tracked separately from pre-task predictive evaluations and do not alter benchmark models.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        inc_records_df = load_actual_incident_records()

        if inc_records_df.empty:
            render_empty_state(
                title="No actual incident records logged yet",
                message="No workplace incidents or near-misses have been recorded in data/incident_records.csv.",
                icon="🛡",
            )
        else:
            st.dataframe(
                inc_records_df,
                use_container_width=True,
                hide_index=True,
            )

            csv_inc = inc_records_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Export Incident Records (CSV)",
                data=csv_inc,
                file_name="actual_incident_records.csv",
                mime="text/csv",
                type="secondary",
            )

        # Log new incident modal/form
        with st.expander("➕ Log New Actual Incident / Near-Miss Record"):
            with st.form("log_incident_form"):
                i_col1, i_col2 = st.columns(2)
                with i_col1:
                    inc_date = st.date_input("Date of Incident", value=datetime.now(timezone.utc).date())
                    inc_proj = st.text_input("Project / Site", value=st.session_state.selected_project)
                    inc_loc = st.text_input("Specific Location", placeholder="e.g. South Scaffold Tower — Level 4")
                    inc_type = st.selectbox("Incident Type", ["Near Miss", "First Aid", "Medical Treatment", "Lost Time Injury", "Property Damage", "Unsafe Condition"])
                with i_col2:
                    inc_sev = st.selectbox("Severity", ["Minor", "Moderate", "Severe", "Critical"])
                    inc_injury = st.text_input("Injury Description (if any)", placeholder="e.g. Minor hand laceration")
                    inc_status = st.selectbox("Status", ["Open", "Under Investigation", "Corrective Action Assigned", "Closed"])

                inc_desc = st.text_area("Incident Description & Sequence of Events")
                inc_cause = st.text_input("Apparent Root Cause")
                inc_action = st.text_input("Immediate Corrective Action Taken")

                submit_inc = st.form_submit_button("Log Incident Record", type="primary")

                if submit_inc and inc_desc.strip():
                    try:
                        new_inc_row = {
                            "incident_id": f"INC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                            "date": str(inc_date),
                            "project/site": inc_proj.strip() or "Unspecified",
                            "location": inc_loc.strip() or "Unspecified",
                            "incident_type": inc_type,
                            "severity": inc_sev,
                            "description": inc_desc.strip(),
                            "injury": inc_injury.strip() or "None",
                            "root_cause": inc_cause.strip() or "Under review",
                            "corrective_action": inc_action.strip() or "Pending",
                            "status": inc_status,
                        }
                        inc_df_to_save = pd.DataFrame([new_inc_row])
                        header = not INCIDENT_RECORDS_PATH.exists() or os.path.getsize(INCIDENT_RECORDS_PATH) == 0
                        inc_df_to_save.to_csv(INCIDENT_RECORDS_PATH, mode="a", header=header, index=False)
                        st.success(f"Incident record '{new_inc_row['incident_id']}' logged successfully.")
                        st.rerun()
                    except Exception as inc_err:
                        st.error(f"Error logging incident: {inc_err}")


# ==============================================================================
# 5. REPORTS
# ==============================================================================

elif current_page == "Reports":
    render_page_hero(
        title="Safety Reports & Briefings",
        subtitle="Generate structured periodic safety reports, compliance briefings, and audit exports.",
        tagline="REPORT CENTER",
    )

    report_tabs = st.tabs(["Weekly Safety Brief", "Monthly Summary", "Custom Date Range", "Safety Alerts", "Exports"])

    with report_tabs[0]:
        render_section_heading("Weekly Safety Briefing", "Operational safety overview for toolbox meetings")
        if data_loaded:
            incidents_df["date"] = pd.to_datetime(incidents_df["date"])
            latest_date = incidents_df["date"].max()
            start_date = latest_date - pd.Timedelta(days=6)
            weekly_df = incidents_df[
                (incidents_df["date"] >= start_date) & (incidents_df["date"] <= latest_date)
            ].copy()

            tot_w = len(weekly_df)
            hc_w = len(weekly_df[weekly_df["risk_level"].isin(["HIGH", "CRITICAL"])])
            crit_w = len(weekly_df[weekly_df["risk_level"] == "CRITICAL"])
            avg_ppe_w = weekly_df["ppe_compliance_pct"].mean() if not weekly_df.empty else 0.0

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_metric_card("Weekly Incidents", tot_w, f"{start_date.strftime('%d %b')} – {latest_date.strftime('%d %b')}", variant="accent")
            with c2:
                render_metric_card("High / Critical", hc_w, "Requires pre-task focus", variant="high")
            with c3:
                render_metric_card("Critical Risks", crit_w, "Immediate stop-work hazard", variant="critical")
            with c4:
                render_metric_card("Average PPE", f"{avg_ppe_w:.1f}%", "Observed weekly average", variant="medium")

            st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
            w_col1, w_col2 = st.columns(2)
            with w_col1:
                render_section_heading("Weekly Risk Distribution", "By risk severity")
                w_risk_counts = (
                    weekly_df["risk_level"]
                    .value_counts()
                    .reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0)
                    .reset_index()
                )
                w_risk_counts.columns = ["Risk Level", "Incidents"]
                fig_w = px.bar(
                    w_risk_counts,
                    x="Risk Level",
                    y="Incidents",
                    color="Risk Level",
                    color_discrete_map=RISK_COLOR_DISCRETE_MAP,
                )
                fig_w.update_layout(showlegend=False)
                style_mercury_chart(fig_w, height=260)
                st.plotly_chart(fig_w, use_container_width=True)

            with w_col2:
                render_section_heading("Priority Activities for Briefing", "Activities with most frequent hazards")
                top_w_act = weekly_df["activity_type"].value_counts().head(5).reset_index()
                top_w_act.columns = ["Activity", "Events"]
                st.dataframe(top_w_act, use_container_width=True, hide_index=True)

            st.markdown(
                """
                <div class="cs-card-flat">
                    <div class="cs-card-header">Weekly Safety Briefing Notes</div>
                    <p style="color: #ededf3; font-size: 0.9rem; line-height: 1.6; margin-bottom: 0.5rem;">
                        <strong>Focus Areas:</strong> Prioritize fall-arrest checks for elevated roof work and inspect all electrical
                        junction enclosures prior to morning shift startup. Verify that all subcontract crews complete daily
                        pre-task risk discussions.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with report_tabs[1]:
        render_section_heading("Monthly Safety Summary", "Aggregated monthly performance metrics")
        render_empty_state(
            title="Monthly report generator ready",
            message="Monthly reports consolidate site assessments, audit findings, and PPE compliance trends across all jobsite locations.",
            icon="▧",
        )

    with report_tabs[2]:
        render_section_heading("Custom Date Range Query", "Select custom start and end parameters")
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.date_input("Start Date", value=datetime.now(timezone.utc).date())
        with r_col2:
            st.date_input("End Date", value=datetime.now(timezone.utc).date())
        render_empty_state(
            title="Select parameters to generate custom report",
            message="Adjust date filters above to isolate specific construction phases or weather event intervals.",
            icon="📅",
        )

    with report_tabs[3]:
        render_section_heading("Active Safety Alerts", "Automated hazard escalation triggers")
        st.markdown(
            """
            <div class="cs-risk-banner high">
                ⚠️ <strong>HIGH WIND WARNING</strong> — Forecast indicates gusts above 25 knots. Cease crane lifting and secure all elevated scaffold materials immediately.
            </div>
            <div class="cs-risk-banner medium">
                ℹ️ <strong>PPE COMPLIANCE NOTICE</strong> — Excavation zones exhibited lower eye-protection compliance. Safety supervisors to conduct spot audits during afternoon shift.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with report_tabs[4]:
        render_section_heading("Formal Export Center", "Download formatted safety data packages")
        st.markdown(
            """
            <div class="cs-card">
                <div class="cs-card-header">Available Export Formats</div>
                <p style="color: #c3c3cc; font-size: 0.9rem;">
                    Safety Intelligence reports can be exported in structured formats for external audit, OSHA documentation, and joint safety committee review.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            if data_loaded:
                st.download_button(
                    label="⬇ Download Complete Incident Dataset (CSV)",
                    data=incidents_df.to_csv(index=False).encode("utf-8"),
                    file_name="incidents_export.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True,
                )
        with exp_col2:
            site_hist = load_prediction_history()
            if not site_hist.empty:
                st.download_button(
                    label="⬇ Download Site Prediction History (CSV)",
                    data=site_hist.to_csv(index=False).encode("utf-8"),
                    file_name="site_prediction_history.csv",
                    mime="text/csv",
                    type="secondary",
                    use_container_width=True,
                )
            else:
                st.button("📄 Export Formal PDF Report (Preparing)", disabled=True, use_container_width=True)


# ==============================================================================
# 6. SAFETY
# ==============================================================================

elif current_page == "Safety":
    render_page_hero(
        title="Field Safety Operations",
        subtitle="Operational hazard controls, preventive action databases, and safety checklist verification.",
        tagline="SAFETY OPERATIONS",
    )

    safety_tabs = st.tabs(["Preventive Actions", "Safety Checklists", "Toolbox Talks Library"])

    with safety_tabs[0]:
        render_section_heading("Task Preventive Controls", "Review and select safety controls tailored to activity and risk severity")
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            sel_act = st.selectbox(
                "Select Construction Activity",
                [
                    "Working at Height", "Electrical Work", "Excavation",
                    "Scaffolding", "Welding", "Lifting",
                    "Material Handling", "Vehicle Movement",
                    "Confined Space", "Housekeeping",
                ],
                key="safety_act_sel",
            )
        with s_col2:
            sel_risk = st.selectbox(
                "Activity Risk Severity",
                ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                index=2,
                key="safety_risk_sel",
            )

        render_section_heading(f"Recommended Controls for {sel_act}", "Required jobsite verifications")
        act_recs = get_recommendations(sel_act)
        for rec in act_recs:
            st.checkbox(rec, value=False, key=f"chk_rec_{rec[:20]}")

        risk_escalation_actions = {
            "LOW": ["Follow standard jobsite PPE policy.", "Verify routine housekeeping in task zone."],
            "MEDIUM": ["Conduct documented pre-task safety discussion.", "Inspect tool condition and electrical cables."],
            "HIGH": ["Verify task-specific Job Hazard Analysis (JHA).", "Designate certified supervisor oversight.", "Establish physical perimeter barricades."],
            "CRITICAL": ["Obtain written high-risk permit before work start.", "Verify zero energy state and secondary safety backups.", "Designate full-time safety observer."],
        }

        render_section_heading(f"Additional Escalation Controls ({sel_risk} Risk)", "Strict enforcement controls")
        for esc in risk_escalation_actions.get(sel_risk, []):
            st.checkbox(esc, value=False, key=f"chk_esc_{esc[:20]}")

    with safety_tabs[1]:
        render_section_heading("Jobsite Pre-Work Safety Checklists", "Standard digital audit templates")
        st.markdown(
            """
            <div class="cs-card-flat">
                <div class="cs-card-header">Daily General Jobsite Audit</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.checkbox("Emergency egress routes and fire extinguishers unobstructed", value=True)
        st.checkbox("All workers equipped with ANSI-approved hard hats, safety glasses, and steel-toe boots", value=True)
        st.checkbox("First aid kits fully stocked and emergency contact numbers posted", value=True)
        st.checkbox("Weather forecast verified against planned heavy lifting and exterior operations", value=False)
        st.checkbox("Hot work permits issued and 30-minute fire watch assigned for welding tasks", value=False)

    with safety_tabs[2]:
        render_section_heading("Toolbox Talks Briefing Repository", "Standard 5-minute pre-shift briefing templates")
        for act in ["Working at Height", "Electrical Work", "Scaffolding", "Excavation", "Lifting"]:
            with st.expander(f"🗣 {act} — Toolbox Talk Topics", expanded=(act == "Working at Height")):
                topics = get_toolbox_topics(act)
                for t in topics:
                    st.markdown(f"• **{t}**")


# ==============================================================================
# 7. AI SAFETY
# ==============================================================================

elif current_page == "AI Safety":
    render_page_hero(
        title="AI Safety Intelligence",
        subtitle="Generative safety analysis, automated hazard explanations, and AI-driven toolbox briefings (Preview).",
        tagline="AI SAFETY CO-PILOT",
    )

    st.markdown(
        """
        <div class="cs-card">
            <div class="cs-card-header">
                <span>✦ AI Safety Co-Pilot Architecture</span>
                <span class="cs-badge cs-badge-medium">Phase 3 Preview</span>
            </div>
            <p style="color: #c3c3cc; font-size: 0.92rem; line-height: 1.6;">
                The AI Safety Intelligence module will integrate deep contextual hazard models with project documents
                to deliver real-time risk explanations, automated weekly executive briefs, and instant regulatory guidance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ai_tab1, ai_tab2, ai_tab3, ai_tab4 = st.tabs([
        "AI Safety Analyst", "AI Risk Explanation", "AI Weekly Safety Brief", "AI Recommendations"
    ])

    with ai_tab1:
        render_section_heading("AI Safety Analyst Query Interface", "Interactive natural-language hazard assistant")
        sample_prompt = st.selectbox(
            "Select an AI Safety Query Template:",
            [
                "Analyze safety risks for multi-crane tandem lift near public roadway",
                "Generate custom toolbox briefing for steel erection in freezing rain",
                "Explain correlation between night shifts and PPE compliance drops",
                "Review OSHA Subpart M fall protection compliance requirements for steep pitch roofs",
            ],
        )
        query_input = st.text_input("Or enter a custom jobsite hazard query:", value=sample_prompt)
        ai_run_btn = st.button("✦ Generate AI Safety Analysis (Preview)", type="primary")

        if ai_run_btn:
            st.markdown(
                f"""
                <div class="cs-card-flat" style="border-left: 3px solid #5266eb;">
                    <div class="cs-card-header">
                        <span>✦ AI Safety Analyst Response (Preview Mode)</span>
                        <span style="font-family: monospace; font-size: 0.78rem; color: #8e8e9c;">Model: Safety-GPT / Engine v2.4</span>
                    </div>
                    <p style="color: #ededf3; font-size: 0.92rem; line-height: 1.6;">
                        <strong>Hazard Synthesis:</strong> For <em>"{query_input}"</em>, historical benchmark data highlights high severity
                        vulnerabilities related to load-line interference, ground bearing capacity under outriggers, and pedestrian barrier failure.
                    </p>
                    <p style="color: #c3c3cc; font-size: 0.88rem; line-height: 1.6;">
                        <strong>Recommended Controls:</strong> (1) Implement engineered lift plan verified by PE; (2) Enforce 1.5x load radius exclusion zone with physical barricades;
                        (3) Appoint dedicated single-channel radio rigger; (4) Pre-check ground compaction with geotechnical logs.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with ai_tab2:
        render_section_heading("AI Multi-Modal Risk Explanation", "Model explainability and causality engine")
        render_empty_state(
            title="AI Explanation Engine Preview",
            message="Translates machine learning feature importance into plain-language safety engineering recommendations for superintendents.",
            icon="✦",
        )

    with ai_tab3:
        render_section_heading("AI Automated Weekly Safety Brief", "One-click safety briefing generation")
        render_empty_state(
            title="Automated Briefing Generator Preview",
            message="Synthesizes weekly incidents and planned tasks into concise bullet points for morning standups.",
            icon="🗣",
        )

    with ai_tab4:
        render_section_heading("AI Preventive Recommendation Engine", "Dynamic rule & pattern discovery")
        render_empty_state(
            title="Dynamic Recommendations Preview",
            message="Learns jobsite-specific hazard patterns to recommend preventive adjustments prior to shift start.",
            icon="🛡",
        )


# ==============================================================================
# 8. PROJECTS
# ==============================================================================

elif current_page == "Projects":
    render_page_hero(
        title="Project Safety Management",
        subtitle="Manage multi-site construction portfolios, assign active site contexts, and track safety metrics.",
        tagline="PORTFOLIO INTELLIGENCE",
    )

    projects_df = load_projects_data()
    site_history_df = load_prediction_history()

    render_section_heading("Active Site Context", "Select the active construction project for site assessments")

    default_projects = [
        {"project_id": "PRJ-001", "project_name": "Metro Tower Expansion — Phase 2", "site/location": "Downtown Metro Hub", "start_date": "2026-01-15", "status": "Active"},
        {"project_id": "PRJ-002", "project_name": "Central Hospital New Wing", "site/location": "North Medical District", "start_date": "2026-02-01", "status": "Active"},
        {"project_id": "PRJ-003", "project_name": "Harbor Bridge Rehabilitation", "site/location": "South Port Pier 4", "start_date": "2025-11-10", "status": "Active"},
    ]

    all_project_names = [p["project_name"] for p in default_projects]
    if not projects_df.empty and "project_name" in projects_df.columns:
        for p_name in projects_df["project_name"].dropna().unique():
            if p_name not in all_project_names:
                all_project_names.append(p_name)

    cur_idx = (
        all_project_names.index(st.session_state.selected_project)
        if st.session_state.selected_project in all_project_names
        else 0
    )

    sel_p = st.selectbox(
        "Active Project Context:",
        all_project_names,
        index=cur_idx,
        key="project_switcher_select",
    )

    if sel_p != st.session_state.selected_project:
        st.session_state.selected_project = sel_p
        st.rerun()

    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    render_section_heading("Project Portfolio Overview", "Registered active jobsites")

    for proj in default_projects:
        p_name = proj["project_name"]
        p_site = proj["site/location"]
        p_stat = proj["status"]
        p_date = proj["start_date"]

        pred_count = 0
        if not site_history_df.empty and "project/site" in site_history_df.columns:
            pred_count = len(site_history_df[site_history_df["project/site"].str.contains(p_name, case=False, na=False)])

        is_active_site = (p_name == st.session_state.selected_project)
        badge_style = "border-color: #5266eb; background: rgba(82,102,235,0.08);" if is_active_site else ""

        st.markdown(
            f"""
            <div class="cs-card" style="{badge_style}">
                <div class="cs-card-header">
                    <div>
                        <strong style="color: #ededf3; font-size: 1.05rem;">{p_name}</strong>
                        {'<span class="cs-badge cs-badge-low" style="margin-left: 0.6rem;">CURRENT ACTIVE SITE</span>' if is_active_site else ''}
                    </div>
                    <span class="cs-badge cs-badge-medium">{p_stat}</span>
                </div>
                <div style="display: flex; gap: 2rem; color: #c3c3cc; font-size: 0.85rem; margin-top: 0.5rem;">
                    <div>📍 Location: <strong style="color: #ededf3;">{p_site}</strong></div>
                    <div>🗓 Started: <strong style="color: #ededf3;">{p_date}</strong></div>
                    <div>◈ Recorded Assessments: <strong style="color: #ededf3;">{pred_count}</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("➕ Register New Project / Jobsite"):
        with st.form("new_project_form"):
            new_p_name = st.text_input("Project Name", placeholder="e.g. Westside Logistics Park — Building B")
            new_p_loc = st.text_input("Site / Location", placeholder="e.g. West Industrial Zone, Plot 14")
            new_p_date = st.date_input("Start Date", value=datetime.now(timezone.utc).date())
            new_p_status = st.selectbox("Status", ["Active", "Planning", "Completed"])
            submit_proj = st.form_submit_button("Register Project", type="primary")

            if submit_proj and new_p_name.strip():
                try:
                    new_row = {
                        "project_id": f"PRJ-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        "project_name": new_p_name.strip(),
                        "site/location": new_p_loc.strip() or "Unspecified",
                        "start_date": str(new_p_date),
                        "status": new_p_status,
                    }
                    df_to_save = pd.DataFrame([new_row])
                    header = not PROJECTS_PATH.exists() or os.path.getsize(PROJECTS_PATH) == 0
                    df_to_save.to_csv(PROJECTS_PATH, mode="a", header=header, index=False)
                    st.success(f"Project '{new_p_name}' successfully registered.")
                    st.session_state.selected_project = new_p_name
                    st.rerun()
                except Exception as p_err:
                    st.error(f"Error registering project: {p_err}")


# ==============================================================================
# GLOBAL FOOTER
# ==============================================================================

render_footer()
