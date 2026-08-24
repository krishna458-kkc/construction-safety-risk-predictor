"""Analytical, alerting, trend detection, heatmap, and reporting engine for company-specific safety data."""

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ==============================================================================
# CENTRALIZED ALERT THRESHOLD CONFIGURATION
# ==============================================================================

DEFAULT_ALERT_THRESHOLDS = {
    "critical_count_threshold": 2,          # Alert if CRITICAL assessments in period >= 2
    "high_critical_pct_threshold": 40.0,    # Alert if HIGH+CRITICAL assessments >= 40.0% of cohort
    "risk_increase_pct_threshold": 15.0,    # Alert if avg risk score increased >= 15.0% vs prior period
    "ppe_compliance_min_threshold": 80.0,   # Alert if observed avg PPE compliance <= 80.0%
    "repeated_activity_threshold": 3,       # Alert if same activity has >= 3 HIGH or CRITICAL assessments
    "severe_incident_threshold": 1,         # Alert if actual incidents have >= 1 Severe/Critical injury or event
}


# ==============================================================================
# DATETIME & FILTERING HELPERS
# ==============================================================================

def safe_parse_datetime(series: pd.Series) -> pd.Series:
    """Safely convert a pandas series to datetime, converting timezone-aware timestamps to UTC."""
    if series.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed


def filter_by_date_range(
    df: pd.DataFrame,
    period: str = "All time",
    custom_start: Optional[date] = None,
    custom_end: Optional[date] = None,
    date_col: str = "timestamp",
) -> pd.DataFrame:
    """Filter dataframe by specified date period preset or custom date range.
    
    Supports: 'All time', 'Last 7 days', 'Last 30 days', 'This month', 'Previous month', 'Custom range'.
    """
    if df.empty or date_col not in df.columns:
        return df.copy()

    df_copy = df.copy()
    dt_series = safe_parse_datetime(df_copy[date_col])
    
    if dt_series.isna().all():
        return df_copy

    now_utc = datetime.now(timezone.utc)
    today_utc = now_utc.date()

    if period == "All time":
        return df_copy

    elif period == "Last 7 days":
        cutoff = now_utc - timedelta(days=7)
        mask = dt_series >= cutoff
        return df_copy[mask].copy()

    elif period == "Last 30 days":
        cutoff = now_utc - timedelta(days=30)
        mask = dt_series >= cutoff
        return df_copy[mask].copy()

    elif period == "This month":
        first_of_month = datetime(today_utc.year, today_utc.month, 1, tzinfo=timezone.utc)
        mask = dt_series >= first_of_month
        return df_copy[mask].copy()

    elif period == "Previous month":
        if today_utc.month == 1:
            prev_year = today_utc.year - 1
            prev_month = 12
        else:
            prev_year = today_utc.year
            prev_month = today_utc.month - 1
        
        first_of_prev = datetime(prev_year, prev_month, 1, tzinfo=timezone.utc)
        first_of_curr = datetime(today_utc.year, today_utc.month, 1, tzinfo=timezone.utc)
        mask = (dt_series >= first_of_prev) & (dt_series < first_of_curr)
        return df_copy[mask].copy()

    elif period == "Custom range":
        if custom_start is not None and custom_end is not None:
            start_dt = datetime.combine(custom_start, datetime.min.time()).replace(tzinfo=timezone.utc)
            end_dt = datetime.combine(custom_end, datetime.max.time()).replace(tzinfo=timezone.utc)
            mask = (dt_series >= start_dt) & (dt_series <= end_dt)
            return df_copy[mask].copy()
        return df_copy

    return df_copy


def get_preceding_period_df(
    df: pd.DataFrame,
    period: str = "Last 7 days",
    date_col: str = "timestamp",
) -> pd.DataFrame:
    """Get the equivalent preceding time window dataframe for period-over-period comparison."""
    if df.empty or date_col not in df.columns:
        return pd.DataFrame()

    dt_series = safe_parse_datetime(df[date_col])
    if dt_series.isna().all():
        return pd.DataFrame()

    now_utc = datetime.now(timezone.utc)
    today_utc = now_utc.date()

    if period == "Last 7 days":
        curr_start = now_utc - timedelta(days=7)
        prev_start = now_utc - timedelta(days=14)
        mask = (dt_series >= prev_start) & (dt_series < curr_start)
        return df[mask].copy()

    elif period == "Last 30 days":
        curr_start = now_utc - timedelta(days=30)
        prev_start = now_utc - timedelta(days=60)
        mask = (dt_series >= prev_start) & (dt_series < curr_start)
        return df[mask].copy()

    elif period == "This month":
        if today_utc.month == 1:
            prev_year = today_utc.year - 1
            prev_month = 12
        else:
            prev_year = today_utc.year
            prev_month = today_utc.month - 1
        first_of_prev = datetime(prev_year, prev_month, 1, tzinfo=timezone.utc)
        first_of_curr = datetime(today_utc.year, today_utc.month, 1, tzinfo=timezone.utc)
        mask = (dt_series >= first_of_prev) & (dt_series < first_of_curr)
        return df[mask].copy()

    return pd.DataFrame()


# ==============================================================================
# KPI CALCULATIONS
# ==============================================================================

def calculate_site_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate company site safety KPIs from prediction history.
    
    Returns structured metrics handling empty dataframes safely.
    """
    if df.empty:
        return {
            "total_assessments": 0,
            "high_risk_count": 0,
            "critical_risk_count": 0,
            "high_critical_count": 0,
            "high_critical_pct": 0.0,
            "avg_risk_score": 0.0,
            "avg_confidence": 0.0,
            "avg_ppe_compliance": 0.0,
            "most_assessed_activity": "N/A",
            "latest_assessment_date": "N/A",
            "risk_counts": {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0},
        }

    total = len(df)
    
    # Risk Level counts
    r_levels = df["predicted_risk_level"].dropna().str.upper() if "predicted_risk_level" in df.columns else pd.Series()
    low_cnt = int((r_levels == "LOW").sum())
    med_cnt = int((r_levels == "MEDIUM").sum())
    high_cnt = int((r_levels == "HIGH").sum())
    crit_cnt = int((r_levels == "CRITICAL").sum())
    high_crit_cnt = high_cnt + crit_cnt
    high_crit_pct = (high_crit_cnt / total * 100) if total > 0 else 0.0

    # Risk Score & Confidence
    avg_score = float(df["risk_score"].mean()) if "risk_score" in df.columns and not df["risk_score"].dropna().empty else 0.0
    
    if "model_confidence" in df.columns and not df["model_confidence"].dropna().empty:
        conf_mean = float(df["model_confidence"].mean())
        avg_conf = conf_mean * 100 if conf_mean <= 1.0 else conf_mean
    else:
        avg_conf = 0.0

    # PPE Compliance
    avg_ppe = float(df["ppe_compliance_pct"].mean()) if "ppe_compliance_pct" in df.columns and not df["ppe_compliance_pct"].dropna().empty else 0.0

    # Most assessed activity
    if "activity_type" in df.columns and not df["activity_type"].dropna().empty:
        act_mode = df["activity_type"].dropna().mode()
        most_assessed = str(act_mode[0]) if not act_mode.empty else "N/A"
    else:
        most_assessed = "N/A"

    # Latest date
    if "timestamp" in df.columns and not df["timestamp"].dropna().empty:
        parsed_dates = safe_parse_datetime(df["timestamp"]).dropna()
        if not parsed_dates.empty:
            latest_dt = parsed_dates.max()
            latest_date_str = latest_dt.strftime("%d %b %Y, %H:%M UTC")
        else:
            latest_date_str = "N/A"
    else:
        latest_date_str = "N/A"

    return {
        "total_assessments": total,
        "high_risk_count": high_cnt,
        "critical_risk_count": crit_cnt,
        "high_critical_count": high_crit_cnt,
        "high_critical_pct": high_crit_pct,
        "avg_risk_score": round(avg_score, 1),
        "avg_confidence": round(avg_conf, 1),
        "avg_ppe_compliance": round(avg_ppe, 1),
        "most_assessed_activity": most_assessed,
        "latest_assessment_date": latest_date_str,
        "risk_counts": {
            "LOW": low_cnt,
            "MEDIUM": med_cnt,
            "HIGH": high_cnt,
            "CRITICAL": crit_cnt,
        },
    }


def calculate_activity_risk_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate assessment volume, average risk score, and high/critical % per activity."""
    if df.empty or "activity_type" not in df.columns:
        return pd.DataFrame(columns=["Activity", "Assessments", "Avg Risk Score", "High / Critical", "High / Critical %", "Avg PPE %"])

    records = []
    for activity, grp in df.groupby("activity_type"):
        total_act = len(grp)
        avg_score = float(grp["risk_score"].mean()) if "risk_score" in grp.columns else 0.0
        r_levels = grp["predicted_risk_level"].dropna().str.upper() if "predicted_risk_level" in grp.columns else pd.Series()
        hc_count = int(r_levels.isin(["HIGH", "CRITICAL"]).sum())
        hc_pct = (hc_count / total_act * 100) if total_act > 0 else 0.0
        avg_ppe = float(grp["ppe_compliance_pct"].mean()) if "ppe_compliance_pct" in grp.columns else 0.0

        records.append({
            "Activity": activity,
            "Assessments": total_act,
            "Avg Risk Score": round(avg_score, 1),
            "High / Critical": hc_count,
            "High / Critical %": round(hc_pct, 1),
            "Avg PPE %": round(avg_ppe, 1),
        })

    result_df = pd.DataFrame(records)
    if not result_df.empty:
        result_df = result_df.sort_values(by=["Avg Risk Score", "Assessments"], ascending=[False, False])
    return result_df


def calculate_ppe_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate observed PPE relationships with risk scores and compliance over time."""
    if df.empty or "ppe_compliance_pct" not in df.columns:
        return {
            "avg_ppe_by_risk": pd.DataFrame(columns=["Risk Level", "Avg PPE %"]),
            "ppe_trend": pd.DataFrame(columns=["Date", "Avg PPE %"]),
        }

    risk_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    if "predicted_risk_level" in df.columns:
        ppe_by_risk = (
            df.groupby("predicted_risk_level")["ppe_compliance_pct"]
            .mean()
            .reindex(risk_order)
            .dropna()
            .reset_index()
        )
        ppe_by_risk.columns = ["Risk Level", "Avg PPE %"]
        ppe_by_risk["Avg PPE %"] = ppe_by_risk["Avg PPE %"].round(1)
    else:
        ppe_by_risk = pd.DataFrame(columns=["Risk Level", "Avg PPE %"])

    if "timestamp" in df.columns:
        dt_col = safe_parse_datetime(df["timestamp"])
        valid_mask = dt_col.notna()
        if valid_mask.any():
            trend_df = df[valid_mask].copy()
            trend_df["date"] = dt_col[valid_mask].dt.date
            ppe_trend = trend_df.groupby("date")["ppe_compliance_pct"].mean().reset_index()
            ppe_trend.columns = ["Date", "Avg PPE %"]
            ppe_trend["Avg PPE %"] = ppe_trend["Avg PPE %"].round(1)
            ppe_trend = ppe_trend.sort_values("Date")
        else:
            ppe_trend = pd.DataFrame(columns=["Date", "Avg PPE %"])
    else:
        ppe_trend = pd.DataFrame(columns=["Date", "Avg PPE %"])

    return {
        "avg_ppe_by_risk": ppe_by_risk,
        "ppe_trend": ppe_trend,
    }


def calculate_project_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comparison metrics across projects/sites."""
    if df.empty or "project/site" not in df.columns:
        return pd.DataFrame(columns=["Project / Site", "Assessments", "Avg Risk Score", "High / Critical %", "Avg PPE %"])

    records = []
    for proj, grp in df.groupby("project/site"):
        total_p = len(grp)
        avg_score = float(grp["risk_score"].mean()) if "risk_score" in grp.columns else 0.0
        r_levels = grp["predicted_risk_level"].dropna().str.upper() if "predicted_risk_level" in grp.columns else pd.Series()
        hc_count = int(r_levels.isin(["HIGH", "CRITICAL"]).sum())
        hc_pct = (hc_count / total_p * 100) if total_p > 0 else 0.0
        avg_ppe = float(grp["ppe_compliance_pct"].mean()) if "ppe_compliance_pct" in grp.columns else 0.0

        records.append({
            "Project / Site": str(proj) if str(proj).strip() else "Unassigned",
            "Assessments": total_p,
            "Avg Risk Score": round(avg_score, 1),
            "High / Critical %": round(hc_pct, 1),
            "Avg PPE %": round(avg_ppe, 1),
        })

    res_df = pd.DataFrame(records)
    if not res_df.empty:
        res_df = res_df.sort_values("Assessments", ascending=False)
    return res_df


# ==============================================================================
# FEATURE 1: SAFETY ALERTS DETECTION
# ==============================================================================

def detect_safety_alerts(
    df_pred: pd.DataFrame,
    df_inc: pd.DataFrame,
    period_label: str = "Selected Period",
    thresholds: Optional[Dict[str, Any]] = None,
    df_pred_prev: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """Detect safety alerts based on company predictions and logged incidents using configurable thresholds.
    
    Alert Categories:
    1. CRITICAL RISK THRESHOLD
    2. HIGH/CRITICAL RISK SPIKE
    3. RISK TREND INCREASE
    4. LOW PPE COMPLIANCE
    5. REPEATED HIGH-RISK ACTIVITY
    6. INCIDENT SEVERITY ALERT
    """
    cfg = dict(DEFAULT_ALERT_THRESHOLDS)
    if thresholds:
        cfg.update(thresholds)

    alerts: List[Dict[str, Any]] = []

    if df_pred.empty and df_inc.empty:
        return alerts

    # --- 1. CRITICAL RISK THRESHOLD ---
    if not df_pred.empty and "predicted_risk_level" in df_pred.columns:
        crit_count = int((df_pred["predicted_risk_level"].str.upper() == "CRITICAL").sum())
        thresh_crit = cfg["critical_count_threshold"]
        if crit_count >= thresh_crit:
            alerts.append({
                "category": "CRITICAL RISK THRESHOLD",
                "severity": "CRITICAL",
                "title": "Critical-Risk Assessments Exceeded Threshold",
                "reason": f"{crit_count} critical-risk evaluations recorded in {period_label}, exceeding threshold ({thresh_crit}).",
                "supporting_value": f"{crit_count} Critical Assessments",
                "threshold": f"Threshold: {thresh_crit}",
                "period": period_label,
                "affected_scope": "Site Operations",
                "recommended_action": "Impose mandatory stop-work verification and require superintendent permit sign-offs for affected zones.",
            })

    # --- 2. HIGH/CRITICAL RISK SPIKE ---
    if not df_pred.empty and len(df_pred) >= 3 and "predicted_risk_level" in df_pred.columns:
        hc_count = int(df_pred["predicted_risk_level"].str.upper().isin(["HIGH", "CRITICAL"]).sum())
        hc_pct = (hc_count / len(df_pred)) * 100
        thresh_hc = cfg["high_critical_pct_threshold"]
        if hc_pct >= thresh_hc:
            alerts.append({
                "category": "HIGH/CRITICAL RISK SPIKE",
                "severity": "HIGH",
                "title": "High / Critical Risk Proportion Spike",
                "reason": f"{hc_pct:.1f}% of evaluations in {period_label} are High or Critical risk, exceeding threshold ({thresh_hc:.1f}%).",
                "supporting_value": f"{hc_pct:.1f}% High/Critical Share",
                "threshold": f"Threshold: {thresh_hc:.1f}%",
                "period": period_label,
                "affected_scope": f"{hc_count} of {len(df_pred)} assessments",
                "recommended_action": "Increase dedicated field safety supervision and review all upcoming high-risk activity permits.",
            })

    # --- 3. RISK TREND INCREASE ---
    if df_pred_prev is not None and not df_pred_prev.empty and not df_pred.empty:
        if len(df_pred) >= 2 and len(df_pred_prev) >= 2 and "risk_score" in df_pred.columns and "risk_score" in df_pred_prev.columns:
            curr_score = float(df_pred["risk_score"].mean())
            prev_score = float(df_pred_prev["risk_score"].mean())
            if prev_score > 0:
                delta_pct = ((curr_score - prev_score) / prev_score) * 100
                thresh_inc = cfg["risk_increase_pct_threshold"]
                if delta_pct >= thresh_inc:
                    alerts.append({
                        "category": "RISK TREND INCREASE",
                        "severity": "HIGH",
                        "title": "Elevated Risk Score Trajectory",
                        "reason": f"Average risk score increased by +{delta_pct:.1f}% (from {prev_score:.1f} to {curr_score:.1f}) compared to previous period.",
                        "supporting_value": f"+{delta_pct:.1f}% Score Increase",
                        "threshold": f"Threshold: +{thresh_inc:.1f}%",
                        "period": period_label,
                        "affected_scope": "Site-wide risk trend",
                        "recommended_action": "Audit change in site work-phases and verify preventive controls on newly commenced activities.",
                    })

    # --- 4. LOW PPE COMPLIANCE ---
    if not df_pred.empty and "ppe_compliance_pct" in df_pred.columns:
        avg_ppe = float(df_pred["ppe_compliance_pct"].mean())
        thresh_ppe = cfg["ppe_compliance_min_threshold"]
        if avg_ppe <= thresh_ppe:
            alerts.append({
                "category": "LOW PPE COMPLIANCE",
                "severity": "MEDIUM",
                "title": "Observed PPE Compliance Below Standard",
                "reason": f"Average PPE compliance observed across evaluations in {period_label} is {avg_ppe:.1f}%, below target standard ({thresh_ppe:.1f}%).",
                "supporting_value": f"{avg_ppe:.1f}% PPE Compliance",
                "threshold": f"Standard: >={thresh_ppe:.1f}%",
                "period": period_label,
                "affected_scope": "Field Crew Compliance",
                "recommended_action": "Conduct immediate jobsite PPE audit and reinforce PPE mandatory requirements in next toolbox briefing.",
            })

    # --- 5. REPEATED HIGH-RISK ACTIVITY ---
    if not df_pred.empty and "activity_type" in df_pred.columns and "predicted_risk_level" in df_pred.columns:
        thresh_rep = cfg["repeated_activity_threshold"]
        hc_df = df_pred[df_pred["predicted_risk_level"].str.upper().isin(["HIGH", "CRITICAL"])]
        if not hc_df.empty:
            act_counts = hc_df["activity_type"].value_counts()
            for act, cnt in act_counts.items():
                if cnt >= thresh_rep:
                    alerts.append({
                        "category": "REPEATED HIGH-RISK ACTIVITY",
                        "severity": "HIGH",
                        "title": f"Recurring Severe Hazard: {act}",
                        "reason": f"'{act}' accumulated {cnt} High/Critical assessments in {period_label}, reaching threshold ({thresh_rep}).",
                        "supporting_value": f"{cnt} Severe Assessments",
                        "threshold": f"Threshold: {thresh_rep}",
                        "period": period_label,
                        "affected_scope": f"Activity: {act}",
                        "recommended_action": f"Deploy task-specific engineered controls and dedicated safety observer for all upcoming '{act}' operations.",
                    })

    # --- 6. INCIDENT SEVERITY ALERT ---
    if not df_inc.empty and "severity" in df_inc.columns:
        sev_count = int(df_inc["severity"].str.lower().isin(["severe", "critical", "lost time injury"]).sum())
        thresh_sev = cfg["severe_incident_threshold"]
        if sev_count >= thresh_sev:
            alerts.append({
                "category": "INCIDENT SEVERITY ALERT",
                "severity": "CRITICAL",
                "title": "Severe Workplace Incident Logged",
                "reason": f"{sev_count} severe/critical incident(s) logged in company records during {period_label}.",
                "supporting_value": f"{sev_count} Severe Incident(s)",
                "threshold": f"Threshold: {thresh_sev}",
                "period": period_label,
                "affected_scope": "Actual Safety Events",
                "recommended_action": "Convene Joint Incident Review Committee immediately and execute corrective action verification plan.",
            })

    # Sort alerts by severity (CRITICAL first, then HIGH, MEDIUM, LOW)
    sev_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    alerts.sort(key=lambda a: sev_rank.get(a["severity"], 99))
    return alerts


# ==============================================================================
# FEATURE 2: STATISTICAL TREND DETECTION
# ==============================================================================

def detect_risk_trends(
    df_current: pd.DataFrame,
    df_previous: pd.DataFrame,
    min_observations: int = 2,
) -> Dict[str, Any]:
    """Calculate statistical risk and PPE trends comparing current period to preceding period."""
    if df_current.empty or len(df_current) < min_observations:
        return {
            "status": "Insufficient data",
            "risk_trend": "Insufficient data",
            "risk_delta_pct": 0.0,
            "ppe_trend": "Insufficient data",
            "ppe_delta_pct": 0.0,
            "current_avg_risk": 0.0,
            "prev_avg_risk": 0.0,
            "current_avg_ppe": 0.0,
            "prev_avg_ppe": 0.0,
            "summary_text": "Not enough observations in the current period to establish a statistical risk trend.",
        }

    curr_avg_risk = float(df_current["risk_score"].mean()) if "risk_score" in df_current.columns else 0.0
    curr_avg_ppe = float(df_current["ppe_compliance_pct"].mean()) if "ppe_compliance_pct" in df_current.columns else 0.0

    if df_previous.empty or len(df_previous) < min_observations:
        return {
            "status": "Insufficient data for period comparison",
            "risk_trend": "Stable",
            "risk_delta_pct": 0.0,
            "ppe_trend": "Stable",
            "ppe_delta_pct": 0.0,
            "current_avg_risk": round(curr_avg_risk, 1),
            "prev_avg_risk": 0.0,
            "current_avg_ppe": round(curr_avg_ppe, 1),
            "prev_avg_ppe": 0.0,
            "summary_text": f"Current period average risk score is {curr_avg_risk:.1f}. Preceding comparative baseline has insufficient observations.",
        }

    prev_avg_risk = float(df_previous["risk_score"].mean()) if "risk_score" in df_previous.columns else 0.0
    prev_avg_ppe = float(df_previous["ppe_compliance_pct"].mean()) if "ppe_compliance_pct" in df_previous.columns else 0.0

    # Risk Delta
    if prev_avg_risk > 0:
        risk_delta = ((curr_avg_risk - prev_avg_risk) / prev_avg_risk) * 100
    else:
        risk_delta = 0.0

    # PPE Delta
    if prev_avg_ppe > 0:
        ppe_delta = ((curr_avg_ppe - prev_avg_ppe) / prev_avg_ppe) * 100
    else:
        ppe_delta = 0.0

    # Classify Risk Trend
    if risk_delta >= 5.0:
        risk_trend = "Increasing"
    elif risk_delta <= -5.0:
        risk_trend = "Decreasing"
    else:
        risk_trend = "Stable"

    # Classify PPE Trend
    if ppe_delta >= 3.0:
        ppe_trend = "Increasing"
    elif ppe_delta <= -3.0:
        ppe_trend = "Decreasing"
    else:
        ppe_trend = "Stable"

    risk_dir_str = f"increased by +{risk_delta:.1f}%" if risk_delta > 0 else (f"decreased by {risk_delta:.1f}%" if risk_delta < 0 else "remained stable")
    summary = f"Average risk score {risk_dir_str} (from {prev_avg_risk:.1f} to {curr_avg_risk:.1f}) compared with the preceding comparative window."

    return {
        "status": "Calculated",
        "risk_trend": risk_trend,
        "risk_delta_pct": round(risk_delta, 1),
        "ppe_trend": ppe_trend,
        "ppe_delta_pct": round(ppe_delta, 1),
        "current_avg_risk": round(curr_avg_risk, 1),
        "prev_avg_risk": round(prev_avg_risk, 1),
        "current_avg_ppe": round(curr_avg_ppe, 1),
        "prev_avg_ppe": round(prev_avg_ppe, 1),
        "summary_text": summary,
    }


# ==============================================================================
# FEATURE 3: RISK HEATMAP GENERATION
# ==============================================================================

def calculate_risk_heatmap_matrix(
    df: pd.DataFrame,
    row_col: str = "activity_type",
    col_col: str = "predicted_risk_level",
    val_mode: str = "count",
) -> pd.DataFrame:
    """Generate a clean cross-tabulation or average score matrix for Plotly Heatmaps."""
    if df.empty or row_col not in df.columns or col_col not in df.columns:
        return pd.DataFrame()

    if val_mode == "count":
        matrix = pd.crosstab(df[row_col], df[col_col])
        if col_col == "predicted_risk_level":
            matrix = matrix.reindex(columns=["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0)
        return matrix
    elif val_mode == "avg_score" and "risk_score" in df.columns:
        matrix = df.groupby([row_col, col_col])["risk_score"].mean().unstack(fill_value=0)
        if col_col == "predicted_risk_level":
            matrix = matrix.reindex(columns=["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0)
        return matrix.round(1)

    return pd.DataFrame()


# ==============================================================================
# FEATURES 4, 5, 6: EXECUTIVE PERIOD SAFETY REPORTS
# ==============================================================================

def generate_period_report(
    df_pred: pd.DataFrame,
    df_inc: pd.DataFrame,
    period_title: str = "Weekly Safety Brief",
    period_label: str = "Last 7 Days",
    df_pred_prev: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Generate a complete, structured executive company safety intelligence report."""
    kpis = calculate_site_kpis(df_pred)
    alerts = detect_safety_alerts(df_pred, df_inc, period_label, df_pred_prev=df_pred_prev)
    trends = detect_risk_trends(df_pred, df_pred_prev if df_pred_prev is not None else pd.DataFrame())
    act_metrics = calculate_activity_risk_metrics(df_pred)
    ppe_data = calculate_ppe_analysis(df_pred)

    # Executive narrative generation
    if df_pred.empty:
        exec_summary = f"No company safety evaluations were recorded for {period_label}. Baseline monitoring remained idle."
    else:
        hc_pct = kpis["high_critical_pct"]
        top_act = kpis["most_assessed_activity"]
        risk_adj = "elevated" if hc_pct >= 40 else ("moderate" if hc_pct >= 20 else "low")
        exec_summary = (
            f"During {period_label}, a total of {kpis['total_assessments']} site safety risk evaluations were recorded. "
            f"Overall risk exposure is assessed as {risk_adj} with {kpis['high_critical_count']} evaluations ({hc_pct:.1f}%) "
            f"meeting High or Critical risk thresholds. Primary task activity under evaluation was '{top_act}'. "
            f"Observed PPE compliance averaged {kpis['avg_ppe_compliance']:.1f}% across all evaluated work zones."
        )

    # Recommendations checklist
    rec_actions: List[str] = []
    if not act_metrics.empty:
        highest_risk_act = str(act_metrics.iloc[0]["Activity"])
        rec_actions.append(f"Focus pre-task safety briefings on top-risk activity: '{highest_risk_act}'.")
        for rec in get_recommendations(highest_risk_act)[:3]:
            rec_actions.append(f"{highest_risk_act}: {rec}")
    else:
        rec_actions.append("Conduct standard daily pre-task job hazard analyses across all active zones.")
        rec_actions.append("Verify mandatory PPE compliance prior to morning shift startup.")

    if kpis["critical_risk_count"] > 0:
        rec_actions.insert(0, f"Mandatory Stop-Work Verification: {kpis['critical_risk_count']} Critical risk evaluation(s) require supervisor sign-off.")

    return {
        "period_title": period_title,
        "period_label": period_label,
        "executive_summary": exec_summary,
        "kpis": kpis,
        "alerts": alerts,
        "trends": trends,
        "activity_metrics": act_metrics,
        "ppe_data": ppe_data,
        "recommended_actions": rec_actions,
        "total_incidents_logged": len(df_inc),
        "incidents_table": df_inc if not df_inc.empty else pd.DataFrame(),
    }
