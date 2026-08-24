"""Analytical and date-filtering engine for company-specific construction safety data."""

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def safe_parse_datetime(series: pd.Series) -> pd.Series:
    """Safely convert a pandas series to datetime, converting timezone-aware timestamps to UTC."""
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
    
    # If all timestamps failed to parse, return original
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

    # Avg PPE by Risk Level
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

    # PPE Trend over time
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
