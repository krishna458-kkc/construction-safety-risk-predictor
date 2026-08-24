"""CSV storage helpers for company-specific safety data.

These files are intentionally separate from the historical dataset used to
train and benchmark the risk model.
"""

import csv
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PREDICTION_HISTORY_PATH = DATA_DIR / "prediction_history.csv"
INCIDENT_RECORDS_PATH = DATA_DIR / "incident_records.csv"
PROJECTS_PATH = DATA_DIR / "projects.csv"

PREDICTION_HISTORY_COLUMNS = [
    "timestamp",
    "project/site",
    "activity_type",
    "location_type",
    "weather",
    "shift",
    "ppe_compliance_pct",
    "crew_size",
    "previous_incidents_30d",
    "description",
    "predicted_risk_level",
    "risk_score",
    "model_confidence",
]

INCIDENT_RECORD_COLUMNS = [
    "incident_id",
    "date",
    "project/site",
    "location",
    "incident_type",
    "severity",
    "description",
    "injury",
    "root_cause",
    "corrective_action",
    "status",
]

PROJECT_COLUMNS = [
    "project_id",
    "project_name",
    "site/location",
    "start_date",
    "status",
]

DEFAULT_PROJECTS = [
    {
        "project_id": "PRJ-001",
        "project_name": "Metro Tower Expansion — Phase 2",
        "site/location": "Downtown Metro Hub",
        "start_date": "2026-01-15",
        "status": "Active",
    },
    {
        "project_id": "PRJ-002",
        "project_name": "Central Hospital New Wing",
        "site/location": "North Medical District",
        "start_date": "2026-02-01",
        "status": "Active",
    },
    {
        "project_id": "PRJ-003",
        "project_name": "Harbor Bridge Rehabilitation",
        "site/location": "South Port Pier 4",
        "start_date": "2025-11-10",
        "status": "Active",
    },
]


def _ensure_csv(path: Path, columns: List[str]) -> None:
    """Create a CSV with its header when it has not yet been created."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists() or os.path.getsize(path) == 0:
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            csv.writer(csv_file).writerow(columns)


def ensure_company_data_stores() -> None:
    """Ensure all company-specific data files exist and are seeded with initial registry if needed."""
    _ensure_csv(PREDICTION_HISTORY_PATH, PREDICTION_HISTORY_COLUMNS)
    _ensure_csv(INCIDENT_RECORDS_PATH, INCIDENT_RECORD_COLUMNS)
    _ensure_csv(PROJECTS_PATH, PROJECT_COLUMNS)

    # Seed default projects if projects.csv contains 0 data rows
    try:
        if PROJECTS_PATH.exists():
            df_proj = pd.read_csv(PROJECTS_PATH)
            if df_proj.empty:
                seed_df = pd.DataFrame(DEFAULT_PROJECTS)
                seed_df.to_csv(PROJECTS_PATH, index=False)
    except Exception:
        pass


def load_projects_data() -> pd.DataFrame:
    """Load projects registry from data/projects.csv."""
    ensure_company_data_stores()
    if not PROJECTS_PATH.exists():
        return pd.DataFrame(DEFAULT_PROJECTS)
    try:
        df = pd.read_csv(PROJECTS_PATH)
        if df.empty:
            df = pd.DataFrame(DEFAULT_PROJECTS)
            df.to_csv(PROJECTS_PATH, index=False)
        return df
    except Exception:
        return pd.DataFrame(DEFAULT_PROJECTS)


def register_project(
    name: str,
    location: str,
    start_date: str,
    status: str = "Active",
) -> Tuple[bool, str]:
    """Register a new construction project in data/projects.csv."""
    clean_name = name.strip()
    clean_loc = location.strip() or "Unspecified"
    clean_date = start_date.strip() or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    clean_status = status.strip() or "Active"

    if not clean_name:
        return False, "Project name cannot be empty."

    df_projects = load_projects_data()
    if not df_projects.empty and "project_name" in df_projects.columns:
        existing_names = [str(n).strip().lower() for n in df_projects["project_name"].dropna()]
        if clean_name.lower() in existing_names:
            return False, f"A project with the name '{clean_name}' already exists."

    new_id = f"PRJ-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    new_row = {
        "project_id": new_id,
        "project_name": clean_name,
        "site/location": clean_loc,
        "start_date": clean_date,
        "status": clean_status,
    }

    updated_df = pd.concat([df_projects, pd.DataFrame([new_row])], ignore_index=True)
    updated_df.to_csv(PROJECTS_PATH, index=False)
    return True, "Project registered successfully."


def update_project(
    project_id: str,
    new_name: str,
    new_location: str,
    new_start_date: str,
    new_status: str,
) -> Tuple[bool, str]:
    """Update project details in data/projects.csv and cascade name updates to prediction & incident history."""
    clean_id = project_id.strip()
    clean_name = new_name.strip()
    clean_loc = new_location.strip() or "Unspecified"
    clean_date = new_start_date.strip() or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    clean_status = new_status.strip() or "Active"

    if not clean_name:
        return False, "Project name cannot be empty."

    df_projects = load_projects_data()
    if df_projects.empty or clean_id not in df_projects["project_id"].astype(str).values:
        return False, f"Project ID '{clean_id}' not found in records."

    # Check for duplicate name among OTHER projects
    other_projects = df_projects[df_projects["project_id"].astype(str) != clean_id]
    if not other_projects.empty and "project_name" in other_projects.columns:
        other_names = [str(n).strip().lower() for n in other_projects["project_name"].dropna()]
        if clean_name.lower() in other_names:
            return False, f"Another project with the name '{clean_name}' already exists."

    # Retrieve old project name
    current_row = df_projects[df_projects["project_id"].astype(str) == clean_id].iloc[0]
    old_name = str(current_row["project_name"]).strip()

    # Update projects dataframe
    df_projects.loc[df_projects["project_id"].astype(str) == clean_id, "project_name"] = clean_name
    df_projects.loc[df_projects["project_id"].astype(str) == clean_id, "site/location"] = clean_loc
    df_projects.loc[df_projects["project_id"].astype(str) == clean_id, "start_date"] = clean_date
    df_projects.loc[df_projects["project_id"].astype(str) == clean_id, "status"] = clean_status
    df_projects.to_csv(PROJECTS_PATH, index=False)

    # If name changed, cascade update to prediction_history.csv and incident_records.csv
    if old_name and clean_name != old_name:
        # Cascade to prediction_history.csv
        if PREDICTION_HISTORY_PATH.exists() and os.path.getsize(PREDICTION_HISTORY_PATH) > 0:
            try:
                df_preds = pd.read_csv(PREDICTION_HISTORY_PATH)
                if not df_preds.empty and "project/site" in df_preds.columns:
                    mask = df_preds["project/site"].astype(str).str.strip() == old_name
                    if mask.any():
                        df_preds.loc[mask, "project/site"] = clean_name
                        df_preds.to_csv(PREDICTION_HISTORY_PATH, index=False)
            except Exception:
                pass

        # Cascade to incident_records.csv
        if INCIDENT_RECORDS_PATH.exists() and os.path.getsize(INCIDENT_RECORDS_PATH) > 0:
            try:
                df_incs = pd.read_csv(INCIDENT_RECORDS_PATH)
                if not df_incs.empty and "project/site" in df_incs.columns:
                    mask_inc = df_incs["project/site"].astype(str).str.strip() == old_name
                    if mask_inc.any():
                        df_incs.loc[mask_inc, "project/site"] = clean_name
                        df_incs.to_csv(INCIDENT_RECORDS_PATH, index=False)
            except Exception:
                pass

    return True, "Project updated successfully."


def record_prediction(
    project_site: str,
    activity_data: Dict[str, Any],
    prediction_result: Dict[str, Any],
) -> None:
    """Append one successfully submitted prediction to company history."""
    _ensure_csv(PREDICTION_HISTORY_PATH, PREDICTION_HISTORY_COLUMNS)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project/site": project_site.strip() or "Unspecified",
        "activity_type": activity_data["activity_type"],
        "location_type": activity_data["location_type"],
        "weather": activity_data["weather"],
        "shift": activity_data["shift"],
        "ppe_compliance_pct": activity_data["ppe_compliance_pct"],
        "crew_size": activity_data["crew_size"],
        "previous_incidents_30d": activity_data["previous_incidents_30d"],
        "description": activity_data["description"],
        "predicted_risk_level": prediction_result["risk_level"],
        "risk_score": prediction_result["risk_score"],
        "model_confidence": prediction_result["confidence"],
    }

    with PREDICTION_HISTORY_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        csv.DictWriter(csv_file, fieldnames=PREDICTION_HISTORY_COLUMNS).writerow(row)
