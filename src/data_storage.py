"""CSV storage helpers for company-specific safety data.

These files are intentionally separate from the historical dataset used to
train and benchmark the risk model.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def _ensure_csv(path: Path, columns: list[str]) -> None:
    """Create a CSV with its header when it has not yet been created."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            csv.writer(csv_file).writerow(columns)


def ensure_company_data_stores() -> None:
    """Ensure all company-specific data files exist without touching benchmarks."""
    _ensure_csv(PREDICTION_HISTORY_PATH, PREDICTION_HISTORY_COLUMNS)
    _ensure_csv(INCIDENT_RECORDS_PATH, INCIDENT_RECORD_COLUMNS)
    _ensure_csv(PROJECTS_PATH, PROJECT_COLUMNS)


def record_prediction(
    project_site: str,
    activity_data: dict[str, Any],
    prediction_result: dict[str, Any],
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
