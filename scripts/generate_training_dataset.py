"""
generate_training_dataset.py

Generates a high-quality expanded training dataset (~2,000 records) for the
Construction Safety Risk Predictor, stored in data/training_incidents_expanded.csv.

CRITICAL CONSTRAINTS & DATA LEAKAGE PREVENTION:
1. data/incidents.csv (500 benchmark records) is NEVER modified.
2. The 100 held-out test records (stratified test_size=0.20, random_state=42)
   are strictly excluded from the candidate training data.
3. The candidate training dataset combines the 400 original training partition
   records + 1,600 new realistic synthetic records (total = 2,000 training records).
4. All domain relationships (activity-location mappings, PPE impact, weather
   penalties, crew size dynamics, realistic hazard descriptions) are strictly preserved.
"""

from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Project root paths
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
INCIDENTS_CSV = os.path.join(PROJECT_ROOT, "data", "incidents.csv")
EXPANDED_TRAIN_CSV = os.path.join(PROJECT_ROOT, "data", "training_incidents_expanded.csv")

# Ensure project root is in sys.path for importing src modules if needed
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    TEXT_FEATURE,
    VALID_CATEGORIES,
)

# Reference definitions
ACTIVITY_TYPES = [
    "Working at Height",
    "Lifting",
    "Scaffolding",
    "Excavation",
    "Electrical Work",
    "Material Handling",
    "Welding",
    "Confined Space",
    "Vehicle Movement",
    "Housekeeping",
]

ACTIVITY_BASE_WEIGHT: dict[str, float] = {
    "Working at Height": 28.0,
    "Confined Space": 30.0,
    "Excavation": 24.0,
    "Electrical Work": 22.0,
    "Scaffolding": 20.0,
    "Lifting": 18.0,
    "Welding": 16.0,
    "Vehicle Movement": 14.0,
    "Material Handling": 10.0,
    "Housekeeping": 4.0,
}

LOCATION_TYPES = [
    "Building Site",
    "Roof",
    "Scaffolding Area",
    "Excavation Area",
    "Warehouse",
    "Loading Area",
    "Electrical Room",
    "Road/Access Area",
]

LOCATION_WEIGHT: dict[str, float] = {
    "Roof": 12.0,
    "Excavation Area": 11.0,
    "Scaffolding Area": 9.0,
    "Electrical Room": 9.0,
    "Road/Access Area": 7.0,
    "Loading Area": 6.0,
    "Building Site": 5.0,
    "Warehouse": 3.0,
}

WEATHER_OPTIONS = ["Clear", "Rain", "Adverse", "Windy"]
SHIFT_OPTIONS = ["Day", "Night"]

ACTIVITY_LOCATION_MAP: dict[str, list[str]] = {
    "Working at Height": ["Roof", "Building Site", "Scaffolding Area"],
    "Lifting": ["Loading Area", "Building Site", "Warehouse"],
    "Scaffolding": ["Scaffolding Area", "Building Site"],
    "Excavation": ["Excavation Area", "Road/Access Area"],
    "Electrical Work": ["Electrical Room", "Building Site"],
    "Material Handling": ["Warehouse", "Loading Area"],
    "Welding": ["Building Site", "Warehouse"],
    "Confined Space": ["Excavation Area", "Electrical Room"],
    "Vehicle Movement": ["Road/Access Area", "Loading Area"],
    "Housekeeping": ["Warehouse", "Building Site"],
}

DESCRIPTION_TEMPLATES: dict[str, list[str]] = {
    "Working at Height": [
        "Unprotected edge observed during elevated work.",
        "Fall arrest harness not properly anchored.",
        "Worker observed near roof edge without guardrail.",
        "Ladder used beyond recommended height without stabilization.",
        "Work on leading edge without adequate perimeter protection.",
        "Static line lifeline inspection overdue during roofing installation.",
        "Improper harness inspection before elevated decking task.",
        "Unsecured tools at elevated height posing drop hazard.",
    ],
    "Lifting": [
        "Improper lifting zone setup observed.",
        "Load exceeded rated capacity of lifting equipment.",
        "Exclusion zone around lift not maintained.",
        "Rigging inspection not completed before lift.",
        "Crane outrigger placed on uncompacted soil.",
        "Tandem lift performed without approved lift plan.",
        "Sling showing visible wear used during heavy assembly hoist.",
        "Tag line omitted during crane load swinging operation.",
    ],
    "Scaffolding": [
        "Scaffold access partially obstructed.",
        "Missing guardrail on scaffold platform.",
        "Scaffold base observed on uneven, unstable ground.",
        "Incomplete scaffold tagging system observed.",
        "Missing toe-boards on working platform exceeding six feet.",
        "Scaffold tie-backs removed prematurely during cladding.",
        "Overloaded scaffold bay holding excess brick pallets.",
        "Damaged ladder rung on internal scaffold access tower.",
    ],
    "Excavation": [
        "Inadequate barricading around excavation area.",
        "Excavation walls showing signs of instability.",
        "No safe access/egress ladder provided in trench.",
        "Spoil pile placed too close to excavation edge.",
        "Trench box not extended above trench crest in loose soil.",
        "Water accumulation undermining trench base after rain.",
        "Underground utility markers ignored near mechanical digging.",
        "Excavation depth exceeded benching ratio requirements.",
    ],
    "Electrical Work": [
        "Damaged electrical cable observed.",
        "Live electrical panel left unlocked and unattended.",
        "Improper lockout/tagout procedure observed.",
        "Exposed wiring near a wet work area.",
        "Uninsulated temporary power distribution board in active zone.",
        "Absence of GFCI protection on portable power tools.",
        "Arc flash PPE missing during high-voltage switchgear testing.",
        "Overloaded circuit breaker showing signs of overheating.",
    ],
    "Material Handling": [
        "Manual handling of heavy load without mechanical aid.",
        "Stacked materials observed in an unstable configuration.",
        "Obstructed walkway due to stored materials.",
        "Improper storage of materials near a walkway.",
        "Pallet racking upright bent by handling equipment.",
        "Excessive repetitive lifting without job rotation.",
        "Unsecured pipes stored on slope without chocks.",
        "Heavy drum handled without designated drum gripper.",
    ],
    "Welding": [
        "Welding conducted near flammable materials.",
        "Inadequate ventilation observed during welding task.",
        "Fire watch not posted during hot work.",
        "Welding screen missing in shared work area.",
        "Damaged gas hose connected to oxy-fuel cutting cylinder.",
        "Combustible solvent containers stored within hot work radius.",
        "Inadequate fume extraction during galvanized steel welding.",
        "Welding cables crossing wet floor without insulation mats.",
    ],
    "Confined Space": [
        "Confined space entry without atmospheric testing.",
        "No standby attendant present during confined space work.",
        "Ventilation equipment not in place for confined space entry.",
        "Emergency retrieval system not set up before entry.",
        "Continuous gas monitor calibrated incorrectly before entry.",
        "Permit-required confined space entry log not signed.",
        "Restricted egress portal partially blocked by ventilation ducting.",
        "Communication failure between entrant and safety attendant.",
    ],
    "Vehicle Movement": [
        "Vehicle reversing without a spotter present.",
        "Pedestrian and vehicle routes not segregated.",
        "Vehicle observed exceeding site speed limit.",
        "Blind spot hazard noted at site access road.",
        "Reversing alarm inoperative on earthmoving equipment.",
        "Unattended vehicle left idling on active haul road.",
        "Forklift mast elevated while traveling in shared zone.",
        "Missing high-visibility delineator posts along haul ramp.",
    ],
    "Housekeeping": [
        "Obstruction observed blocking emergency access route.",
        "Debris accumulation observed in walkway.",
        "Spilled material not cleaned up promptly.",
        "Tools left in walking path after task completion.",
        "Scrap timber with protruding nails left on ground.",
        "Emergency eyewash station blocked by storage boxes.",
        "Liquid chemical spill without containment barrier.",
        "Inadequate lighting along primary pedestrian egress corridor.",
    ],
}

SEVERITY_OPTIONS = ["Near Miss", "Minor", "Moderate", "Major"]


def bucket_risk(score: float) -> str:
    """Convert numeric risk score to risk_level bucket."""
    if score < 28.0:
        return "LOW"
    elif score < 48.0:
        return "MEDIUM"
    elif score < 65.0:
        return "HIGH"
    else:
        return "CRITICAL"


def severity_from_risk(risk_level: str, rng: np.random.Generator) -> str:
    """Sample severity conditioned on risk level."""
    weights = {
        "LOW": [0.55, 0.30, 0.12, 0.03],
        "MEDIUM": [0.35, 0.35, 0.22, 0.08],
        "HIGH": [0.20, 0.30, 0.32, 0.18],
        "CRITICAL": [0.10, 0.20, 0.35, 0.35],
    }
    return str(rng.choice(SEVERITY_OPTIONS, p=weights[risk_level]))


def generate_synthetic_records(n: int = 1600, seed: int = 2026, start_id: int = 501) -> pd.DataFrame:
    """
    Generate n new realistic construction safety records using fixed random seed.
    Preserves all physical and operational correlations.
    """
    rng = np.random.default_rng(seed=seed)
    rows: list[dict[str, Any]] = []
    start_date = datetime(2024, 1, 1)

    for i in range(n):
        incident_idx = start_id + i
        activity = str(rng.choice(ACTIVITY_TYPES))
        location = str(rng.choice(ACTIVITY_LOCATION_MAP[activity]))
        weather = str(rng.choice(WEATHER_OPTIONS, p=[0.55, 0.20, 0.15, 0.10]))
        shift = str(rng.choice(SHIFT_OPTIONS, p=[0.75, 0.25]))
        
        # PPE Compliance: realistic distribution (mean 78%, std 14%), clamped [30, 100]
        ppe_compliance = int(np.clip(rng.normal(78.5, 14.0), 30, 100))
        
        # Previous incidents: realistic Poisson-like distribution
        previous_incidents = int(rng.choice([0, 0, 0, 1, 1, 2, 3], p=[0.38, 0.22, 0.15, 0.11, 0.07, 0.04, 0.03]))
        
        # Crew size: realistic crew size (mean 6, std 3), clamped [1, 20]
        crew_size = int(np.clip(rng.normal(5.8, 2.9), 1, 20))
        
        # Description
        description = str(rng.choice(DESCRIPTION_TEMPLATES[activity]))
        
        # Domain Risk Score calculation with realistic physical relationships
        score = ACTIVITY_BASE_WEIGHT[activity]
        score += LOCATION_WEIGHT[location]
        
        # Weather penalty
        if weather == "Adverse":
            score += 16.0
        elif weather in ("Rain", "Windy"):
            score += 7.0
            
        # PPE compliance penalty
        if ppe_compliance < 70:
            score += 20.0
        elif ppe_compliance < 85:
            score += 10.0
            
        # Shift fatigue
        if shift == "Night":
            score += 3.0
            
        # Previous incidents penalty
        score += previous_incidents * 9.0
        
        # Crew coordination overhead
        score += max(0, crew_size - 8) * 1.5
        
        # Natural variance / irreducible real-world noise
        score += float(rng.normal(0, 6.8))
        score = float(np.clip(score, 0.0, 100.0))
        
        risk_level = bucket_risk(score)
        severity = severity_from_risk(risk_level, rng)
        
        # Incident date and time
        days_offset = int(rng.integers(0, 750))
        incident_date = start_date + timedelta(days=days_offset)
        hour = int(rng.integers(6, 19)) if shift == "Day" else int(rng.integers(19, 24))
        minute = int(rng.integers(0, 60))
        time_str = f"{hour:02d}:{minute:02d}"
        
        rows.append({
            "incident_id": f"INC-EXP-{incident_idx:04d}",
            "date": incident_date.strftime("%Y-%m-%d"),
            "time": time_str,
            "activity_type": activity,
            "location_type": location,
            "description": description,
            "severity": severity,
            "risk_score": round(score, 1),
            "risk_level": risk_level,
            "weather": weather,
            "shift": shift,
            "ppe_compliance_pct": ppe_compliance,
            "previous_incidents_30d": previous_incidents,
            "crew_size": crew_size,
        })

    df = pd.DataFrame(rows)
    return df


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Strictly validates dataset quality and returns validation results."""
    issues = []
    
    # 1. Null check
    null_counts = df.isnull().sum()
    if null_counts.any():
        issues.append(f"Missing values found: {null_counts[null_counts > 0].to_dict()}")
        
    # 2. Schema check
    expected_cols = [
        "incident_id", "date", "time", "activity_type", "location_type",
        "description", "severity", "risk_score", "risk_level",
        "weather", "shift", "ppe_compliance_pct", "previous_incidents_30d", "crew_size"
    ]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        
    # 3. Categorical value validity
    for col, valids in VALID_CATEGORIES.items():
        if col in df.columns:
            invalid_vals = set(df[col].unique()) - set(valids)
            if invalid_vals:
                issues.append(f"Invalid values in '{col}': {invalid_vals}")
                
    # 4. Numerical range checks
    if (df["ppe_compliance_pct"] < 0).any() or (df["ppe_compliance_pct"] > 100).any():
        issues.append("ppe_compliance_pct out of range [0, 100]")
    if (df["crew_size"] < 1).any() or (df["crew_size"] > 50).any():
        issues.append("crew_size out of realistic range [1, 50]")
    if (df["previous_incidents_30d"] < 0).any():
        issues.append("previous_incidents_30d has negative values")
    if (df["risk_score"] < 0).any() or (df["risk_score"] > 100).any():
        issues.append("risk_score out of range [0, 100]")
        
    # 5. Duplicate check across feature columns
    dup_mask = df.duplicated(subset=ALL_FEATURE_COLUMNS, keep="first")
    num_duplicates = int(dup_mask.sum())
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "num_rows": len(df),
        "num_duplicates": num_duplicates,
        "class_distribution": df["risk_level"].value_counts().to_dict(),
    }


def main():
    print("=" * 70)
    print("PHASE 5: EXPAND TRAINING DATASET (2,000 RECORDS)")
    print("=" * 70)
    
    # 1. Verify original dataset hash
    with open(INCIDENTS_CSV, "rb") as f:
        orig_hash_before = hashlib.sha256(f.read()).hexdigest()
    print(f"Original incidents.csv SHA256: {orig_hash_before}")
    
    # 2. Load original 500 records and split to isolate test set
    df_original = pd.read_csv(INCIDENTS_CSV)
    print(f"Loaded original benchmark records: {len(df_original)}")
    
    # Split into original 400 train and 100 test (held-out)
    train_orig_df, test_orig_df = train_test_split(
        df_original,
        test_size=0.20,
        stratify=df_original[TARGET_COLUMN],
        random_state=42,
    )
    print(f"Original split: Train={len(train_orig_df)}, Held-Out Test={len(test_orig_df)}")
    print(f"Held-out test class distribution:\n{test_orig_df['risk_level'].value_counts().to_string()}\n")
    
    # 3. Generate 1,600 new synthetic training records
    target_new_records = 1600
    print(f"Generating {target_new_records} new realistic synthetic training records...")
    df_new = generate_synthetic_records(n=target_new_records, seed=2026, start_id=501)
    
    # 4. Combine original 400 train records + 1,600 new records
    combined_train_df = pd.concat([train_orig_df, df_new], ignore_index=True)
    
    # Deduplicate if any exact feature matches exist
    initial_count = len(combined_train_df)
    combined_train_df = combined_train_df.drop_duplicates(subset=ALL_FEATURE_COLUMNS).reset_index(drop=True)
    dups_removed = initial_count - len(combined_train_df)
    
    # If duplicates were removed, generate top-up records to maintain exact target ~2,000
    if len(combined_train_df) < 2000:
        topup_needed = 2000 - len(combined_train_df)
        print(f"Generating {topup_needed} top-up records to maintain ~2,000 total...")
        df_topup = generate_synthetic_records(n=topup_needed + 10, seed=3042, start_id=2200)
        combined_train_df = pd.concat([combined_train_df, df_topup], ignore_index=True)
        combined_train_df = combined_train_df.drop_duplicates(subset=ALL_FEATURE_COLUMNS).iloc[:2000].reset_index(drop=True)
        
    print(f"Total expanded training records: {len(combined_train_df)} (Duplicates removed: {dups_removed})")
    
    # 5. Validate dataset
    val_report = validate_dataset(combined_train_df)
    print(f"\nValidation status: {'PASSED' if val_report['valid'] else 'FAILED'}")
    if val_report["issues"]:
        for issue in val_report["issues"]:
            print(f"  [ERROR] {issue}")
    print(f"Class distribution in expanded training set:\n{pd.Series(val_report['class_distribution']).to_string()}")
    
    # 6. Save expanded training dataset
    combined_train_df.to_csv(EXPANDED_TRAIN_CSV, index=False)
    print(f"\nSaved expanded training dataset to: {EXPANDED_TRAIN_CSV}")
    
    # 7. Check original incidents.csv hash after generation
    with open(INCIDENTS_CSV, "rb") as f:
        orig_hash_after = hashlib.sha256(f.read()).hexdigest()
    assert orig_hash_before == orig_hash_after, "CRITICAL ERROR: incidents.csv was modified!"
    print(f"Verified incidents.csv untouched: SHA256 matches: {orig_hash_after == orig_hash_before}")
    print("generate_training_dataset.py completed successfully.\n")


if __name__ == "__main__":
    main()
