"""
data_generator.py

Creates two SYNTHETIC datasets for the Construction Safety Risk Predictor MVP:

    1. data/incidents.csv            -> ~500 historical incident records (with a risk_level label)
    2. data/planned_activities.csv   -> ~20 upcoming planned activities (no label yet - these are
                                          what the app will let a user browse/predict on)

IMPORTANT: This data is entirely synthetic (randomly generated). It is NOT sourced from any real
company or real incident reports. It exists so we have something realistic to train and demo on.

RISK-LABELING METHODOLOGY (documented, as required by the project spec):
--------------------------------------------------------------------------
Each incident gets a numeric "risk_score" built from a simple, transparent weighted sum:

    risk_score = activity_weight
               + location_weight
               + weather_penalty            (if weather == "Adverse")
               + low_ppe_penalty            (if ppe_compliance_pct < 70)
               + previous_incidents_penalty (scales with previous_incidents_30d)
               + crew_size_penalty          (larger crews score slightly higher)
               + random_noise               (small random term, so labels aren't perfectly
                                              deterministic from the features - this mimics
                                              the fact that real-world outcomes have some
                                              irreducible randomness)

The resulting numeric score (roughly 0-100) is then bucketed into four risk levels:

    score < 28            -> LOW
    28 <= score < 48       -> MEDIUM
    48 <= score < 65       -> HIGH
    score >= 65            -> CRITICAL

This rule is intentionally simple and fully disclosed. It is NOT a claim about real construction
safety statistics - it's a reproducible way to generate a labeled dataset so we have something to
train a classifier on. The ML model's job (later, in train_model.py) is to see whether it can learn
to approximately recover this pattern from the input features alone, without being told the formula.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Fixed seed -> re-running this script always produces the exact same dataset.
# This matters for reproducibility: if you regenerate data, train, and demo again tomorrow,
# you get consistent results instead of a different random dataset each time.
RNG = np.random.default_rng(seed=42)

N_INCIDENTS = 500
N_PLANNED_ACTIVITIES = 20

# ---------------------------------------------------------------------------
# 1. Reference lists (activity types, locations, etc.)
# ---------------------------------------------------------------------------

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

# Each activity gets a base weight representing its inherent hazard level.
# These are illustrative, documented assumptions for the synthetic dataset -
# NOT a claim about real-world OSHA statistics.
ACTIVITY_BASE_WEIGHT = {
    "Working at Height": 28,
    "Confined Space": 30,
    "Excavation": 24,
    "Electrical Work": 22,
    "Scaffolding": 20,
    "Lifting": 18,
    "Welding": 16,
    "Vehicle Movement": 14,
    "Material Handling": 10,
    "Housekeeping": 4,
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

LOCATION_WEIGHT = {
    "Roof": 12,
    "Excavation Area": 11,
    "Scaffolding Area": 9,
    "Electrical Room": 9,
    "Road/Access Area": 7,
    "Loading Area": 6,
    "Building Site": 5,
    "Warehouse": 3,
}

WEATHER_OPTIONS = ["Clear", "Rain", "Adverse", "Windy"]
SHIFT_OPTIONS = ["Day", "Night"]

# Which locations make sense for which activity (keeps the synthetic data plausible
# instead of e.g. "Excavation" happening in a "Warehouse").
ACTIVITY_LOCATION_MAP = {
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

# Realistic-sounding hazard descriptions per activity. Multiple templates per activity
# give the text some variety so TF-IDF later has real signal to work with.
DESCRIPTION_TEMPLATES = {
    "Working at Height": [
        "Unprotected edge observed during elevated work.",
        "Fall arrest harness not properly anchored.",
        "Worker observed near roof edge without guardrail.",
        "Ladder used beyond recommended height without stabilization.",
    ],
    "Lifting": [
        "Improper lifting zone setup observed.",
        "Load exceeded rated capacity of lifting equipment.",
        "Exclusion zone around lift not maintained.",
        "Rigging inspection not completed before lift.",
    ],
    "Scaffolding": [
        "Scaffold access partially obstructed.",
        "Missing guardrail on scaffold platform.",
        "Scaffold base observed on uneven, unstable ground.",
        "Incomplete scaffold tagging system observed.",
    ],
    "Excavation": [
        "Inadequate barricading around excavation area.",
        "Excavation walls showing signs of instability.",
        "No safe access/egress ladder provided in trench.",
        "Spoil pile placed too close to excavation edge.",
    ],
    "Electrical Work": [
        "Damaged electrical cable observed.",
        "Live electrical panel left unlocked and unattended.",
        "Improper lockout/tagout procedure observed.",
        "Exposed wiring near a wet work area.",
    ],
    "Material Handling": [
        "Manual handling of heavy load without mechanical aid.",
        "Stacked materials observed in an unstable configuration.",
        "Obstructed walkway due to stored materials.",
        "Improper storage of materials near a walkway.",
    ],
    "Welding": [
        "Welding conducted near flammable materials.",
        "Inadequate ventilation observed during welding task.",
        "Fire watch not posted during hot work.",
        "Welding screen missing in shared work area.",
    ],
    "Confined Space": [
        "Confined space entry without atmospheric testing.",
        "No standby attendant present during confined space work.",
        "Ventilation equipment not in place for confined space entry.",
        "Emergency retrieval system not set up before entry.",
    ],
    "Vehicle Movement": [
        "Vehicle reversing without a spotter present.",
        "Pedestrian and vehicle routes not segregated.",
        "Vehicle observed exceeding site speed limit.",
        "Blind spot hazard noted at site access road.",
    ],
    "Housekeeping": [
        "Obstruction observed blocking emergency access route.",
        "Debris accumulation observed in walkway.",
        "Spilled material not cleaned up promptly.",
        "Tools left in walking path after task completion.",
    ],
}

SEVERITY_OPTIONS = ["Near Miss", "Minor", "Moderate", "Major"]


def bucket_risk(score: float) -> str:
    """Convert a numeric risk_score into a risk_level bucket. See module docstring for thresholds."""
    if score < 28:
        return "LOW"
    elif score < 48:
        return "MEDIUM"
    elif score < 65:
        return "HIGH"
    else:
        return "CRITICAL"


def severity_from_risk(risk_level: str) -> str:
    """Pick a severity that's plausible given the risk level (higher risk -> more likely to be
    a more serious severity, but not guaranteed - a HIGH risk activity can still be a near miss)."""
    weights = {
        "LOW": [0.55, 0.30, 0.12, 0.03],
        "MEDIUM": [0.35, 0.35, 0.22, 0.08],
        "HIGH": [0.20, 0.30, 0.32, 0.18],
        "CRITICAL": [0.10, 0.20, 0.35, 0.35],
    }
    return RNG.choice(SEVERITY_OPTIONS, p=weights[risk_level])


def generate_incidents(n=N_INCIDENTS) -> pd.DataFrame:
    rows = []
    start_date = datetime(2024, 1, 1)

    for i in range(1, n + 1):
        activity = RNG.choice(ACTIVITY_TYPES)
        location = RNG.choice(ACTIVITY_LOCATION_MAP[activity])
        weather = RNG.choice(WEATHER_OPTIONS, p=[0.55, 0.20, 0.15, 0.10])
        shift = RNG.choice(SHIFT_OPTIONS, p=[0.75, 0.25])
        ppe_compliance = int(np.clip(RNG.normal(80, 15), 30, 100))
        previous_incidents_30d = int(RNG.choice([0, 0, 0, 1, 1, 2, 3], p=[0.35, 0.2, 0.15, 0.12, 0.08, 0.06, 0.04]))
        crew_size = int(np.clip(RNG.normal(6, 3), 1, 20))
        description = RNG.choice(DESCRIPTION_TEMPLATES[activity])

        # --- risk score formula (see module docstring) ---
        score = ACTIVITY_BASE_WEIGHT[activity]
        score += LOCATION_WEIGHT[location]
        score += 16 if weather == "Adverse" else (7 if weather in ("Rain", "Windy") else 0)
        score += 20 if ppe_compliance < 70 else (10 if ppe_compliance < 85 else 0)
        score += previous_incidents_30d * 9
        score += max(0, crew_size - 8) * 1.5
        score += RNG.normal(0, 7)  # noise
        score = float(np.clip(score, 0, 100))

        risk_level = bucket_risk(score)
        severity = severity_from_risk(risk_level)

        incident_date = start_date + timedelta(days=int(RNG.integers(0, 545)))
        hour = int(RNG.integers(6, 19)) if shift == "Day" else int(RNG.integers(19, 24))
        time_str = f"{hour:02d}:{int(RNG.integers(0, 60)):02d}"

        rows.append({
            "incident_id": f"INC-{i:04d}",
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
            "previous_incidents_30d": previous_incidents_30d,
            "crew_size": crew_size,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def generate_planned_activities(n=N_PLANNED_ACTIVITIES) -> pd.DataFrame:
    """Upcoming planned activities - NO risk_level here. These are what the Risk Predictor
    page will let the user pick from / use as starting points, then get a live prediction."""
    rows = []
    start_date = datetime(2026, 8, 17)  # a Monday, upcoming week

    for i in range(1, n + 1):
        activity = RNG.choice(ACTIVITY_TYPES)
        location = RNG.choice(ACTIVITY_LOCATION_MAP[activity])
        weather = RNG.choice(WEATHER_OPTIONS, p=[0.55, 0.20, 0.15, 0.10])
        shift = RNG.choice(SHIFT_OPTIONS, p=[0.75, 0.25])
        ppe_compliance = int(np.clip(RNG.normal(82, 12), 40, 100))
        crew_size = int(np.clip(RNG.normal(6, 3), 1, 20))
        planned_date = start_date + timedelta(days=int(RNG.integers(0, 14)))

        rows.append({
            "date": planned_date.strftime("%Y-%m-%d"),
            "activity_type": activity,
            "location_type": location,
            "crew_size": crew_size,
            "weather": weather,
            "shift": shift,
            "ppe_compliance_pct": ppe_compliance,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def main():
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    incidents_df = generate_incidents()
    planned_df = generate_planned_activities()

    incidents_path = os.path.join(out_dir, "incidents.csv")
    planned_path = os.path.join(out_dir, "planned_activities.csv")

    incidents_df.to_csv(incidents_path, index=False)
    planned_df.to_csv(planned_path, index=False)

    print(f"Wrote {len(incidents_df)} incidents to {incidents_path}")
    print(f"Wrote {len(planned_df)} planned activities to {planned_path}")
    print("\nRisk level distribution:")
    print(incidents_df["risk_level"].value_counts())
    print("\nSample rows:")
    print(incidents_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
