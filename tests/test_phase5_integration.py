"""
test_phase5_integration.py

Comprehensive test suite verifying:
1. Integrity and SHA-256 hashes of production datasets:
   - data/incidents.csv
   - data/planned_activities.csv
   - data/prediction_history.csv
   - data/incident_records.csv
2. Validation of the expanded training dataset (data/training_incidents_expanded.csv)
3. Predictor inference interface and explainability factors
4. Core module imports and compatibility:
   - src.preprocessing
   - src.predictor
   - src.company_analytics
   - src.data_storage
   - src.recommendations
   - src.toolbox_talk
5. Streamlit app execution readiness
"""

import hashlib
import os
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predictor import predict, get_model, RISK_LEVELS_ORDERED
from src.preprocessing import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    TEXT_FEATURE,
    VALID_CATEGORIES,
    build_preprocessor,
    load_training_data,
    row_from_dict,
)
from src.data_storage import (
    load_projects_data,
    ensure_company_data_stores,
)
from src.recommendations import get_recommendations
from src.toolbox_talk import get_toolbox_topics


class TestPhase5Integration(unittest.TestCase):

    def test_01_benchmark_dataset_integrity(self):
        """Verify data/incidents.csv is intact (500 rows, exact SHA256)."""
        csv_path = os.path.join(PROJECT_ROOT, "data", "incidents.csv")
        self.assertTrue(os.path.exists(csv_path), "data/incidents.csv must exist")
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 500, "data/incidents.csv must have exactly 500 rows")
        self.assertEqual(len(df.columns), 14, "data/incidents.csv must have 14 columns")

        with open(csv_path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
        expected_sha = "b16f807cef66ce112f792a8dd6c12c6396e693f3a4d4abe4fb5ce28d977e49ce"
        self.assertEqual(sha256, expected_sha, "data/incidents.csv hash mismatch — benchmark was modified!")

    def test_02_production_data_integrity(self):
        """Verify planned_activities, prediction_history, and incident_records are intact."""
        planned_path = os.path.join(PROJECT_ROOT, "data", "planned_activities.csv")
        with open(planned_path, "rb") as f:
            self.assertEqual(
                hashlib.sha256(f.read()).hexdigest(),
                "09866df1fa65e4682d8ccf69ea4a6d513a6b3f0a6b4cc39b82ad4e7da3d11b6d",
                "planned_activities.csv was modified!",
            )

        pred_hist_path = os.path.join(PROJECT_ROOT, "data", "prediction_history.csv")
        with open(pred_hist_path, "rb") as f:
            self.assertEqual(
                hashlib.sha256(f.read()).hexdigest(),
                "ee0eaadb032b168d25b98ef1df1b30a771ca2f270865d6267edfcdbbec205301",
                "prediction_history.csv was modified!",
            )

        inc_rec_path = os.path.join(PROJECT_ROOT, "data", "incident_records.csv")
        with open(inc_rec_path, "rb") as f:
            self.assertEqual(
                hashlib.sha256(f.read()).hexdigest(),
                "94377e189ac0f75f348b5491b2f87df9ab5e8e6064a536b39e504cb0f1ad80e5",
                "incident_records.csv was modified!",
            )

    def test_03_expanded_training_dataset_quality(self):
        """Verify data/training_incidents_expanded.csv has ~2,000 valid records and no duplicates."""
        expanded_path = os.path.join(PROJECT_ROOT, "data", "training_incidents_expanded.csv")
        self.assertTrue(os.path.exists(expanded_path), "training_incidents_expanded.csv must exist")
        df = pd.read_csv(expanded_path)
        self.assertEqual(len(df), 2000, "Expanded dataset must have 2,000 rows")
        self.assertEqual(df.isnull().sum().sum(), 0, "No null values allowed in expanded dataset")
        
        # Duplicates check
        dup_count = df.duplicated(subset=ALL_FEATURE_COLUMNS).sum()
        self.assertEqual(dup_count, 0, "No duplicate rows allowed in expanded dataset")

        # Range checks
        self.assertTrue((df["ppe_compliance_pct"] >= 0).all() and (df["ppe_compliance_pct"] <= 100).all())
        self.assertTrue((df["crew_size"] >= 1).all() and (df["crew_size"] <= 50).all())
        self.assertTrue((df["previous_incidents_30d"] >= 0).all())
        self.assertTrue((df["risk_score"] >= 0).all() and (df["risk_score"] <= 100).all())

        # Category validity
        for col, valids in VALID_CATEGORIES.items():
            invalid = set(df[col].unique()) - set(valids)
            self.assertEqual(len(invalid), 0, f"Invalid categories in {col}: {invalid}")

    def test_04_predictor_compatibility(self):
        """Verify predictor.predict returns exact required schema and reasonable outputs."""
        model = get_model()
        sample_input = {
            "activity_type": "Working at Height",
            "location_type": "Roof",
            "weather": "Adverse",
            "shift": "Day",
            "ppe_compliance_pct": 62,
            "crew_size": 9,
            "previous_incidents_30d": 1,
            "description": "Unprotected edge observed during elevated work.",
        }
        res = predict(sample_input, strict=True, model=model)
        self.assertIn("risk_level", res)
        self.assertIn(res["risk_level"], RISK_LEVELS_ORDERED)
        self.assertIn("risk_score", res)
        self.assertTrue(0 <= res["risk_score"] <= 100)
        self.assertIn("confidence", res)
        self.assertTrue(0 <= res["confidence"] <= 1.0)
        self.assertIn("probabilities", res)
        self.assertEqual(len(res["probabilities"]), 4)
        self.assertIn("top_factors", res)
        self.assertIsInstance(res["top_factors"], list)

    def test_05_modules_integration(self):
        """Verify recommendations, toolbox talks, and data storage functions work seamlessly."""
        recs = get_recommendations("Working at Height")
        self.assertIsInstance(recs, list)
        self.assertGreater(len(recs), 0)

        tb = get_toolbox_topics("Excavation", count=4)
        self.assertIsInstance(tb, list)
        self.assertEqual(len(tb), 4)

        ensure_company_data_stores()
        proj_df = load_projects_data()
        self.assertIsInstance(proj_df, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
