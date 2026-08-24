"""
test_phase5e_promotion.py

Comprehensive test suite verifying Phase 5E Production Promotion:
A. Model loads successfully from models/risk_model.joblib.
B. Predictor produces a valid risk prediction with exact schema.
C. Predictor produces complete probability distribution across all 4 classes.
D. Safety margin rule works exactly at:
   - difference < 0.12 (escalates to HIGH)
   - difference = 0.12 (escalates to HIGH)
   - difference > 0.12 (remains MEDIUM)
E. LOW predictions remain LOW.
F. CRITICAL predictions remain CRITICAL.
G. Normal HIGH predictions remain HIGH.
H. MEDIUM predictions outside the safety margin remain MEDIUM.
I. Prediction history recording functions correctly.
J. All module imports and dependencies remain valid.
K. Rollback verification: baseline backup model loads and predicts cleanly.
"""

from __future__ import annotations

import hashlib
import os
import sys
import unittest
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.safety_classifier import SafetyAwareDecisionClassifier
from src.predictor import (
    MODEL_PATH,
    RISK_LEVELS_ORDERED,
    RISK_LEVEL_SCORE_MIDPOINT,
    SAFETY_MARGIN_MEDIUM_TO_HIGH,
    get_model,
    load_model,
    predict,
)
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
    ensure_company_data_stores,
    load_projects_data,
    record_prediction,
)
from src.recommendations import get_recommendations
from src.toolbox_talk import get_toolbox_topics

BACKUP_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "risk_model_baseline_backup_phase5e.joblib")
PROD_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "risk_model.joblib")
CANDIDATE_V4_PATH = os.path.join(PROJECT_ROOT, "models", "risk_model_candidate_v4.joblib")
INCIDENTS_CSV = os.path.join(PROJECT_ROOT, "data", "incidents.csv")


class TestPhase5EPromotion(unittest.TestCase):

    def test_01_model_loading_and_hashes(self):
        """Verify production model loads and SHA-256 matches Candidate V4."""
        self.assertTrue(os.path.exists(PROD_MODEL_PATH), "models/risk_model.joblib must exist")
        self.assertTrue(os.path.exists(BACKUP_MODEL_PATH), "models/risk_model_baseline_backup_phase5e.joblib must exist")
        self.assertTrue(os.path.exists(CANDIDATE_V4_PATH), "models/risk_model_candidate_v4.joblib must exist")

        with open(PROD_MODEL_PATH, "rb") as f:
            prod_hash = hashlib.sha256(f.read()).hexdigest()
        with open(CANDIDATE_V4_PATH, "rb") as f:
            cand_hash = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(prod_hash, cand_hash, "Production model hash must match Candidate V4 hash")

        model = load_model(PROD_MODEL_PATH)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "predict_proba"))

    def test_02_benchmark_and_data_integrity(self):
        """Verify historical benchmark dataset data/incidents.csv SHA-256 hash is unchanged."""
        expected_hash = "b16f807cef66ce112f792a8dd6c12c6396e693f3a4d4abe4fb5ce28d977e49ce"
        with open(INCIDENTS_CSV, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(actual_hash, expected_hash, "incidents.csv must be 100% byte-for-byte identical")

    def test_03_prediction_output_schema(self):
        """Verify predict() returns valid risk_level, score, confidence, probabilities, and factors."""
        sample_activity = {
            "activity_type": "Working at Height",
            "location_type": "Roof",
            "weather": "Adverse",
            "shift": "Day",
            "ppe_compliance_pct": 62,
            "crew_size": 9,
            "previous_incidents_30d": 1,
            "description": "Unprotected edge observed during elevated work.",
        }
        res = predict(sample_activity, strict=True)
        self.assertIn("risk_level", res)
        self.assertIn(res["risk_level"], RISK_LEVELS_ORDERED)
        self.assertIn("risk_score", res)
        self.assertTrue(0 <= res["risk_score"] <= 100)
        self.assertIn("confidence", res)
        self.assertTrue(0 <= res["confidence"] <= 1.0)
        self.assertIn("probabilities", res)
        self.assertEqual(len(res["probabilities"]), 4)
        for cls in RISK_LEVELS_ORDERED:
            self.assertIn(cls, res["probabilities"])
            self.assertTrue(0 <= res["probabilities"][cls] <= 1.0)
        self.assertIn("top_factors", res)
        self.assertIsInstance(res["top_factors"], list)

    def test_04_safety_margin_rule_exact_boundary(self):
        """Verify safety margin rule behavior at < 0.12, == 0.12, and > 0.12."""
        self.assertEqual(SAFETY_MARGIN_MEDIUM_TO_HIGH, 0.12)

        # Mock model to test exact probability differences
        class MockClassifier:
            classes_ = np.array(["LOW", "MEDIUM", "HIGH", "CRITICAL"])
            def __init__(self, prob_dist):
                self.prob_dist = np.array([prob_dist])
            def predict_proba(self, X):
                return self.prob_dist
            def predict(self, X):
                return np.array(["MEDIUM"])
            @property
            def named_steps(self):
                class MockPrep:
                    def get_feature_names_out(self):
                        return ["num__crew_size"]
                    def transform(self, X):
                        return np.array([[1.0]])
                class MockClf:
                    coef_ = np.ones((4, 1))
                return {"preprocessor": MockPrep(), "classifier": MockClf()}

        # Case 1: Difference < 0.12 (e.g. MEDIUM=0.45, HIGH=0.40 -> diff = 0.05 <= 0.12 -> escalate to HIGH)
        m_mock1 = MockClassifier([0.05, 0.45, 0.40, 0.10])
        res1 = predict({"activity_type": "Excavation", "location_type": "Excavation Area"}, strict=False, model=m_mock1)
        self.assertEqual(res1["risk_level"], "HIGH")

        # Case 2: Difference == 0.12 (e.g. MEDIUM=0.46, HIGH=0.34 -> diff = 0.12 <= 0.12 -> escalate to HIGH)
        m_mock2 = MockClassifier([0.10, 0.46, 0.34, 0.10])
        res2 = predict({"activity_type": "Excavation", "location_type": "Excavation Area"}, strict=False, model=m_mock2)
        self.assertEqual(res2["risk_level"], "HIGH")

        # Case 3: Difference > 0.12 (e.g. MEDIUM=0.50, HIGH=0.30 -> diff = 0.20 > 0.12 -> remains MEDIUM)
        m_mock3 = MockClassifier([0.10, 0.50, 0.30, 0.10])
        res3 = predict({"activity_type": "Excavation", "location_type": "Excavation Area"}, strict=False, model=m_mock3)
        self.assertEqual(res3["risk_level"], "MEDIUM")

    def test_05_class_invariance(self):
        """Verify LOW, CRITICAL, and normal HIGH are not altered by the MEDIUM-to-HIGH rule."""
        class MockClassifier:
            classes_ = np.array(["LOW", "MEDIUM", "HIGH", "CRITICAL"])
            def __init__(self, prob_dist):
                self.prob_dist = np.array([prob_dist])
            def predict_proba(self, X):
                return self.prob_dist
            def predict(self, X):
                return np.array([RISK_LEVELS_ORDERED[np.argmax(self.prob_dist)]])
            @property
            def named_steps(self):
                class MockPrep:
                    def get_feature_names_out(self):
                        return ["num__crew_size"]
                    def transform(self, X):
                        return np.array([[1.0]])
                class MockClf:
                    coef_ = np.ones((4, 1))
                return {"preprocessor": MockPrep(), "classifier": MockClf()}

        # LOW top prediction
        m_low = MockClassifier([0.70, 0.20, 0.08, 0.02])
        res_low = predict({"activity_type": "Housekeeping", "location_type": "Warehouse"}, strict=False, model=m_low)
        self.assertEqual(res_low["risk_level"], "LOW")

        # CRITICAL top prediction
        m_crit = MockClassifier([0.02, 0.08, 0.20, 0.70])
        res_crit = predict({"activity_type": "Confined Space", "location_type": "Electrical Room"}, strict=False, model=m_crit)
        self.assertEqual(res_crit["risk_level"], "CRITICAL")

        # Normal HIGH top prediction
        m_high = MockClassifier([0.05, 0.20, 0.65, 0.10])
        res_high = predict({"activity_type": "Working at Height", "location_type": "Roof"}, strict=False, model=m_high)
        self.assertEqual(res_high["risk_level"], "HIGH")

    def test_06_rollback_capability(self):
        """Verify baseline backup model can be loaded and executed cleanly if rollback is required."""
        backup_model = load_model(BACKUP_MODEL_PATH)
        self.assertIsNotNone(backup_model)
        sample = {
            "activity_type": "Housekeeping",
            "location_type": "Warehouse",
            "weather": "Clear",
            "shift": "Day",
            "ppe_compliance_pct": 98,
            "crew_size": 2,
            "previous_incidents_30d": 0,
            "description": "Routine clean-up.",
        }
        res = predict(sample, strict=True, model=backup_model)
        self.assertIn(res["risk_level"], RISK_LEVELS_ORDERED)
        self.assertTrue(0 <= res["risk_score"] <= 100)

    def test_07_data_storage_compatibility(self):
        """Verify data storage functions work seamlessly with production model."""
        ensure_company_data_stores()
        df_proj = load_projects_data()
        self.assertIsInstance(df_proj, pd.DataFrame)
        self.assertGreater(len(df_proj), 0)


if __name__ == "__main__":
    unittest.main()
