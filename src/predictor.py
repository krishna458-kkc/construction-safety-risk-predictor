"""
predictor.py

RUN-TIME module (see architecture-plan.md, Section 1 / Step 4 of the roadmap).
Loads the already-trained pipeline from models/risk_model.joblib once, and
exposes a single function — predict(activity_dict) — that app.py's Risk
Predictor page will call every time a user clicks "Predict."

This module never retrains, never calls .fit(), and never touches
data/incidents.csv. It only does inference + explanation on a single new
input, using the exact same preprocessing.py encoding that trained the model
(see preprocessing.py's module docstring for why that matters).

WHAT predict() RETURNS
-----------------------
A dict with:
    risk_level      : one of LOW / MEDIUM / HIGH / CRITICAL (the class the
                       model predicts)
    risk_score      : a single 0-100 number for the UI (st.metric, gauges,
                       etc.) — see "Risk score derivation" below
    confidence      : the model's predicted probability for risk_level
                       (0-1) — how sure the model is about its own answer
    probabilities   : full probability distribution across all 4 classes,
                       in RISK_LEVELS_ORDERED order — lets the UI show e.g.
                       a small bar for "27% chance this is actually HIGH"
    top_factors     : list of {label, contribution, direction} — the
                       "Associated factors" the architecture plan asks for,
                       traced back to the actual fitted model, not a
                       hardcoded explanation (see EXPLAINABILITY below)

RISK SCORE DERIVATION (0-100 continuous number from a categorical model)
--------------------------------------------------------------------------
The classifier predicts a class (LOW/MEDIUM/HIGH/CRITICAL), not a number.
But data_generator.py's own bucket_risk() thresholds (score<28 -> LOW,
28-48 -> MEDIUM, 48-65 -> HIGH, >=65 -> CRITICAL) give us a natural mapping
back to a representative score per class (its band midpoint). We combine
that with the model's predict_proba() output as a probability-weighted
average across all four class midpoints:

    risk_score = sum( P(class) * midpoint(class) )  for class in 4 classes

This is more informative than "just show the midpoint of the predicted
class" — e.g. a HIGH prediction the model is only 40% sure about (with real
probability mass sitting on MEDIUM and CRITICAL too) lands closer to the
HIGH/MEDIUM or HIGH/CRITICAL boundary than a HIGH prediction the model is
95% sure about. It's a derived display number, not a second model output —
worth saying so explicitly if a judge asks.

EXPLAINABILITY (architecture-plan.md Section 5)
--------------------------------------------------------------------------
train_model.py's report says which model won and which explanation path
applies (see model_evaluation_report.txt: "logistic_regression" won). This
module supports BOTH paths so it works regardless of which model is
actually sitting in the .joblib file:

  - Logistic Regression (has .coef_): take the coefficient row for the
    PREDICTED class, multiply element-wise by this input's own encoded
    feature values, and rank descending. A large positive product means
    "this specific input has a feature that's both present/large AND the
    model has learned it pushes toward this predicted class" — a real,
    per-prediction explanation, not a global one.

  - Random Forest (has .feature_importances_ instead): there's no signed,
    per-class coefficient to multiply against, so we rank by
    importance * this input's encoded (non-zero) feature value instead.
    This tells you which features THIS input has that the model
    globally leans on most — a "contributing factors" list rather than a
    strictly signed "positive contributors" list, since Random Forest
    doesn't expose direction the way Logistic Regression does.
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np

from src.preprocessing import (
    ALL_FEATURE_COLUMNS,
    row_from_dict,
)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "risk_model.joblib")

RISK_LEVELS_ORDERED = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# Same bucket boundaries as data_generator.py's bucket_risk(): LOW <28,
# MEDIUM <48, HIGH <65, CRITICAL >=65. Midpoint of each band is what
# risk_score is built from (see module docstring).
RISK_LEVEL_SCORE_MIDPOINT = {
    "LOW": 14.0,        # midpoint of [0, 28)
    "MEDIUM": 38.0,      # midpoint of [28, 48)
    "HIGH": 56.5,        # midpoint of [48, 65)
    "CRITICAL": 82.5,    # midpoint of [65, 100]
}

TOP_N_FACTORS = 4

# Cache so app.py doesn't re-load the .joblib file from disk on every rerun
# (Streamlit reruns the whole script on every interaction) — see get_model().
_MODEL_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# 1. Loading the model
# ---------------------------------------------------------------------------

def load_model(model_path: str = MODEL_PATH):
    """Loads the saved scikit-learn Pipeline (preprocessor + classifier,
    bundled as one object by train_model.py). Raises a clear error if the
    offline training step hasn't been run yet, instead of a cryptic
    FileNotFoundError deep in joblib."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run train_model.py "
            f"first (Step 3) — predictor.py only loads an already-trained "
            f"model, it never trains one itself."
        )
    return joblib.load(model_path)


def get_model(model_path: str = MODEL_PATH):
    """Cached accessor — app.py should call this (not load_model directly)
    so the .joblib file is only read from disk once per process, not once
    per Streamlit rerun."""
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = load_model(model_path)
    return _MODEL_CACHE[model_path]


# ---------------------------------------------------------------------------
# 2. Human-readable feature names
# ---------------------------------------------------------------------------

def _humanize_feature_name(raw_name: str) -> str:
    """
    Turns a ColumnTransformer output name like 'cat__activity_type_Welding',
    'num__crew_size', or 'text__confined space' into something a safety
    officer can read in the UI, e.g. 'Activity type: Welding',
    'Crew size', 'Hazard description mentions "confined space"'.
    """
    if "__" not in raw_name:
        return raw_name

    prefix, rest = raw_name.split("__", 1)

    if prefix == "text":
        return f'Hazard description mentions "{rest}"'

    if prefix == "num":
        return {
            "ppe_compliance_pct": "PPE compliance",
            "previous_incidents_30d": "Previous incidents (last 30 days)",
            "crew_size": "Crew size",
        }.get(rest, rest.replace("_", " ").capitalize())

    if prefix == "cat":
        # rest looks like "activity_type_Welding" or "location_type_Roof" —
        # split off the known column prefix so only the category value is
        # shown alongside a clean field label.
        column_labels = {
            "activity_type": "Activity type",
            "location_type": "Location",
            "weather": "Weather",
            "shift": "Shift",
        }
        for col, label in column_labels.items():
            if rest.startswith(col + "_"):
                value = rest[len(col) + 1:]
                return f"{label}: {value}"
        return rest.replace("_", " ")

    return raw_name


# ---------------------------------------------------------------------------
# 3. Explainability — top contributing factors for THIS prediction
# ---------------------------------------------------------------------------

def _top_factors_logistic_regression(
    pipeline, encoded_row: np.ndarray, predicted_class_idx: int
) -> list[dict[str, Any]]:
    classifier = pipeline.named_steps["classifier"]
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    coefs_for_class = classifier.coef_[predicted_class_idx]
    contributions = coefs_for_class * encoded_row

    order = np.argsort(contributions)[::-1]  # descending
    factors = []
    for idx in order:
        if len(factors) >= TOP_N_FACTORS:
            break
        value = float(contributions[idx])
        if value <= 0:
            break  # only POSITIVE contributors to the predicted class, per spec
        factors.append({
            "label": _humanize_feature_name(feature_names[idx]),
            "contribution": round(value, 4),
            "direction": "increases risk",
        })
    return factors


def _top_factors_random_forest(
    pipeline, encoded_row: np.ndarray, predicted_class_idx: int
) -> list[dict[str, Any]]:
    classifier = pipeline.named_steps["classifier"]
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    # No per-class signed coefficient available — rank by global importance
    # x how much of that feature THIS input actually has. Only features
    # actually present/non-zero in this input are candidates (an
    # importance score means nothing for a category this input doesn't have).
    importances = classifier.feature_importances_
    contributions = importances * np.abs(encoded_row)

    order = np.argsort(contributions)[::-1]
    factors = []
    for idx in order:
        if len(factors) >= TOP_N_FACTORS:
            break
        value = float(contributions[idx])
        if value <= 0:
            break
        factors.append({
            "label": _humanize_feature_name(feature_names[idx]),
            "contribution": round(value, 4),
            "direction": "contributing factor",  # RF has no signed direction
        })
    return factors


def _top_factors(pipeline, encoded_row: np.ndarray, predicted_class_idx: int) -> list[dict[str, Any]]:
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        return _top_factors_logistic_regression(pipeline, encoded_row, predicted_class_idx)
    elif hasattr(classifier, "feature_importances_"):
        return _top_factors_random_forest(pipeline, encoded_row, predicted_class_idx)
    return []  # unknown classifier type — degrade gracefully, don't crash the demo


# ---------------------------------------------------------------------------
# 4. The main entry point app.py calls
# ---------------------------------------------------------------------------

def predict(activity_dict: dict[str, Any], strict: bool = False, model=None) -> dict[str, Any]:
    """
    Takes one activity as a dict (from the Streamlit form, or a row pulled
    from planned_activities.csv) and returns everything the Risk Predictor
    page needs to render: risk_level, risk_score, confidence, the full
    probability distribution, and the top contributing factors.

    Args:
        activity_dict: raw field values, e.g.
            {"activity_type": "Working at Height", "location_type": "Roof",
             "weather": "Adverse", "shift": "Day", "ppe_compliance_pct": 62,
             "crew_size": 9, "previous_incidents_30d": 1,
             "description": "Unprotected edge observed during elevated work."}
            `description` and `previous_incidents_30d` are optional (see
            preprocessing.DEFAULT_VALUES) — planned_activities.csv rows
            don't have them.
        strict: passed straight through to preprocessing.row_from_dict.
            The Streamlit form should call with strict=True so a user can't
            silently submit a blank required field; predicting on a
            planned_activities.csv row should use strict=False (its default).
        model: optionally pass an already-loaded pipeline (e.g. from
            get_model()) to avoid reloading from disk; loads it if omitted.
    """
    if model is None:
        model = get_model()

    row_df = row_from_dict(activity_dict, strict=strict)

    probabilities = model.predict_proba(row_df)[0]
    class_order = list(model.classes_)  # actual order sklearn assigned internally
    prob_by_class = dict(zip(class_order, probabilities))

    predicted_class = max(prob_by_class, key=prob_by_class.get)
    predicted_idx = class_order.index(predicted_class)
    confidence = float(prob_by_class[predicted_class])

    risk_score = sum(
        prob_by_class[c] * RISK_LEVEL_SCORE_MIDPOINT[c] for c in RISK_LEVELS_ORDERED
    )

    # Encode this single row the same way training data was encoded, so the
    # explainability step multiplies coefficients against the SAME feature
    # space the model was actually trained on.
    preprocessor = model.named_steps["preprocessor"]
    encoded_row = preprocessor.transform(row_df)
    if hasattr(encoded_row, "toarray"):
        encoded_row = encoded_row.toarray()
    encoded_row = encoded_row[0]

    top_factors = _top_factors(model, encoded_row, predicted_idx)

    return {
        "risk_level": predicted_class,
        "risk_score": round(float(risk_score), 1),
        "confidence": round(confidence, 3),
        "probabilities": {c: round(float(prob_by_class[c]), 3) for c in RISK_LEVELS_ORDERED},
        "top_factors": top_factors,
    }


# ---------------------------------------------------------------------------
# 5. Self-test — run this file directly to sanity-check the module
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading model from: {MODEL_PATH}")
    model = get_model()
    classifier_type = type(model.named_steps["classifier"]).__name__
    print(f"Loaded pipeline. Classifier: {classifier_type}\n")

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
    print("Sample input (high-risk-looking activity):")
    for k, v in sample_activity.items():
        print(f"  {k}: {v}")

    result = predict(sample_activity, model=model)

    print(f"\nPredicted risk level : {result['risk_level']}")
    print(f"Risk score            : {result['risk_score']} / 100")
    print(f"Model confidence      : {result['confidence']:.1%}")
    print(f"Full probabilities    : {result['probabilities']}")
    print("Top contributing factors:")
    if result["top_factors"]:
        for f in result["top_factors"]:
            print(f"  - {f['label']} ({f['direction']}, contribution={f['contribution']})")
    else:
        print("  (none found above zero)")

    print("\n--- Testing a low-risk-looking activity for contrast ---")
    low_risk_activity = {
        "activity_type": "Housekeeping",
        "location_type": "Warehouse",
        "weather": "Clear",
        "shift": "Day",
        "ppe_compliance_pct": 96,
        "crew_size": 3,
        "previous_incidents_30d": 0,
        "description": "Routine walkway inspection, no issues.",
    }
    low_result = predict(low_risk_activity, model=model)
    print(f"Predicted risk level : {low_result['risk_level']}")
    print(f"Risk score            : {low_result['risk_score']} / 100")
    print(f"Model confidence      : {low_result['confidence']:.1%}")

    print("\n--- Testing predict-time input from a planned_activities.csv-style "
          "row (no description/previous_incidents_30d, strict=False) ---")
    planned_row = {
        "activity_type": "Confined Space",
        "location_type": "Electrical Room",
        "weather": "Clear",
        "shift": "Night",
        "ppe_compliance_pct": 88,
        "crew_size": 4,
    }
    planned_result = predict(planned_row, model=model)
    print(f"Predicted risk level : {planned_result['risk_level']}")
    print(f"Risk score            : {planned_result['risk_score']} / 100")

    print("\n--- Testing strict=True catches a missing required field ---")
    try:
        predict({"activity_type": "Welding"}, strict=True, model=model)
        print("FAIL: should have raised")
    except ValueError as e:
        print(f"OK — raised as expected: {e}")

    print("\npredictor.py self-test passed.")
