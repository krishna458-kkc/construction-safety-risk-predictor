"""
preprocessing.py

Shared feature-encoding logic for the Construction Safety Risk Predictor.

WHY THIS FILE EXISTS (see architecture-plan.md, Section 2):
    The exact same encoding — activity type -> one-hot, PPE % -> numeric scale,
    hazard description -> TF-IDF vector, etc. — has to happen both when TRAINING
    the model (train_model.py, on historical incidents.csv) and when PREDICTING
    on a brand-new form submission (predictor.py, on live user input). If that
    logic were duplicated in two places, they would eventually drift apart and
    predictions would silently break (e.g. a category seen at predict time but
    never seen at train time, or a differently-ordered feature vector). Every
    other module in this project imports FROM this file — this file imports
    from nothing project-specific.

WHAT THIS FILE DOES NOT DO:
    It does not fit a model. It does not read/write risk_model.joblib. It only
    defines (a) which columns are features, (b) how to turn a raw row/dict into
    a well-formed single-row DataFrame, and (c) an UNFITTED scikit-learn
    ColumnTransformer that train_model.py will .fit_transform() on the training
    set and predictor.py will re-use (already fitted, loaded from the saved
    pipeline) to .transform() a single new input. Keeping the transformer
    unfitted here, and only ever fitting it once inside the saved Pipeline in
    train_model.py, is what guarantees train-time and predict-time encoding
    can never diverge.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# 1. Feature schema — the single source of truth for "what goes into the model"
# ---------------------------------------------------------------------------
# Anything added/removed here automatically flows through to the
# ColumnTransformer, to training, and to prediction. Nothing downstream
# should hardcode a column list separately from this.

CATEGORICAL_FEATURES = ["activity_type", "location_type", "weather", "shift"]
NUMERICAL_FEATURES = ["ppe_compliance_pct", "previous_incidents_30d", "crew_size"]
TEXT_FEATURE = "description"

TARGET_COLUMN = "risk_level"

# All the columns a raw input row (training row OR a predict-time form dict)
# must be able to supply, one way or another.
ALL_FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TEXT_FEATURE]

# Sensible defaults for predict-time inputs. planned_activities.csv (see
# data_generator.py) does NOT include `description` or `previous_incidents_30d`
# — those are only known at the moment someone is actually about to run the
# activity, so the Risk Predictor form collects them live. If a caller omits
# them, we fall back to these rather than raising, so predictor.py can accept
# a partial dict built from a planned_activities.csv row.
DEFAULT_VALUES: dict[str, Any] = {
    "description": "",
    "previous_incidents_30d": 0,
}

# Valid category values, mirrored from data_generator.py. Used only for
# input validation / a helpful error message — NOT passed to OneHotEncoder
# (see note below on handle_unknown).
VALID_CATEGORIES: dict[str, list[str]] = {
    "activity_type": [
        "Working at Height", "Lifting", "Scaffolding", "Excavation",
        "Electrical Work", "Material Handling", "Welding", "Confined Space",
        "Vehicle Movement", "Housekeeping",
    ],
    "location_type": [
        "Building Site", "Roof", "Scaffolding Area", "Excavation Area",
        "Warehouse", "Loading Area", "Electrical Room", "Road/Access Area",
    ],
    "weather": ["Clear", "Rain", "Adverse", "Windy"],
    "shift": ["Day", "Night"],
}


# ---------------------------------------------------------------------------
# 2. Building the (unfitted) ColumnTransformer
# ---------------------------------------------------------------------------

def build_preprocessor(max_text_features: int = 100) -> ColumnTransformer:
    """
    Returns an UNFITTED ColumnTransformer implementing the encoding described
    in architecture-plan.md Section 5:

        categorical -> OneHotEncoder
        numerical   -> StandardScaler
        text        -> TfidfVectorizer (small vocab, so it stays interpretable
                        and doesn't overwhelm the ~7 structured features)

    train_model.py fits this (inside a Pipeline, alongside the classifier).
    predictor.py never calls this directly — it loads the already-fitted
    version out of the saved .joblib Pipeline. Defining it here just once
    is what keeps the two in sync.
    """
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"  # a category unseen at train time -> all-zero
                                  # row instead of a crash; safer for a live demo
    )
    numerical_transformer = StandardScaler()

    # TfidfVectorizer expects a 1-D iterable of strings, so it's applied to a
    # single column by name (ColumnTransformer passes a 1-D Series for a
    # string column key, unlike list-of-columns which passes a 2-D frame).
    text_transformer = TfidfVectorizer(
        max_features=max_text_features,
        stop_words="english",
        ngram_range=(1, 2),  # unigrams + bigrams: "confined space", "no ventilation"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("text", text_transformer, TEXT_FEATURE),
        ],
        remainder="drop",  # any extra column (incident_id, date, risk_score...)
                             # is intentionally ignored, not fed to the model
    )
    return preprocessor


def get_output_feature_names(fitted_preprocessor: ColumnTransformer) -> list[str]:
    """
    Human-readable name for every column the ColumnTransformer outputs, in
    order — e.g. 'cat__activity_type_Welding', 'num__crew_size',
    'text__confined space'. predictor.py's explainability step (Section 5:
    "top 3-4 positive contributors") needs this to turn a raw coefficient
    index back into something a safety officer can read. Only works AFTER
    the preprocessor has been fit (it needs to know which categories/vocab
    words actually showed up in training).
    """
    return list(fitted_preprocessor.get_feature_names_out())


# ---------------------------------------------------------------------------
# 3. Loading the training set
# ---------------------------------------------------------------------------

def load_training_data(incidents_csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Reads incidents.csv and splits it into X (feature columns only) and
    y (the risk_level label). Deliberately does NOT include risk_score as a
    feature: risk_score is the continuous value risk_level was bucketed FROM
    (see data_generator.py), so including it would let the model "cheat" by
    just re-reading the label instead of learning from the real-world-style
    features. Same reasoning excludes incident_id, date, time, severity —
    identifiers/outcomes, not predictive inputs.
    """
    df = pd.read_csv(incidents_csv_path)

    missing = [c for c in ALL_FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(
            f"incidents.csv is missing expected column(s): {missing}. "
            f"Was it generated by the current data_generator.py?"
        )

    X = df[ALL_FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


# ---------------------------------------------------------------------------
# 4. Turning a single predict-time input into a model-ready row
# ---------------------------------------------------------------------------

def row_from_dict(input_dict: dict[str, Any], strict: bool = False) -> pd.DataFrame:
    """
    Converts one form submission / planned-activity dict into a single-row
    DataFrame with exactly ALL_FEATURE_COLUMNS, in the right dtypes, ready to
    hand to a fitted pipeline's .transform() or .predict().

    - Fills `description` and `previous_incidents_30d` from DEFAULT_VALUES
      if the caller didn't supply them (this is the normal path when the
      input comes straight from a planned_activities.csv row).
    - If strict=True, raises on any missing required field or unrecognized
      category instead of silently defaulting/ignoring — useful for the
      Streamlit form, where we DO want to force the user to pick values
      rather than guess.
    """
    row: dict[str, Any] = {}

    for col in CATEGORICAL_FEATURES:
        if col not in input_dict or input_dict[col] in (None, ""):
            if strict:
                raise ValueError(f"Missing required field: '{col}'")
            row[col] = DEFAULT_VALUES.get(col, "")
        else:
            value = input_dict[col]
            if col in VALID_CATEGORIES and value not in VALID_CATEGORIES[col]:
                # Not fatal (handle_unknown="ignore" copes with it downstream),
                # but flag it clearly rather than silently mis-scoring.
                raise ValueError(
                    f"Unrecognized value {value!r} for '{col}'. "
                    f"Expected one of: {VALID_CATEGORIES[col]}"
                )
            row[col] = value

    for col in NUMERICAL_FEATURES:
        if col not in input_dict or input_dict[col] in (None, ""):
            if col in DEFAULT_VALUES:
                row[col] = DEFAULT_VALUES[col]
            elif strict:
                raise ValueError(f"Missing required field: '{col}'")
            else:
                row[col] = 0
        else:
            row[col] = input_dict[col]

    row[TEXT_FEATURE] = input_dict.get(TEXT_FEATURE, DEFAULT_VALUES[TEXT_FEATURE]) or ""

    return pd.DataFrame([row], columns=ALL_FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# 5. Self-test — run this file directly to sanity-check the module
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    here = os.path.dirname(__file__)
    incidents_path = os.path.abspath(os.path.join(here, "..", "data", "incidents.csv"))

    print(f"Loading training data from: {incidents_path}")
    X, y = load_training_data(incidents_path)
    print(f"X shape: {X.shape}   y shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    print(f"Label distribution:\n{y.value_counts()}\n")

    print("Fitting a throwaway preprocessor to check shapes / output names...")
    pre = build_preprocessor()
    X_transformed = pre.fit_transform(X, y)
    dense_shape = (
        X_transformed.shape if not hasattr(X_transformed, "toarray")
        else X_transformed.toarray().shape
    )
    print(f"Transformed feature matrix shape: {dense_shape}")

    names = get_output_feature_names(pre)
    print(f"Total output features: {len(names)}")
    print(f"First 10 feature names: {names[:10]}")

    print("\nSimulating a predict-time input from a planned_activities.csv-style row "
          "(no description, no previous_incidents_30d)...")
    sample_input = {
        "activity_type": "Working at Height",
        "location_type": "Roof",
        "weather": "Adverse",
        "shift": "Day",
        "ppe_compliance_pct": 62,
        "crew_size": 9,
    }
    row = row_from_dict(sample_input)
    print(row.to_string(index=False))

    row_transformed = pre.transform(row)
    row_shape = (
        row_transformed.shape if not hasattr(row_transformed, "toarray")
        else row_transformed.toarray().shape
    )
    print(f"Transformed single-row shape: {row_shape} (should match "
          f"(1, {dense_shape[1]}))")

    print("\nTesting strict mode catches a missing field...")
    try:
        row_from_dict({"activity_type": "Welding"}, strict=True)
        print("FAIL: should have raised")
    except ValueError as e:
        print(f"OK — raised as expected: {e}")

    print("\npreprocessing.py self-test passed.")
