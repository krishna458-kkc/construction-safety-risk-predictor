"""
train_model.py

OFFLINE / BUILD-TIME script (see architecture-plan.md Section 1). Run this
manually, once, whenever data/incidents.csv changes. It:

    1. Loads the historical incidents via preprocessing.load_training_data()
    2. Splits into train/validation (stratified, so the small CRITICAL class
       is represented in both)
    3. Trains TWO candidate models behind the SAME preprocessing pipeline:
         - Logistic Regression (multinomial)  -> directly interpretable
         - Random Forest                       -> usually stronger on mixed
                                                    categorical/numeric/text data
    4. Evaluates both on the validation set: accuracy, per-class precision/
       recall/F1, confusion matrix
    5. Picks a winner using validation RECALL ON {HIGH, CRITICAL} specifically
       — NOT raw accuracy (see rationale below)
    6. Saves the winning model as a single, ready-to-use scikit-learn
       Pipeline (preprocessor + classifier bundled together) to
       models/risk_model.joblib
    7. Writes a plain-text evaluation report and confusion-matrix images to
       outputs/, for demo prep / judge Q&A

WHY VALIDATION RECALL ON HIGH/CRITICAL, NOT ACCURACY (architecture-plan.md
Section 5):
    This is a safety tool. A model that's 85% "accurate" but misses half of
    the CRITICAL cases (calling them MEDIUM) is actively dangerous to rely
    on, even though its accuracy number looks fine — CRITICAL is also the
    smallest class, so accuracy alone hides poor performance on it. Missing
    a HIGH/CRITICAL case (false negative) is a far worse failure mode here
    than being over-cautious on a LOW case (false positive), so recall on
    those two classes specifically is the number this project actually cares
    about, and it's documented as the selection criterion rather than
    picked after the fact.

A NOTE ON EXPLAINABILITY AND WHICH MODEL WINS:
    architecture-plan.md Section 5 describes the explainability mechanism in
    terms of Logistic Regression's coefficients ("top 3-4 positive
    contributors" from coef_ * feature values). If Random Forest wins
    instead, predictor.py (Step 4) needs a parallel path using
    `feature_importances_` rather than `coef_` — both are printed below so
    whichever model wins, predictor.py can be built against the right one.
    This script prints which explainability path applies to the saved model.
"""

from __future__ import annotations

import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: never tries to open a GUI window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor, load_training_data

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
INCIDENTS_CSV = os.path.join(PROJECT_ROOT, "data", "incidents.csv")
MODEL_OUT_PATH = os.path.join(PROJECT_ROOT, "models", "risk_model.joblib")
REPORT_OUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "model_evaluation_report.txt")

RISK_LEVELS_ORDERED = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]  # fixed display order
PRIORITY_CLASSES = ["HIGH", "CRITICAL"]  # what the selection metric is based on

RANDOM_STATE = 42


def build_candidate_models() -> dict[str, Pipeline]:
    """
    Each candidate is a full Pipeline: preprocessing + classifier bundled as
    ONE object. That's what gets saved to .joblib — predictor.py will later
    load this single object and just call .predict()/.predict_proba() on a
    raw-ish DataFrame; it never has to know the preprocessing existed.

    class_weight="balanced" on both models: CRITICAL is ~10% of the data
    (see Step 2 output), so an unweighted model can reach OK accuracy while
    barely learning to recognize CRITICAL at all. Balancing re-weights the
    loss so rare classes aren't ignored — directly in service of the
    HIGH/CRITICAL recall goal above.
    """
    logreg = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            # solver="lbfgs" natively handles multinomial (softmax) loss for
            # multi-class problems in current scikit-learn - no multi_class=
            # kwarg needed (removed/deprecated in newer sklearn versions).
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    rf = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    return {"logistic_regression": logreg, "random_forest": rf}


def evaluate_model(name: str, model: Pipeline, X_val, y_val) -> dict:
    """Runs one fitted model on the validation set and returns everything
    needed for both the printed report and the model-selection decision."""
    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    report_text = classification_report(
        y_val, y_pred, labels=RISK_LEVELS_ORDERED, zero_division=0
    )
    cm = confusion_matrix(y_val, y_pred, labels=RISK_LEVELS_ORDERED)

    # Per-class recall, in RISK_LEVELS_ORDERED order, so we can pull out
    # HIGH and CRITICAL specifically for the selection metric.
    per_class_recall = recall_score(
        y_val, y_pred, labels=RISK_LEVELS_ORDERED, average=None, zero_division=0
    )
    recall_by_class = dict(zip(RISK_LEVELS_ORDERED, per_class_recall))
    priority_recall = float(np.mean([recall_by_class[c] for c in PRIORITY_CLASSES]))

    return {
        "name": name,
        "model": model,
        "accuracy": accuracy,
        "report_text": report_text,
        "confusion_matrix": cm,
        "recall_by_class": recall_by_class,
        "priority_recall": priority_recall,  # <- the actual selection criterion
    }


def explainability_note(model: Pipeline) -> str:
    """Which explainability path (Step 4) applies to this specific saved model."""
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        return (
            "Logistic Regression won: predictor.py should explain predictions using "
            "classifier.coef_ for the predicted class, multiplied by this input's "
            "encoded feature values (architecture-plan.md Section 5's original plan)."
        )
    elif hasattr(classifier, "feature_importances_"):
        return (
            "Random Forest won: predictor.py should explain predictions using "
            "classifier.feature_importances_ combined with this input's encoded "
            "feature values (global importance x local presence), since Random "
            "Forest has no per-class coefficients to use instead."
        )
    return "Unknown classifier type - explainability path needs manual review."


def save_confusion_matrix_plot(cm: np.ndarray, name: str, out_dir: str) -> str:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=RISK_LEVELS_ORDERED)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix — {name}")
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"confusion_matrix_{name}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_OUT_PATH), exist_ok=True)

    print(f"Loading training data from: {INCIDENTS_CSV}")
    X, y = load_training_data(INCIDENTS_CSV)
    print(f"Loaded {len(X)} rows.\n")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,          # keeps LOW/MEDIUM/HIGH/CRITICAL proportions in both splits
        random_state=RANDOM_STATE,
    )
    print(f"Train: {len(X_train)} rows   Validation: {len(X_val)} rows")
    print(f"Validation class counts:\n{y_val.value_counts()}\n")

    candidates = build_candidate_models()
    results = []

    for name, model in candidates.items():
        print(f"--- Training {name} ---")
        model.fit(X_train, y_train)
        result = evaluate_model(name, model, X_val, y_val)
        results.append(result)

        print(f"Accuracy: {result['accuracy']:.3f}")
        print(f"Recall by class: "
              + ", ".join(f"{c}={r:.3f}" for c, r in result["recall_by_class"].items()))
        print(f"--> Selection metric (avg recall on HIGH+CRITICAL): "
              f"{result['priority_recall']:.3f}")
        print(result["report_text"])
        print(f"Confusion matrix (rows=actual, cols=predicted, order={RISK_LEVELS_ORDERED}):")
        print(result["confusion_matrix"])
        print()

    # --- Model selection: highest avg recall on HIGH+CRITICAL wins. Ties broken by accuracy. ---
    winner = max(results, key=lambda r: (r["priority_recall"], r["accuracy"]))
    loser = next(r for r in results if r["name"] != winner["name"])

    print("=" * 70)
    print(f"WINNER: {winner['name']}")
    print(f"  {winner['name']}: priority_recall={winner['priority_recall']:.3f}, "
          f"accuracy={winner['accuracy']:.3f}")
    print(f"  {loser['name']}: priority_recall={loser['priority_recall']:.3f}, "
          f"accuracy={loser['accuracy']:.3f}")
    note = explainability_note(winner["model"])
    print(f"\n{note}")
    print("=" * 70)

    # --- Save the winning full pipeline (preprocessing + classifier, one object) ---
    joblib.dump(winner["model"], MODEL_OUT_PATH)
    print(f"\nSaved winning model to: {MODEL_OUT_PATH}")

    # --- Save confusion matrix plots for BOTH models (useful for the demo / judge Q&A) ---
    outputs_dir = os.path.dirname(REPORT_OUT_PATH)
    plot_paths = []
    for result in results:
        p = save_confusion_matrix_plot(result["confusion_matrix"], result["name"], outputs_dir)
        plot_paths.append(p)
        print(f"Saved confusion matrix plot: {p}")

    # --- Write a plain-text report summarizing everything, for demo prep ---
    with open(REPORT_OUT_PATH, "w") as f:
        f.write("CONSTRUCTION SAFETY RISK PREDICTOR - MODEL EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Training rows: {len(X_train)}   Validation rows: {len(X_val)}\n")
        f.write(f"Validation class distribution:\n{y_val.value_counts().to_string()}\n\n")

        for result in results:
            f.write("-" * 70 + "\n")
            f.write(f"MODEL: {result['name']}\n")
            f.write(f"Accuracy: {result['accuracy']:.3f}\n")
            f.write(f"Recall by class: "
                    + ", ".join(f"{c}={r:.3f}" for c, r in result["recall_by_class"].items())
                    + "\n")
            f.write(f"Selection metric (avg recall HIGH+CRITICAL): "
                    f"{result['priority_recall']:.3f}\n\n")
            f.write(result["report_text"] + "\n")
            f.write(f"Confusion matrix (rows=actual, cols=predicted, "
                    f"order={RISK_LEVELS_ORDERED}):\n")
            f.write(str(result["confusion_matrix"]) + "\n\n")

        f.write("=" * 70 + "\n")
        f.write(f"WINNER: {winner['name']}\n")
        f.write(f"Reason: highest avg recall on HIGH+CRITICAL "
                f"({winner['priority_recall']:.3f} vs {loser['priority_recall']:.3f})\n")
        f.write(f"{note}\n")
        f.write(f"Saved to: {MODEL_OUT_PATH}\n")

    print(f"\nSaved evaluation report to: {REPORT_OUT_PATH}")
    print("\ntrain_model.py finished successfully.")


if __name__ == "__main__":
    main()
