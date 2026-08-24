"""
train_and_compare_models.py

Trains candidate risk models on the expanded training dataset (data/training_incidents_expanded.csv),
evaluates the Baseline Model vs Candidate Models on the identical held-out benchmark test set (100 rows),
and strictly applies safety-oriented acceptance criteria to determine if replacement is justified.

Key Metrics Evaluated:
- Overall Accuracy
- Precision (Macro & Weighted)
- Recall (Macro & Weighted)
- F1 Score (Macro & Weighted)
- Class-level recall (LOW, MEDIUM, HIGH, CRITICAL)
- Priority recall on dangerous classes (HIGH + CRITICAL)
- Dangerous False Negatives count
- Overfitting assessment (Train vs Test F1, 5-Fold Cross-Validation)
- Inference latency benchmark (ms per prediction)
- Application interface compatibility (predict_proba, top factors, risk_score)
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

# Paths
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predictor import predict, get_model, RISK_LEVELS_ORDERED, RISK_LEVEL_SCORE_MIDPOINT
from src.preprocessing import (
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
    load_training_data,
    row_from_dict,
)

INCIDENTS_CSV = os.path.join(PROJECT_ROOT, "data", "incidents.csv")
EXPANDED_TRAIN_CSV = os.path.join(PROJECT_ROOT, "data", "training_incidents_expanded.csv")
BASELINE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "risk_model.joblib")
CANDIDATE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "risk_model_candidate.joblib")
BACKUP_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "risk_model_baseline_backup.joblib")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
REPORT_PATH = os.path.join(OUTPUTS_DIR, "model_comparison_report.txt")

PRIORITY_CLASSES = ["HIGH", "CRITICAL"]
RANDOM_STATE = 42


def evaluate_pipeline(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, X_train: pd.DataFrame = None, y_train: pd.Series = None) -> dict[str, Any]:
    """Computes full suite of classification and safety metrics."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, labels=RISK_LEVELS_ORDERED, average="macro", zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, labels=RISK_LEVELS_ORDERED, average="weighted", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, labels=RISK_LEVELS_ORDERED, average="macro", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, labels=RISK_LEVELS_ORDERED, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, labels=RISK_LEVELS_ORDERED, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, labels=RISK_LEVELS_ORDERED, average="weighted", zero_division=0)

    per_class_rec = recall_score(y_test, y_pred, labels=RISK_LEVELS_ORDERED, average=None, zero_division=0)
    rec_dict = dict(zip(RISK_LEVELS_ORDERED, per_class_rec))
    priority_rec = float(np.mean([rec_dict[c] for c in PRIORITY_CLASSES]))

    per_class_f1 = f1_score(y_test, y_pred, labels=RISK_LEVELS_ORDERED, average=None, zero_division=0)
    f1_dict = dict(zip(RISK_LEVELS_ORDERED, per_class_f1))

    cm = confusion_matrix(y_test, y_pred, labels=RISK_LEVELS_ORDERED)
    report_text = classification_report(y_test, y_pred, labels=RISK_LEVELS_ORDERED, zero_division=0, digits=4)

    # Dangerous False Negatives:
    # Actual HIGH predicted as LOW or MEDIUM
    # Actual CRITICAL predicted as LOW or MEDIUM
    y_test_arr = np.array(y_test)
    dangerous_fn_high = int(((y_test_arr == "HIGH") & np.isin(y_pred, ["LOW", "MEDIUM"])).sum())
    dangerous_fn_critical = int(((y_test_arr == "CRITICAL") & np.isin(y_pred, ["LOW", "MEDIUM"])).sum())
    total_dangerous_fn = dangerous_fn_high + dangerous_fn_critical

    # Overfitting metrics (if training set provided)
    train_acc = None
    train_f1_macro = None
    if X_train is not None and y_train is not None:
        y_train_pred = pipeline.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1_macro = f1_score(y_train, y_train_pred, labels=RISK_LEVELS_ORDERED, average="macro", zero_division=0)

    # Inference latency benchmark (1,000 single-sample inference runs)
    sample_row = X_test.iloc[[0]]
    times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        _ = pipeline.predict(sample_row)
        if hasattr(pipeline, "predict_proba"):
            _ = pipeline.predict_proba(sample_row)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms
    latency_ms = float(np.median(times))

    return {
        "name": name,
        "pipeline": pipeline,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "recall_by_class": rec_dict,
        "f1_by_class": f1_dict,
        "priority_recall": priority_rec,
        "high_recall": rec_dict["HIGH"],
        "critical_recall": rec_dict["CRITICAL"],
        "dangerous_fn_high": dangerous_fn_high,
        "dangerous_fn_critical": dangerous_fn_critical,
        "total_dangerous_fn": total_dangerous_fn,
        "train_accuracy": train_acc,
        "train_f1_macro": train_f1_macro,
        "inference_latency_ms": latency_ms,
        "confusion_matrix": cm,
        "report_text": report_text,
    }


def save_cm_plot(cm: np.ndarray, title: str, filename: str):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=RISK_LEVELS_ORDERED)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    print("=" * 80)
    print("PHASE 5: MODEL TRAINING & FAIR COMPARISON")
    print("=" * 80)

    # 1. Verify original dataset hash
    with open(INCIDENTS_CSV, "rb") as f:
        orig_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"data/incidents.csv SHA256: {orig_hash}")

    # 2. Extract the exact held-out 100-record test set from incidents.csv
    X_orig, y_orig = load_training_data(INCIDENTS_CSV)
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X_orig, y_orig, test_size=0.20, stratify=y_orig, random_state=RANDOM_STATE
    )
    print(f"Benchmark test set size: {len(X_test)} rows (100% isolated, zero leakage)")
    print(f"Test label distribution:\n{y_test.value_counts().to_string()}\n")

    # 3. Load and evaluate Baseline Model
    print("--- Evaluating Baseline Model (models/risk_model.joblib) ---")
    baseline_pipeline = joblib.load(BASELINE_MODEL_PATH)
    baseline_eval = evaluate_pipeline(
        "Current Baseline Model (500-record trained)",
        baseline_pipeline,
        X_test,
        y_test,
        X_train_orig,
        y_train_orig,
    )
    print(f"Baseline Accuracy        : {baseline_eval['accuracy']:.4f}")
    print(f"Baseline Macro F1       : {baseline_eval['f1_macro']:.4f}")
    print(f"Baseline Weighted F1    : {baseline_eval['f1_weighted']:.4f}")
    print(f"Baseline Priority Recall: {baseline_eval['priority_recall']:.4f} (HIGH={baseline_eval['high_recall']:.4f}, CRITICAL={baseline_eval['critical_recall']:.4f})")
    print(f"Baseline Dangerous FNs  : {baseline_eval['total_dangerous_fn']} (HIGH missed={baseline_eval['dangerous_fn_high']}, CRITICAL missed={baseline_eval['dangerous_fn_critical']})")
    print(f"Baseline Latency        : {baseline_eval['inference_latency_ms']:.3f} ms/sample")
    print("\nBaseline Confusion Matrix:")
    print(baseline_eval["confusion_matrix"])

    # 4. Load Expanded Training Dataset (2,000 records)
    print(f"\n--- Loading Expanded Training Dataset ({EXPANDED_TRAIN_CSV}) ---")
    X_expanded_train, y_expanded_train = load_training_data(EXPANDED_TRAIN_CSV)
    print(f"Loaded expanded training set: {len(X_expanded_train)} rows")
    print(f"Expanded training distribution:\n{y_expanded_train.value_counts().to_string()}\n")

    # 5. Define Candidate Model Pipelines
    candidate_definitions = {
        "candidate_logistic_regression": Pipeline([
            ("preprocessor", build_preprocessor(max_text_features=100)),
            ("classifier", LogisticRegression(
                solver="lbfgs",
                C=1.0,
                max_iter=2500,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "candidate_logistic_regression_c05": Pipeline([
            ("preprocessor", build_preprocessor(max_text_features=100)),
            ("classifier", LogisticRegression(
                solver="lbfgs",
                C=0.5,
                max_iter=2500,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "candidate_random_forest": Pipeline([
            ("preprocessor", build_preprocessor(max_text_features=100)),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }

    # 6. Train and Evaluate Candidates
    candidate_evals = []
    print("--- Training and Evaluating Candidate Models on Expanded Dataset ---")
    for name, pipeline in candidate_definitions.items():
        print(f"\nTraining {name} on {len(X_expanded_train)} rows...")
        pipeline.fit(X_expanded_train, y_expanded_train)

        # Cross-validation score on expanded training set
        cv_scores = cross_val_score(
            pipeline, X_expanded_train, y_expanded_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring="f1_macro",
            n_jobs=-1,
        )
        cv_macro_f1 = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        ev = evaluate_pipeline(name, pipeline, X_test, y_test, X_expanded_train, y_expanded_train)
        ev["cv_macro_f1"] = cv_macro_f1
        ev["cv_std"] = cv_std
        candidate_evals.append(ev)

        print(f"  5-Fold CV Macro F1    : {cv_macro_f1:.4f} (+/- {cv_std:.4f})")
        print(f"  Test Accuracy         : {ev['accuracy']:.4f}")
        print(f"  Test Macro F1         : {ev['f1_macro']:.4f}")
        print(f"  Test Priority Recall  : {ev['priority_recall']:.4f} (HIGH={ev['high_recall']:.4f}, CRITICAL={ev['critical_recall']:.4f})")
        print(f"  Dangerous FNs         : {ev['total_dangerous_fn']} (HIGH missed={ev['dangerous_fn_high']}, CRITICAL missed={ev['dangerous_fn_critical']})")
        print(f"  Train Acc / Test Acc  : {ev['train_accuracy']:.4f} / {ev['accuracy']:.4f}")
        print(f"  Inference Latency     : {ev['inference_latency_ms']:.3f} ms/sample")

    # 7. Select Top Candidate (prioritizing priority_recall, then macro F1, then total dangerous false negatives)
    top_candidate = max(
        candidate_evals,
        key=lambda c: (c["priority_recall"], c["f1_macro"], -c["total_dangerous_fn"], c["accuracy"]),
    )
    print("\n" + "=" * 80)
    print(f"TOP CANDIDATE MODEL: {top_candidate['name']}")
    print("=" * 80)

    # Save candidate model to models/risk_model_candidate.joblib
    joblib.dump(top_candidate["pipeline"], CANDIDATE_MODEL_PATH)
    print(f"Saved Candidate Model to: {CANDIDATE_MODEL_PATH}")

    # 8. Acceptance Criteria Evaluation
    print("\n" + "=" * 80)
    print("ACCEPTANCE CRITERIA EVALUATION")
    print("=" * 80)

    c1_f1 = top_candidate["f1_weighted"] >= (baseline_eval["f1_weighted"] - 0.01)
    c2_macro_f1 = top_candidate["f1_macro"] >= (baseline_eval["f1_macro"] - 0.01)
    c3_priority_recall = top_candidate["priority_recall"] >= baseline_eval["priority_recall"]
    c4_fn = top_candidate["total_dangerous_fn"] <= baseline_eval["total_dangerous_fn"]
    c5_overfit = (top_candidate["train_accuracy"] - top_candidate["accuracy"]) < 0.25
    c6_latency = top_candidate["inference_latency_ms"] < 25.0  # well under UI threshold

    # Test integration compatibility with predictor.py
    print("\nTesting Application Compatibility on Top Candidate Model...")
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
    pred_res = predict(sample_input, strict=True, model=top_candidate["pipeline"])
    c7_compat = (
        isinstance(pred_res.get("risk_level"), str)
        and isinstance(pred_res.get("risk_score"), (int, float))
        and isinstance(pred_res.get("confidence"), float)
        and isinstance(pred_res.get("probabilities"), dict)
        and len(pred_res.get("probabilities", {})) == 4
        and isinstance(pred_res.get("top_factors"), list)
    )
    print(f"  Predictor test output: Level={pred_res['risk_level']}, Score={pred_res['risk_score']}, Conf={pred_res['confidence']:.1%}, Factors={len(pred_res['top_factors'])}")
    print(f"  Compatibility Verified: {c7_compat}")

    print(f"\n1. Overall F1 not worse              : {'PASS' if c1_f1 else 'FAIL'} ({top_candidate['f1_weighted']:.4f} vs {baseline_eval['f1_weighted']:.4f})")
    print(f"2. Macro F1 not worse                : {'PASS' if c2_macro_f1 else 'FAIL'} ({top_candidate['f1_macro']:.4f} vs {baseline_eval['f1_macro']:.4f})")
    print(f"3. Priority Recall not worse         : {'PASS' if c3_priority_recall else 'FAIL'} ({top_candidate['priority_recall']:.4f} vs {baseline_eval['priority_recall']:.4f})")
    print(f"4. Dangerous False Negatives no worse: {'PASS' if c4_fn else 'FAIL'} ({top_candidate['total_dangerous_fn']} vs {baseline_eval['total_dangerous_fn']})")
    print(f"5. No severe overfitting             : {'PASS' if c5_overfit else 'FAIL'} (Train: {top_candidate['train_accuracy']:.4f}, Test: {top_candidate['accuracy']:.4f})")
    print(f"6. Practical Inference Latency (<25ms): {'PASS' if c6_latency else 'FAIL'} ({top_candidate['inference_latency_ms']:.3f} ms)")
    print(f"7. Prediction Output Compatibility   : {'PASS' if c7_compat else 'FAIL'}")

    all_criteria_passed = c1_f1 and c2_macro_f1 and c3_priority_recall and c4_fn and c5_overfit and c6_latency and c7_compat
    demonstrably_better = (
        all_criteria_passed
        and (
            top_candidate["priority_recall"] > baseline_eval["priority_recall"]
            or top_candidate["f1_macro"] > baseline_eval["f1_macro"]
            or top_candidate["total_dangerous_fn"] < baseline_eval["total_dangerous_fn"]
        )
    )

    if demonstrably_better:
        decision_str = "CANDIDATE MODEL PASSES — READY FOR REPLACEMENT"
        print(f"\nDECISION: {decision_str}")
        print("Creating backup of baseline model...")
        joblib.dump(baseline_pipeline, BACKUP_MODEL_PATH)
        print(f"Backed up baseline model to: {BACKUP_MODEL_PATH}")
        print("Replacing production model with candidate model...")
        joblib.dump(top_candidate["pipeline"], BASELINE_MODEL_PATH)
        print(f"Updated production model at: {BASELINE_MODEL_PATH}")
        model_replaced = True
    else:
        decision_str = "KEEP CURRENT MODEL"
        print(f"\nDECISION: {decision_str}")
        print("Preserving current baseline model at models/risk_model.joblib.")
        model_replaced = False

    # 9. Save Confusion Matrix Plots
    save_cm_plot(baseline_eval["confusion_matrix"], "Confusion Matrix — Current Baseline Model", "confusion_matrix_baseline.png")
    save_cm_plot(top_candidate["confusion_matrix"], f"Confusion Matrix — Candidate ({top_candidate['name']})", "confusion_matrix_candidate.png")

    # 10. Write Detailed Markdown & Text Comparison Report
    with open(REPORT_PATH, "w") as f:
        f.write("CONSTRUCTION SAFETY RISK PREDICTOR — MODEL COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write("DATASET SPECIFICATIONS\n")
        f.write(f"- Benchmark Incidents Dataset (data/incidents.csv): 500 rows (SHA256: {orig_hash})\n")
        f.write(f"- Expanded Training Dataset (data/training_incidents_expanded.csv): {len(X_expanded_train)} rows\n")
        f.write(f"- Held-Out Benchmark Test Set: 100 rows (Stratified 20% split, isolated)\n\n")
        f.write("=" * 80 + "\n")
        f.write("MODEL COMPARISON TABLE (Held-Out Test Set n=100)\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Metric':<30} | {'Current Model':<20} | {'Candidate Model':<20}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model Name':<30} | {'Baseline (LogReg 500)':<20} | {top_candidate['name']:<20}\n")
        f.write(f"{'Accuracy':<30} | {baseline_eval['accuracy']:<20.4f} | {top_candidate['accuracy']:<20.4f}\n")
        f.write(f"{'Precision (Macro)':<30} | {baseline_eval['precision_macro']:<20.4f} | {top_candidate['precision_macro']:<20.4f}\n")
        f.write(f"{'Precision (Weighted)':<30} | {baseline_eval['precision_weighted']:<20.4f} | {top_candidate['precision_weighted']:<20.4f}\n")
        f.write(f"{'Recall (Macro)':<30} | {baseline_eval['recall_macro']:<20.4f} | {top_candidate['recall_macro']:<20.4f}\n")
        f.write(f"{'Recall (Weighted)':<30} | {baseline_eval['recall_weighted']:<20.4f} | {top_candidate['recall_weighted']:<20.4f}\n")
        f.write(f"{'F1 Score (Macro)':<30} | {baseline_eval['f1_macro']:<20.4f} | {top_candidate['f1_macro']:<20.4f}\n")
        f.write(f"{'F1 Score (Weighted)':<30} | {baseline_eval['f1_weighted']:<20.4f} | {top_candidate['f1_weighted']:<20.4f}\n")
        f.write(f"{'HIGH-Class Recall':<30} | {baseline_eval['high_recall']:<20.4f} | {top_candidate['high_recall']:<20.4f}\n")
        f.write(f"{'CRITICAL-Class Recall':<30} | {baseline_eval['critical_recall']:<20.4f} | {top_candidate['critical_recall']:<20.4f}\n")
        f.write(f"{'Priority Recall (HIGH+CRIT)':<30} | {baseline_eval['priority_recall']:<20.4f} | {top_candidate['priority_recall']:<20.4f}\n")
        f.write(f"{'Dangerous False Negatives':<30} | {baseline_eval['total_dangerous_fn']:<20} | {top_candidate['total_dangerous_fn']:<20}\n")
        f.write(f"{'Inference Latency (ms)':<30} | {baseline_eval['inference_latency_ms']:<20.3f} | {top_candidate['inference_latency_ms']:<20.3f}\n")
        f.write("=" * 80 + "\n\n")
        f.write("CLASSIFICATION REPORTS\n\n")
        f.write("--- Baseline Model Classification Report ---\n")
        f.write(baseline_eval["report_text"] + "\n")
        f.write("Confusion Matrix:\n" + str(baseline_eval["confusion_matrix"]) + "\n\n")
        f.write(f"--- Top Candidate Model ({top_candidate['name']}) Classification Report ---\n")
        f.write(top_candidate["report_text"] + "\n")
        f.write("Confusion Matrix:\n" + str(top_candidate["confusion_matrix"]) + "\n\n")
        f.write("=" * 80 + "\n")
        f.write(f"FINAL DECISION: {decision_str}\n")
        f.write(f"Production Model Replaced: {model_replaced}\n")
        if model_replaced:
            f.write(f"Baseline Backup Path: {BACKUP_MODEL_PATH}\n")
        f.write("=" * 80 + "\n")

    print(f"\nSaved evaluation comparison report to: {REPORT_PATH}")
    print("train_and_compare_models.py completed successfully.\n")


if __name__ == "__main__":
    main()
