"""
safety_classifier.py

Defines SafetyAwareDecisionClassifier for packaging and serializing Candidate V4 model pipeline.
"""

from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline


class SafetyAwareDecisionClassifier(BaseEstimator, ClassifierMixin):
    """
    Wraps a fitted base pipeline and applies a safety-aware decision boundary rule.
    If the base model predicts MEDIUM, but P(HIGH) is within safety_margin of P(MEDIUM),
    the case is escalated to HIGH to prevent dangerous false negatives on borderline cases.
    """

    def __init__(self, base_pipeline: Pipeline, safety_margin: float = 0.05):
        self.base_pipeline = base_pipeline
        self.safety_margin = safety_margin
        self.classes_ = getattr(base_pipeline, "classes_", None)
        self.named_steps = getattr(base_pipeline, "named_steps", None)

    def fit(self, X, y):
        self.base_pipeline.fit(X, y)
        self.classes_ = self.base_pipeline.classes_
        self.named_steps = self.base_pipeline.named_steps
        return self

    def predict_proba(self, X):
        return self.base_pipeline.predict_proba(X)

    def predict(self, X):
        probs = self.predict_proba(X)
        class_list = list(self.classes_)
        med_idx = class_list.index("MEDIUM") if "MEDIUM" in class_list else -1
        high_idx = class_list.index("HIGH") if "HIGH" in class_list else -1

        raw_preds = np.argmax(probs, axis=1)
        final_preds = []

        for i, pred_idx in enumerate(raw_preds):
            pred_class = class_list[pred_idx]
            if pred_class == "MEDIUM" and med_idx != -1 and high_idx != -1:
                p_med = probs[i, med_idx]
                p_high = probs[i, high_idx]
                if (p_med - p_high) <= self.safety_margin:
                    final_preds.append("HIGH")
                else:
                    final_preds.append("MEDIUM")
            else:
                final_preds.append(pred_class)

        return np.array(final_preds)
