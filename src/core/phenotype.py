"""
Dysphagia phenotype classification from manifold metrics.

Implements Section 6: clinical phenotypes as manifold distortions.
Each phenotype has a distinct geometric signature:
- Fibrosis: contracted manifold, low curvature, short range
- Weakness: preserved manifold, low speed
- Aspiration risk: bottleneck traversal failure
- Compensation: branching trajectories, high path length
- Neurogenic: loss of reproducibility
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from typing import Dict, List, Tuple, Optional
import pandas as pd

from .trajectory import SwallowingTrajectory
from .metrics import extract_all_metrics


PHENOTYPE_LABELS = {
    0: "healthy",
    1: "fibrotic",
    2: "weak",
    3: "compensatory",
    4: "neurogenic",
}

# Geometric signatures for each phenotype (for rule-based classification)
PHENOTYPE_SIGNATURES = {
    "fibrotic": {
        "geodesic_length": "low",
        "mean_curvature": "low",
        "peak_velocity": "low",
        "smoothness_index": "high",  # rigid = smooth but constrained
    },
    "weak": {
        "geodesic_length": "moderate",
        "mean_curvature": "moderate",
        "peak_velocity": "low",
        "smoothness_index": "moderate",
    },
    "compensatory": {
        "geodesic_length": "high",
        "mean_curvature": "high_spiky",
        "peak_velocity": "moderate",
        "smoothness_index": "low",
    },
    "neurogenic": {
        "geodesic_length": "variable",
        "mean_curvature": "variable",
        "peak_velocity": "variable",
        "smoothness_index": "variable",
    },
}


class PhenotypeClassifier:
    """
    Classify dysphagia phenotypes from manifold trajectory metrics.

    Parameters
    ----------
    method : str
        'random_forest', 'gradient_boosting', or 'svm'.
    """

    def __init__(self, method: str = "random_forest"):
        self.method = method
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names_ = None
        self.feature_importances_ = None

        if method == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            )
        elif method == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )
        elif method == "svm":
            self.model = SVC(kernel="rbf", probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")

    def extract_features(
        self,
        trajectories: List[SwallowingTrajectory],
        landmark_groups: Optional[Dict[str, List[int]]] = None,
    ) -> pd.DataFrame:
        """
        Extract manifold metric features from a list of trajectories.

        Returns a DataFrame with one row per trajectory.
        """
        rows = []
        for traj in trajectories:
            metrics = extract_all_metrics(traj, landmark_groups)
            metrics["subject_id"] = traj.subject_id
            metrics["condition"] = traj.condition
            rows.append(metrics)

        df = pd.DataFrame(rows)
        self.feature_names_ = [
            c for c in df.columns
            if c not in ("subject_id", "condition")
        ]
        return df

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the classifier on feature matrix X and labels y."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        if hasattr(self.model, "feature_importances_"):
            self.feature_importances_ = self.model.feature_importances_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict phenotype labels."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict phenotype probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, n_folds: int = 5
    ) -> Dict:
        """
        Perform stratified k-fold cross-validation.

        Returns dict with mean/std accuracy and per-fold scores.
        """
        X_scaled = self.scaler.fit_transform(X)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        scores = cross_val_score(
            self.model, X_scaled, y, cv=cv, scoring="accuracy"
        )

        return {
            "mean_accuracy": float(np.mean(scores)),
            "std_accuracy": float(np.std(scores)),
            "fold_scores": scores.tolist(),
        }

    def evaluate(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict:
        """
        Full evaluation: confusion matrix, classification report, AUC.
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        cm = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)

        # One-vs-rest AUC
        n_classes = len(np.unique(y))
        if n_classes > 2:
            auc = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")
        else:
            auc = roc_auc_score(y, y_proba[:, 1])

        return {
            "confusion_matrix": cm,
            "classification_report": report,
            "macro_auc": float(auc),
        }


def compute_inter_swallow_variability(
    trajectories: List[SwallowingTrajectory],
) -> float:
    """
    Compute within-subject trajectory variability.

    For neurogenic dysphagia detection: high variability indicates
    unstable control.

    Returns mean pairwise Euclidean distance between resampled trajectories.
    """
    if len(trajectories) < 2:
        return 0.0

    n_resample = 100
    resampled = [t.interpolate(n_resample).landmarks for t in trajectories]

    dists = []
    for i in range(len(resampled)):
        for j in range(i + 1, len(resampled)):
            d = np.sqrt(np.mean((resampled[i] - resampled[j]) ** 2))
            dists.append(d)

    return float(np.mean(dists))
