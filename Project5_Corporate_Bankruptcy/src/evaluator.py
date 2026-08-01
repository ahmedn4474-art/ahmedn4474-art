"""Machine Learning modeling, cross-validation, and cost matrix loss optimization module."""

import logging
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class InsolvencyEvaluator:
    """Manages pipeline fitting, cross-validation evaluation, and monetary cost optimization."""

    def __init__(self, random_seed: int = 42, n_splits: int = 5) -> None:
        self.random_seed = random_seed
        self.n_splits = n_splits
        self.pipeline: Any = self._build_pipeline()

    def _build_pipeline(self) -> Any:
        """Constructs an imbalance-aware pipeline preventing CV data leakage."""
        if IMBLEARN_AVAILABLE:
            return ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=self.random_seed)),
                ("classifier", RandomForestClassifier(
                    n_estimators=150, max_depth=12, random_state=self.random_seed, n_jobs=-1
                )),
            ])
        return ImbPipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=150, max_depth=12, class_weight="balanced", random_state=self.random_seed, n_jobs=-1
            )),
        ])

    def run_cross_validation(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
        """Executes 5-Fold Stratified Cross-Validation strictly inside pipeline folds.

        Args:
            X_train: Training feature matrix.
            y_train: Training target labels.

        Returns:
            Dictionary containing mean CV metrics and standard deviations.
        """
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
        scores = cross_validate(
            self.pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=["roc_auc", "average_precision", "f1"],
            n_jobs=-1,
        )
        metrics = {
            "mean_roc_auc": float(np.mean(scores["test_roc_auc"])),
            "std_roc_auc": float(np.std(scores["test_roc_auc"])),
            "mean_pr_auc": float(np.mean(scores["test_average_precision"])),
            "std_pr_auc": float(np.std(scores["test_average_precision"])),
            "mean_f1": float(np.mean(scores["test_f1"])),
        }
        logger.info("CV Mean ROC-AUC: %.4f | Mean PR-AUC: %.4f", metrics["mean_roc_auc"], metrics["mean_pr_auc"])
        return metrics

    def optimize_cost_matrix(
        self, y_true: pd.Series, y_probs: np.ndarray, cost_fn: float, cost_fp: float
    ) -> tuple[float, float, float]:
        """Calculates decision probability threshold minimizing total financial risk loss.

        Args:
            y_true: True out-of-sample binary labels.
            y_probs: Predicted positive class probabilities.
            cost_fn: Monetary cost of a False Negative (missed default).
            cost_fp: Monetary cost of a False Positive (audit alarm).

        Returns:
            Tuple of (Optimal threshold, Minimum financial loss, Default 0.50 threshold loss).
        """
        thresholds = np.linspace(0.01, 0.99, 99)
        costs: list[float] = []

        for th in thresholds:
            preds = (y_probs >= th).astype(int)
            cm = confusion_matrix(y_true, preds)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            costs.append(float((fn * cost_fn) + (fp * cost_fp)))

        best_idx = int(np.argmin(costs))
        best_threshold = float(thresholds[best_idx])
        min_cost = costs[best_idx]

        def_preds = (y_probs >= 0.50).astype(int)
        cm_def = confusion_matrix(y_true, def_preds)
        tn, fp, fn, tp = cm_def.ravel()
        default_cost = float((fn * cost_fn) + (fp * cost_fp))

        logger.info("Optimal threshold: %.2f | Min loss: $%.2f | Capital saved: $%.2f", best_threshold, min_cost, default_cost - min_cost)
        return best_threshold, min_cost, default_cost
