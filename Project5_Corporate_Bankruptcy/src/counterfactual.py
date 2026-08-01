"""Counterfactual financial risk simulation and ratio perturbation solver."""

import logging
from typing import Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CounterfactualSolver:
    """Solves minimal solvency ratio perturbation required to lower default probability below target threshold."""

    def __init__(self, target_threshold: float = 0.20, step_multiplier: float = 1.03, max_iterations: int = 30) -> None:
        self.target_threshold = target_threshold
        self.step_multiplier = step_multiplier
        self.max_iterations = max_iterations

    def solve(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, y_probs: np.ndarray
    ) -> dict[str, Any]:
        """Calculates counterfactual shift for a high-risk insolvent company.

        Args:
            model: Fitted pipeline model.
            X_test: Test features DataFrame.
            y_test: Test labels Series.
            y_probs: Predicted test default probabilities.

        Returns:
            Dictionary containing simulation metrics, feature adjustments, and percentage shift.
        """
        high_risk_mask = (y_probs > 0.75) & (y_test.values == 1)
        if not np.any(high_risk_mask):
            return {"status": "No high-risk candidate firm identified"}

        candidate_idx = int(np.where(high_risk_mask)[0][0])
        company_features = X_test.iloc[candidate_idx].copy()
        initial_prob = float(y_probs[candidate_idx])

        rf_classifier = model.named_steps["classifier"]
        importances = rf_classifier.feature_importances_
        top_3_idx = np.argsort(importances)[-3:]
        top_features = list(X_test.columns[top_3_idx])

        simulated_features = company_features.copy()
        current_prob = initial_prob
        iterations = 0

        while current_prob > self.target_threshold and iterations < self.max_iterations:
            simulated_features[top_features] *= self.step_multiplier
            sim_df = pd.DataFrame([simulated_features], columns=X_test.columns)
            current_prob = float(model.predict_proba(sim_df)[0, 1])
            iterations += 1

        percentage_shift = float(((self.step_multiplier ** iterations) - 1.0) * 100.0)

        logger.info("Counterfactual solved | Initial: %.2f%% -> Final: %.2f%% in %d steps", initial_prob * 100, current_prob * 100, iterations)
        return {
            "status": "Success",
            "candidate_index": candidate_idx,
            "initial_probability": initial_prob,
            "final_probability": current_prob,
            "targeted_features": top_features,
            "iterations": iterations,
            "percentage_shift": percentage_shift,
        }
