"""Pytest test suite for Corporate Bankruptcy Prediction modules."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PipelineConfig
from src.evaluator import InsolvencyEvaluator
from src.counterfactual import CounterfactualSolver


@pytest.fixture
def dummy_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generates synthetic balance sheet data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"ratio_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]))
    return X, y


def test_pipeline_config() -> None:
    """Tests PipelineConfig initialization and properties."""
    config = PipelineConfig()
    assert config.test_size == 0.20
    assert config.n_splits == 5
    assert config.cost_false_negative == 100000.0


def test_evaluator_cross_validation(dummy_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Tests InsolvencyEvaluator 5-fold cross-validation metrics."""
    X, y = dummy_data
    evaluator = InsolvencyEvaluator(random_seed=42, n_splits=3)
    metrics = evaluator.run_cross_validation(X, y)

    assert "mean_roc_auc" in metrics
    assert "mean_pr_auc" in metrics
    assert 0.0 <= metrics["mean_roc_auc"] <= 1.0


def test_cost_matrix_optimization(dummy_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Tests financial cost loss threshold optimization."""
    X, y = dummy_data
    evaluator = InsolvencyEvaluator(random_seed=42, n_splits=2)
    evaluator.pipeline.fit(X, y)
    probs = evaluator.pipeline.predict_proba(X)[:, 1]

    best_th, min_cost, def_cost = evaluator.optimize_cost_matrix(
        y, probs, cost_fn=100000.0, cost_fp=5000.0
    )
    assert 0.01 <= best_th <= 0.99
    assert min_cost <= def_cost
