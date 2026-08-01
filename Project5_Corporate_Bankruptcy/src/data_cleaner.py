"""Data ingestion, auditing, and cleaning module for corporate balance sheet data."""

import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """Handles dataset loading, null auditing, infinite value replacement, and target split."""

    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path

    def load_and_clean(self) -> tuple[pd.DataFrame, pd.Series, str]:
        """Loads data, strips column whitespace, removes duplicates and non-finite values.

        Returns:
            Tuple of (Feature DataFrame X, Target Series y, Target column name).
        """
        logger.info("Loading raw dataset from %s", self.dataset_path)
        raw_df = pd.read_csv(self.dataset_path)
        raw_df.columns = [col.strip() for col in raw_df.columns]

        logger.info("Raw shape: %s | Missing values: %d", raw_df.shape, raw_df.isnull().sum().sum())
        
        # Deduplication and Numerical Stability
        cleaned_df = raw_df.drop_duplicates().copy()
        cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan).dropna()
        
        target_col: str = str(cleaned_df.columns[0])
        X: pd.DataFrame = cleaned_df.drop(columns=[target_col])
        y: pd.Series = cleaned_df[target_col].astype(int)

        logger.info("Cleaned shape: %s | Solvent: %d | Insolvent: %d", X.shape, (y == 0).sum(), (y == 1).sum())
        return X, y, target_col
