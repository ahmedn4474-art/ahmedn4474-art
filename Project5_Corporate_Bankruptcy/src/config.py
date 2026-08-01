"""Configuration dataclass for Corporate Bankruptcy Prediction & Risk Modeling."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Holds configuration parameters, file paths, and cost loss weights."""

    data_dir: Path = field(
        default_factory=lambda: Path(r"D:\download\protfolio\archive (4)")
    )
    dataset_filename: str = "data.csv"
    random_seed: int = 42
    test_size: float = 0.20
    n_splits: int = 5
    target_threshold: float = 0.20
    
    # Financial Cost Matrix Weights ($)
    cost_false_negative: float = 100000.0  # Missed bankruptcy write-off
    cost_false_positive: float = 5000.0   # Precautionary audit field cost

    @property
    def dataset_path(self) -> Path:
        """Returns candidate dataset file path."""
        candidates = [
            self.data_dir / self.dataset_filename,
            Path(__file__).resolve().parents[1] / "data.csv",
            Path(__file__).resolve().parents[2] / "archive (4)" / self.dataset_filename,
            Path("data.csv"),
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"Dataset '{self.dataset_filename}' not found in candidate paths."
        )
