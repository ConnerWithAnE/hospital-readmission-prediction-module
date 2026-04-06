"""
Inventory Demand Forecasting — Configuration
=============================================
Centralised settings for datasets, training, and model hyperparameters.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Dataset paths ───────────────────────────────────────────────────────
    data_dir: Path = Path(__file__).resolve().parent.parent.parent / "datasets" / "inventory"

    # Medicare Part D Prescriber Public Use File (annual CSVs)
    # Download: https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers
    medicare_dir: str = "medicare_part_d"

    # FDA State Drug Utilization Data (quarterly CSVs)
    # Download: https://www.medicaid.gov/medicaid/prescription-drugs/state-drug-utilization-data
    fda_dir: str = "fda_drug_utilization"

    @property
    def medicare_path(self) -> Path:
        return self.data_dir / self.medicare_dir

    @property
    def fda_path(self) -> Path:
        return self.data_dir / self.fda_dir

    # ── Output ──────────────────────────────────────────────────────────────
    output_dir: Path = Path(__file__).resolve().parent / "output"
    models_dir: Path = Path(__file__).resolve().parent / "saved_models"

    # ── Granularity options to compare ──────────────────────────────────────
    # "quarterly" = native FDA resolution; "monthly" = interpolated from quarterly
    granularities: list = field(default_factory=lambda: ["quarterly", "monthly"])

    # ── Feature engineering ─────────────────────────────────────────────────
    lag_periods: list = field(default_factory=lambda: [1, 2, 4])
    rolling_windows: list = field(default_factory=lambda: [2, 4])

    # ── Train / val / test split ──────────────────────────────────────────
    test_periods: int = 4     # for time-based split (multi-year data)
    val_periods: int = 2
    val_size: float = 0.15    # for group-based split (single-year data)
    test_size: float = 0.15

    # ── GBM hyperparameters ─────────────────────────────────────────────────
    gbm_n_estimators: int = 500
    gbm_max_depth: int = 6
    gbm_learning_rate: float = 0.05
    gbm_early_stopping: int = 30

    # ── LSTM hyperparameters ────────────────────────────────────────────────
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_seq_len: int = 8
    lstm_epochs: int = 100
    lstm_batch_size: int = 64
    lstm_lr: float = 1e-3
    lstm_patience: int = 15

    # ── Hybrid model ────────────────────────────────────────────────────────
    hybrid_gbm_n_estimators: int = 300
    hybrid_gbm_max_depth: int = 5

    # ── General ─────────────────────────────────────────────────────────────
    seed: int = 42
    device: str = "cpu"
    target_col: str = "units"