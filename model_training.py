# model_training.py

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

DATA_PATH = Path("dengue dataset.csv")
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "sarima_model.pkl"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"


class DengueModelTrainer:
    """
    This trainer isolates production training from exploratory analysis so the
    deployed API depends on one stable serialized artifact rather than notebook-style code.
    """

    def __init__(self, data_path: Path, model_path: Path, metadata_path: Path) -> None:
        self.data_path = data_path
        self.model_path = model_path
        self.metadata_path = metadata_path

    def load_data(self) -> pd.DataFrame:
        """
        Data loading is separated so schema validation can fail early before any
        model work begins, reducing debugging latency and preventing partial runs.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        df = pd.read_csv(self.data_path)

        required_columns = {"calendar_start_date", "dengue_total"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        return df

    def prepare_series(self, df: pd.DataFrame) -> pd.Series:
        """
        The production model only needs the approved target series. This keeps
        serving logic parsimonious and avoids coupling deployment to experimental features.
        """
        df = df.copy()

        df["calendar_start_date"] = pd.to_datetime(
            df["calendar_start_date"],
            errors="coerce",
        )
        df = df.dropna(subset=["calendar_start_date"]).sort_values("calendar_start_date")

        cutoff = df["calendar_start_date"].max() - pd.DateOffset(years=15)
        df = df[df["calendar_start_date"] >= cutoff].copy()

        df["cases"] = pd.to_numeric(df["dengue_total"], errors="coerce")
        df = df.dropna(subset=["cases"]).copy()
        df["log_cases"] = np.log1p(df["cases"])

        ts = (
            df.set_index("calendar_start_date")["log_cases"]
            .sort_index()
            .asfreq("MS")
        )

        if ts.isna().any():
            missing_dates = ts[ts.isna()].index.strftime("%Y-%m-%d").tolist()
            raise ValueError(
                "Time series has missing monthly periods after alignment. "
                f"Missing dates: {missing_dates[:10]}"
            )

        if len(ts) < 72:
            raise ValueError(
                "Insufficient monthly history for seasonal SARIMA. "
                f"Found {len(ts)} periods; need at least 72."
            )

        return ts

    def train_model(self, ts: pd.Series):
        """
        This method trains the final approved production model only. Research
        benchmarks belong in the analysis script, not here.
        """
        model = SARIMAX(
            ts,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        return fitted

    def save_artifacts(self, fitted_model, ts: pd.Series) -> None:
        """
        Serializing both the model and minimal metadata creates a reproducible
        inference boundary and simplifies operational troubleshooting.
        """
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.model_path, "wb") as fh:
            pickle.dump(fitted_model, fh)

        metadata = {
            "model_type": "SARIMA",
            "order": [1, 1, 1],
            "seasonal_order": [1, 1, 1, 12],
            "training_start": str(ts.index.min().date()),
            "training_end": str(ts.index.max().date()),
            "n_periods": int(len(ts)),
            "target": "log1p(dengue_total)",
        }

        with open(self.metadata_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        logging.info("Saved model to %s", self.model_path)
        logging.info("Saved metadata to %s", self.metadata_path)

    def run(self) -> None:
        df = self.load_data()
        ts = self.prepare_series(df)
        fitted = self.train_model(ts)
        self.save_artifacts(fitted, ts)


if __name__ == "__main__":
    trainer = DengueModelTrainer(DATA_PATH, MODEL_PATH, METADATA_PATH)
    trainer.run()