from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

MODEL_PATH = Path("artifacts/sarima_model.pkl")


@dataclass
class ForecastResult:
    """
    This response contract keeps the API and update pipeline aligned on one
    strongly typed forecast structure instead of passing loose dictionaries.
    """
    predicted_cases: float
    lower_bound: float
    upper_bound: float


class ForecastService:
    """
    This service isolates artifact loading and forecast generation so both the
    API and scheduled pipeline use the same deterministic inference path.
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._model = None

    def load_model(self):
        """
        Lazy loading avoids repeated disk I/O on every request while still
        failing fast if the trained artifact is missing.
        """
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model artifact not found: {self.model_path}. "
                    "Run model_training.py first."
                )

            with open(self.model_path, "rb") as fh:
                self._model = pickle.load(fh)

        return self._model

    @staticmethod
    def _invert_log1p(value: float) -> float:
        """
        The SARIMA model predicts log1p(cases), so the result is converted back
        to case-count scale for dashboard readability and operational usefulness.
        """
        return max(0.0, float(math.expm1(value)))

    def generate_forecast(self) -> ForecastResult:
        """
        Produces a one-step-ahead forecast from the trained SARIMA artifact and
        returns both the point estimate and 95% interval bounds.
        """
        model = self.load_model()

        forecast_obj = model.get_forecast(steps=1)
        pred_log = float(forecast_obj.predicted_mean.iloc[0])

        conf_int = forecast_obj.conf_int(alpha=0.05)
        lower_log = float(conf_int.iloc[0, 0])
        upper_log = float(conf_int.iloc[0, 1])

        return ForecastResult(
            predicted_cases=self._invert_log1p(pred_log),
            lower_bound=self._invert_log1p(lower_log),
            upper_bound=self._invert_log1p(upper_log),
        )


_service = ForecastService(MODEL_PATH)


def generate_forecast() -> ForecastResult:
    """
    This wrapper preserves the import contract expected by api_server.py and
    update_pipeline.py while delegating to the shared service instance.
    """
    return _service.generate_forecast()