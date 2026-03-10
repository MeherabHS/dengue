"""
api_server.py

FastAPI backend for the dengue forecasting system.

Endpoints
---------
GET /forecast
    Returns the dengue-case forecast produced by model_forecast.generate_forecast().

Run with:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_forecast import generate_forecast

app = FastAPI(
    title="Dengue Forecast API",
    description="Returns next-month dengue case predictions from the SARIMA model.",
    version="1.0.0",
)


class ForecastResponse(BaseModel):
    """Schema returned by GET /forecast."""

    predicted_cases: float
    lower_bound: float
    upper_bound: float
    timestamp: datetime.datetime


@app.get("/forecast", response_model=ForecastResponse, summary="Get next-month dengue forecast")
def get_forecast() -> ForecastResponse:
    """Load the forecasting model and return next-month predictions.

    Returns
    -------
    ForecastResponse
        JSON object containing:
        - **predicted_cases** – point-forecast for the next calendar month
        - **lower_bound**     – lower edge of the 95 % confidence interval
        - **upper_bound**     – upper edge of the 95 % confidence interval
        - **timestamp**       – UTC datetime when the prediction was generated
    """
    try:
        result = generate_forecast()
    except Exception as exc:
        # Log the full exception server-side; return a generic message to the client.
        import logging

        logging.exception("Forecast generation failed")
        raise HTTPException(status_code=500, detail="Forecast generation failed.") from exc

    return ForecastResponse(
        predicted_cases=result.predicted_cases,
        lower_bound=result.lower_bound,
        upper_bound=result.upper_bound,
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )


@app.get("/health", summary="Health check")
def health() -> dict:
    """Simple liveness probe used by monitoring tools and the update pipeline."""
    return {"status": "ok"}
 