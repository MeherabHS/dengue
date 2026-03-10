"""
dashboard.py

Streamlit dashboard for the dengue forecasting system.

The dashboard calls the FastAPI backend (api_server.py) to fetch the
latest forecast and renders:
  - A summary metric card
  - A bar/interval chart showing the prediction with confidence bounds

Run with:
    streamlit run dashboard.py

The API base URL can be customized via the API_BASE_URL environment variable
(default: http://localhost:8000).
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
FORECAST_URL: str = f"{API_BASE_URL}/forecast"


def fetch_forecast() -> dict:
    """Call the FastAPI /forecast endpoint and return the JSON payload.

    Raises
    ------
    requests.HTTPError
        If the API returns a non-2xx status code.
    requests.ConnectionError
        If the API server is unreachable.
    """
    response = requests.get(FORECAST_URL, timeout=10)
    response.raise_for_status()
    return response.json()


def build_chart_data(forecast: dict) -> pd.DataFrame:
    """Convert the forecast dict into a tidy DataFrame suitable for plotting."""
    return pd.DataFrame(
        {
            "Label": ["Lower bound", "Predicted cases", "Upper bound"],
            "Cases": [
                forecast["lower_bound"],
                forecast["predicted_cases"],
                forecast["upper_bound"],
            ],
        }
    )


def main() -> None:
    st.set_page_config(
        page_title="Dengue Forecast Dashboard",
        page_icon="🦟",
        layout="centered",
    )

    st.title("🦟 Dengue Case Forecast Dashboard")
    st.caption(f"Data source: {FORECAST_URL}")

    # --- Fetch data --------------------------------------------------------
    with st.spinner("Fetching forecast from API…"):
        try:
            forecast = fetch_forecast()
        except requests.ConnectionError:
            st.error(
                "❌ Could not connect to the API server. "
                f"Make sure it is running at **{API_BASE_URL}**."
            )
            st.stop()
        except requests.HTTPError as exc:
            st.error(f"❌ API returned an error: {exc}")
            st.stop()

    # --- Summary metrics ---------------------------------------------------
    st.subheader("Next-Month Prediction")

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Cases", f"{forecast['predicted_cases']:.1f}")
    col2.metric("Lower Bound (95 %)", f"{forecast['lower_bound']:.1f}")
    col3.metric("Upper Bound (95 %)", f"{forecast['upper_bound']:.1f}")

    st.caption(f"Forecast generated at: {forecast['timestamp']} UTC")

    # --- Chart -------------------------------------------------------------
    st.subheader("Forecast with Confidence Interval")

    chart_df = build_chart_data(forecast)
    st.bar_chart(chart_df.set_index("Label"))

    # --- Raw data expander -------------------------------------------------
    with st.expander("Raw API response"):
        st.json(forecast)


if __name__ == "__main__":
    main()