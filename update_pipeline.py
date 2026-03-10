"""
update_pipeline.py

Automated forecast-refresh script for the dengue forecasting system.

What it does
------------
1. Calls generate_forecast() directly (no HTTP round-trip required).
2. Appends the result to a CSV log file (forecast_log.csv).
3. Prints a summary to stdout.
4. Can be scheduled via cron (or any scheduler) to run periodically, e.g.:

    # Run every day at 06:00
    0 6 * * * /usr/bin/python /path/to/update_pipeline.py

Usage
-----
    python update_pipeline.py [--log-file PATH]

Options
-------
--log-file PATH   Path for the CSV forecast log (default: forecast_log.csv).
--once            Run a single update and exit (default behaviour).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys

from model_forecast import ForecastResult, generate_forecast

DEFAULT_LOG_FILE = "forecast_log.csv"
CSV_FIELDNAMES = ["timestamp", "predicted_cases", "lower_bound", "upper_bound"]


def append_to_log(result: ForecastResult, log_file: str, timestamp: datetime.datetime) -> None:
    """Append a single forecast result row to the CSV log.

    The file is created with a header row if it does not already exist.

    Parameters
    ----------
    result    : ForecastResult returned by generate_forecast()
    log_file  : Path to the CSV file
    timestamp : UTC datetime of this prediction run
    """
    file_exists = os.path.isfile(log_file)

    with open(log_file, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": timestamp.isoformat(),
                "predicted_cases": result.predicted_cases,
                "lower_bound": result.lower_bound,
                "upper_bound": result.upper_bound,
            }
        )


def run_update(log_file: str) -> None:
    """Execute one forecast update cycle.

    Steps
    -----
    1. Call generate_forecast() to obtain fresh predictions.
    2. Record the result in the CSV log.
    3. Print a human-readable summary.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc)

    print(f"[{timestamp.isoformat()}] Running forecast update…")

    try:
        result = generate_forecast()
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("ERROR: Forecast generation failed (see traceback above).", file=sys.stderr)
        sys.exit(1)

    append_to_log(result, log_file, timestamp)

    print(
        f"  Predicted cases : {result.predicted_cases}\n"
        f"  Lower bound     : {result.lower_bound}\n"
        f"  Upper bound     : {result.upper_bound}\n"
        f"  Logged to       : {log_file}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update the dengue forecast and append the result to a CSV log."
    )
    parser.add_argument(
        "--log-file",
        default=DEFAULT_LOG_FILE,
        help=f"Path to the CSV forecast log (default: {DEFAULT_LOG_FILE})",
    )
    args = parser.parse_args()

    run_update(log_file=args.log_file)


if __name__ == "__main__":
    main()