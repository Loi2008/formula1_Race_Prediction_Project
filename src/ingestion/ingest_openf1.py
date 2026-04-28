import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


BASE_URL = "https://api.openf1.org/v1"
RAW_DIR = Path("data/raw/openf1")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch(
    endpoint: str,
    params: dict[str, Any] | None = None,
    max_retries: int = 6
) -> list[dict[str, Any]]:
    """
    Fetch data from OpenF1 with:
    - retry for 429 rate limits
    - immediate skip for 404 not found
    - retry for transient request failures
    """
    url = f"{BASE_URL}/{endpoint}"

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params or {}, timeout=60)

            if response.status_code == 404:
                print(f"No data for endpoint={endpoint}, params={params}. Skipping.")
                return []

            if response.status_code == 429:
                wait_time = min(60, 5 * attempt)
                print(
                    f"Rate limit hit for endpoint={endpoint}, params={params}. "
                    f"Waiting {wait_time} seconds..."
                )
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            time.sleep(0.8)
            return response.json()

        except requests.exceptions.RequestException as exc:
            wait_time = min(60, 3 * attempt)
            print(
                f"Request failed for endpoint={endpoint}, params={params}: {exc}. "
                f"Retrying in {wait_time} seconds..."
            )
            time.sleep(wait_time)

    print(f"Failed after retries: endpoint={endpoint}, params={params}")
    return []


def safe_read_existing_csv(path: Path) -> pd.DataFrame | None:
    """Read an existing CSV if it exists."""
    if path.exists():
        return pd.read_csv(path)
    return None


def merge_and_save(rows: list[dict[str, Any]], filename: str, dedupe_cols: list[str]) -> None:
    """
    Append new rows to existing file and remove duplicates.
    """
    if not rows:
        print(f"No new rows to save for {filename}")
        return

    new_df = pd.DataFrame(rows)
    output_path = RAW_DIR / filename

    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=dedupe_cols)
    else:
        combined_df = new_df.drop_duplicates(subset=dedupe_cols)

    combined_df.to_csv(output_path, index=False)
    print(f"Saved {len(combined_df):,} total rows to {output_path}")


def get_existing_session_keys(filename: str) -> set[int]:
    """
    Return set of already-downloaded session keys from an existing CSV.
    """
    path = RAW_DIR / filename
    df = safe_read_existing_csv(path)

    if df is None or df.empty or "session_key" not in df.columns:
        return set()

    return set(pd.to_numeric(df["session_key"], errors="coerce").dropna().astype(int))


def save_core_tables(year: int) -> list[int]:
    """
    Fetch and save meetings + sessions for a given year.
    Returns all discovered session_keys.
    """
    print(f"Fetching OpenF1 meetings and sessions for {year}...")

    meetings = fetch("meetings", params={"year": year})
    sessions = fetch("sessions", params={"year": year})

    merge_and_save(meetings, "meetings.csv", ["meeting_key"])
    merge_and_save(sessions, "sessions.csv", ["session_key"])

    session_keys = sorted(
        {
            int(session["session_key"])
            for session in sessions
            if session.get("session_key") is not None
        }
    )
    return session_keys


def process_session(session_key: int) -> None:
    """
    Fetch and save all endpoint data for one session_key.
    """
    print(f"\nProcessing session_key={session_key} ...")

    completed_laps = get_existing_session_keys("laps.csv")
    completed_drivers = get_existing_session_keys("drivers.csv")
    completed_weather = get_existing_session_keys("weather.csv")
    completed_position = get_existing_session_keys("position.csv")

    # Laps
    if session_key in completed_laps:
        print(f"Skipping laps for session {session_key} (already downloaded).")
    else:
        laps = fetch("laps", params={"session_key": session_key})
        merge_and_save(
            laps,
            "laps.csv",
            ["session_key", "driver_number", "lap_number"]
        )

    # Drivers
    if session_key in completed_drivers:
        print(f"Skipping drivers for session {session_key} (already downloaded).")
    else:
        drivers = fetch("drivers", params={"session_key": session_key})
        merge_and_save(
            drivers,
            "drivers.csv",
            ["session_key", "driver_number"]
        )

    # Weather
    if session_key in completed_weather:
        print(f"Skipping weather for session {session_key} (already downloaded).")
    else:
        weather = fetch("weather", params={"session_key": session_key})
        merge_and_save(
            weather,
            "weather.csv",
            ["session_key", "date"]
        )

    # Position
    if session_key in completed_position:
        print(f"Skipping position for session {session_key} (already downloaded).")
    else:
        position = fetch("position", params={"session_key": session_key})
        merge_and_save(
            position,
            "position.csv",
            ["session_key", "driver_number", "date"]
        )

    time.sleep(1.5)


def main(year: int) -> None:
    session_keys = save_core_tables(year)

    for session_key in session_keys:
        process_session(session_key)

    print("\nOpenF1 ingestion completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    main(args.year)