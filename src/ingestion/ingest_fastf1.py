import argparse
import time
from pathlib import Path

import fastf1
import pandas as pd


RAW_DIR = Path("data/raw/fastf1")
CACHE_DIR = Path("data/raw/fastf1_cache")

RAW_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Enable FastF1 cache
fastf1.Cache.enable_cache(str(CACHE_DIR))


def safe_read_existing_csv(path: Path) -> pd.DataFrame | None:
    """
    Read an existing CSV if it exists.
    """
    if path.exists():
        return pd.read_csv(path)
    return None


def merge_and_save(df: pd.DataFrame, filename: str, dedupe_cols: list[str]) -> None:
    """
    Append new rows to existing file and remove duplicates.
    """
    if df is None or df.empty:
        print(f"No new rows to save for {filename}")
        return

    output_path = RAW_DIR / filename

    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=dedupe_cols)
    else:
        combined_df = df.drop_duplicates(subset=dedupe_cols)

    combined_df.to_csv(output_path, index=False)
    print(f"Saved {len(combined_df):,} total rows to {output_path}")


def get_completed_rounds(filename: str, year: int) -> set[int]:
    """
    Return set of rounds already processed for a given year.
    """
    path = RAW_DIR / filename
    df = safe_read_existing_csv(path)

    if df is None or df.empty:
        return set()

    if "season" not in df.columns or "round" not in df.columns:
        return set()

    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["round"] = pd.to_numeric(df["round"], errors="coerce")

    completed = df[df["season"] == year]["round"].dropna().astype(int).unique()
    return set(completed)


def clean_laps_df(laps: pd.DataFrame, season: int, round_number: int, event_name: str) -> pd.DataFrame:
    """
    Clean and standardize FastF1 laps output.
    """
    laps = laps.copy()
    laps["season"] = season
    laps["round"] = round_number
    laps["event_name"] = event_name

    # Convert time-like columns to strings for CSV compatibility
    timedelta_cols = ["Time", "LapTime", "PitOutTime", "PitInTime", "Sector1Time", "Sector2Time", "Sector3Time"]
    for col in timedelta_cols:
        if col in laps.columns:
            laps[col] = laps[col].astype(str)

    return laps


def clean_weather_df(weather: pd.DataFrame, season: int, round_number: int, event_name: str) -> pd.DataFrame:
    """
    Clean and standardize FastF1 weather output.
    """
    weather = weather.copy()
    weather["season"] = season
    weather["round"] = round_number
    weather["event_name"] = event_name
    return weather


def clean_results_df(results: pd.DataFrame, season: int, round_number: int, event_name: str) -> pd.DataFrame:
    """
    Clean and standardize FastF1 results output.
    """
    results = results.copy()
    results["season"] = season
    results["round"] = round_number
    results["event_name"] = event_name
    return results


def process_session(year: int, round_number: int, event_name: str, session_code: str) -> None:
    """
    Load one FastF1 session safely and save outputs incrementally.
    """
    try:
        print(f"\nLoading {year} round {round_number} - {event_name} ({session_code})...")

        session = fastf1.get_session(year, round_number, session_code)
        session.load()

        # Laps
        try:
            laps = session.laps.copy()
            laps = clean_laps_df(laps, year, round_number, event_name)
            merge_and_save(
                laps,
                f"laps_{year}_{session_code}.csv",
                ["season", "round", "Driver", "LapNumber"]
            )
        except Exception as e:
            print(f"Could not process laps for {event_name}: {e}")

        # Weather
        try:
            weather = session.weather_data.copy()
            weather = clean_weather_df(weather, year, round_number, event_name)
            merge_and_save(
                weather,
                f"weather_{year}_{session_code}.csv",
                ["season", "round", "Time"]
            )
        except Exception as e:
            print(f"Could not process weather for {event_name}: {e}")

        # Results
        try:
            results = session.results.copy()
            results = clean_results_df(results, year, round_number, event_name)
            merge_and_save(
                results,
                f"results_{year}_{session_code}.csv",
                ["season", "round", "Abbreviation"]
            )
        except Exception as e:
            print(f"Could not process results for {event_name}: {e}")

        time.sleep(1.5)

    except Exception as e:
        print(f"Skipping {year} round {round_number} - {event_name}: {e}")


def main(year: int, session_code: str) -> None:
    print(f"Fetching FastF1 schedule for {year}...")

    schedule = fastf1.get_event_schedule(year)
    schedule = schedule[["RoundNumber", "EventName"]].drop_duplicates()

    completed_lap_rounds = get_completed_rounds(f"laps_{year}_{session_code}.csv", year)

    for _, row in schedule.iterrows():
        round_number = int(row["RoundNumber"])
        event_name = row["EventName"]

        if round_number in completed_lap_rounds:
            print(f"Skipping round {round_number} ({event_name}) - already processed.")
            continue

        process_session(year, round_number, event_name, session_code)

    print("\nFastF1 ingestion completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--session",
        type=str,
        default="R",
        help="Session code: R, Q, FP1, FP2, FP3, S, SQ"
    )
    args = parser.parse_args()

    main(args.year, args.session)