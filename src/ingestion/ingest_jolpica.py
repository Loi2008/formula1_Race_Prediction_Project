import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


BASE_URL = "https://api.jolpi.ca/ergast/f1"
RAW_DIR = Path("data/raw/jolpica")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_json(url: str, sleep_seconds: float = 0.8, max_retries: int = 6) -> dict[str, Any]:
    """
    Fetch JSON with retry handling for rate limits and transient failures.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=60)

            if response.status_code == 429:
                wait_time = min(60, 5 * attempt)
                print(f"Rate limit hit for {url}. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            time.sleep(sleep_seconds)
            return response.json()

        except requests.exceptions.RequestException as exc:
            wait_time = min(60, 3 * attempt)
            print(f"Request failed for {url}: {exc}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    raise RuntimeError(f"Failed after {max_retries} retries: {url}")


def safe_read_existing_csv(path: Path) -> pd.DataFrame | None:
    """
    Read an existing CSV if present, otherwise return None.
    """
    if path.exists():
        return pd.read_csv(path)
    return None


def merge_and_save(rows: list[dict[str, Any]], filename: str, dedupe_cols: list[str]) -> None:
    """
    Append new rows to existing file and remove duplicates.
    """
    new_df = pd.DataFrame(rows)
    out_path = RAW_DIR / filename

    if out_path.exists():
        existing_df = pd.read_csv(out_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=dedupe_cols)
    else:
        combined_df = new_df.drop_duplicates(subset=dedupe_cols)

    combined_df.to_csv(out_path, index=False)
    print(f"Saved {len(combined_df):,} total rows to {out_path}")


def extract_races(season: int) -> list[dict[str, Any]]:
    url = f"{BASE_URL}/{season}/races/"
    payload = fetch_json(url)
    races = payload["MRData"]["RaceTable"]["Races"]

    out = []
    for race in races:
        out.append(
            {
                "season": int(race["season"]),
                "round": int(race["round"]),
                "race_name": race.get("raceName"),
                "race_date": race.get("date"),
                "race_time": race.get("time"),
                "circuit_id": race["Circuit"].get("circuitId"),
                "circuit_name": race["Circuit"].get("circuitName"),
                "locality": race["Circuit"]["Location"].get("locality"),
                "country": race["Circuit"]["Location"].get("country"),
                "lat": race["Circuit"]["Location"].get("lat"),
                "long": race["Circuit"]["Location"].get("long"),
            }
        )
    return out


def extract_results(season: int) -> list[dict[str, Any]]:
    url = f"{BASE_URL}/{season}/results/"
    payload = fetch_json(url)
    races = payload["MRData"]["RaceTable"]["Races"]

    out = []
    for race in races:
        season_val = int(race["season"])
        round_val = int(race["round"])
        race_name = race.get("raceName")

        for result in race.get("Results", []):
            driver = result["Driver"]
            constructor = result["Constructor"]
            fastest_lap = result.get("FastestLap", {})

            out.append(
                {
                    "season": season_val,
                    "round": round_val,
                    "race_name": race_name,
                    "driver_id": driver.get("driverId"),
                    "driver_code": driver.get("code"),
                    "driver_number": driver.get("permanentNumber"),
                    "given_name": driver.get("givenName"),
                    "family_name": driver.get("familyName"),
                    "date_of_birth": driver.get("dateOfBirth"),
                    "nationality": driver.get("nationality"),
                    "constructor_id": constructor.get("constructorId"),
                    "constructor_name": constructor.get("name"),
                    "constructor_nationality": constructor.get("nationality"),
                    "grid": result.get("grid"),
                    "position": result.get("position"),
                    "position_text": result.get("positionText"),
                    "points": result.get("points"),
                    "laps": result.get("laps"),
                    "status": result.get("status"),
                    "finish_time_ms": result.get("Time", {}).get("millis"),
                    "finish_time_text": result.get("Time", {}).get("time"),
                    "fastest_lap_rank": fastest_lap.get("rank"),
                    "fastest_lap_number": fastest_lap.get("lap"),
                    "fastest_lap_time": fastest_lap.get("Time", {}).get("time"),
                    "fastest_lap_avg_speed": fastest_lap.get("AverageSpeed", {}).get("speed"),
                    "fastest_lap_speed_units": fastest_lap.get("AverageSpeed", {}).get("units"),
                }
            )
    return out


def extract_qualifying(season: int) -> list[dict[str, Any]]:
    url = f"{BASE_URL}/{season}/qualifying/"
    payload = fetch_json(url)
    races = payload["MRData"]["RaceTable"]["Races"]

    out = []
    for race in races:
        season_val = int(race["season"])
        round_val = int(race["round"])
        race_name = race.get("raceName")

        for result in race.get("QualifyingResults", []):
            driver = result["Driver"]
            constructor = result["Constructor"]

            out.append(
                {
                    "season": season_val,
                    "round": round_val,
                    "race_name": race_name,
                    "position": result.get("position"),
                    "driver_id": driver.get("driverId"),
                    "driver_code": driver.get("code"),
                    "driver_number": driver.get("permanentNumber"),
                    "given_name": driver.get("givenName"),
                    "family_name": driver.get("familyName"),
                    "constructor_id": constructor.get("constructorId"),
                    "constructor_name": constructor.get("name"),
                    "q1": result.get("Q1"),
                    "q2": result.get("Q2"),
                    "q3": result.get("Q3"),
                }
            )
    return out


def extract_laps_for_race(season: int, round_number: int) -> list[dict[str, Any]]:
    url = f"{BASE_URL}/{season}/{round_number}/laps/"
    payload = fetch_json(url, sleep_seconds=1.0)
    races = payload["MRData"]["RaceTable"]["Races"]

    out = []
    for race in races:
        season_val = int(race["season"])
        round_val = int(race["round"])
        race_name = race.get("raceName")

        for lap in race.get("Laps", []):
            lap_number = int(lap["number"])
            for timing in lap.get("Timings", []):
                out.append(
                    {
                        "season": season_val,
                        "round": round_val,
                        "race_name": race_name,
                        "lap_number": lap_number,
                        "driver_id": timing.get("driverId"),
                        "position": timing.get("position"),
                        "lap_time": timing.get("time"),
                    }
                )
    return out


def get_completed_race_keys(filename: str) -> set[tuple[int, int]]:
    """
    Return set of (season, round) already present in an existing raw table.
    """
    path = RAW_DIR / filename
    df = safe_read_existing_csv(path)
    if df is None or df.empty:
        return set()

    if "season" not in df.columns or "round" not in df.columns:
        return set()

    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["round"] = pd.to_numeric(df["round"], errors="coerce")

    completed = {
        (int(season), int(round_number))
        for season, round_number in zip(df["season"], df["round"])
        if pd.notna(season) and pd.notna(round_number)
    }
    return completed


def ingest_races_results_qualifying(start_season: int, end_season: int) -> None:
    completed_races = get_completed_race_keys("races.csv")
    completed_results = get_completed_race_keys("results.csv")
    completed_qualifying = get_completed_race_keys("qualifying.csv")

    all_races = []
    all_results = []
    all_qualifying = []

    for season in range(start_season, end_season + 1):
        print(f"Fetching season {season} core tables...")

        races = extract_races(season)
        results = extract_results(season)
        qualifying = extract_qualifying(season)

        for row in races:
            key = (row["season"], row["round"])
            if key not in completed_races:
                all_races.append(row)

        for row in results:
            key = (row["season"], row["round"])
            if key not in completed_results:
                all_results.append(row)

        for row in qualifying:
            key = (row["season"], row["round"])
            if key not in completed_qualifying:
                all_qualifying.append(row)

    if all_races:
        merge_and_save(all_races, "races.csv", ["season", "round"])
    else:
        print("No new race rows to save.")

    if all_results:
        merge_and_save(all_results, "results.csv", ["season", "round", "driver_id"])
    else:
        print("No new result rows to save.")

    if all_qualifying:
        merge_and_save(all_qualifying, "qualifying.csv", ["season", "round", "driver_id"])
    else:
        print("No new qualifying rows to save.")


def ingest_laps(start_season: int, end_season: int) -> None:
    completed_lap_races = get_completed_race_keys("lap_times.csv")
    races_path = RAW_DIR / "races.csv"
    races_df = safe_read_existing_csv(races_path)

    if races_df is None or races_df.empty:
        raise RuntimeError(
            "races.csv not found. Run core ingestion first without --include-laps."
        )

    races_df["season"] = pd.to_numeric(races_df["season"], errors="coerce")
    races_df["round"] = pd.to_numeric(races_df["round"], errors="coerce")
    races_df = races_df.dropna(subset=["season", "round"]).copy()
    races_df["season"] = races_df["season"].astype(int)
    races_df["round"] = races_df["round"].astype(int)

    filtered_races = races_df[
        (races_df["season"] >= start_season) & (races_df["season"] <= end_season)
    ][["season", "round"]].drop_duplicates()

    all_laps = []

    for _, race in filtered_races.sort_values(["season", "round"]).iterrows():
        season = int(race["season"])
        round_number = int(race["round"])
        key = (season, round_number)

        if key in completed_lap_races:
            print(f"Skipping laps for {season} round {round_number} (already downloaded).")
            continue

        print(f"Fetching laps for {season} round {round_number}...")
        lap_rows = extract_laps_for_race(season, round_number)
        all_laps.extend(lap_rows)

        if all_laps:
            merge_and_save(
                all_laps,
                "lap_times.csv",
                ["season", "round", "driver_id", "lap_number"]
            )
            all_laps = []

        time.sleep(1.5)

    if not all_laps:
        print("Lap ingestion complete.")


def main(start_season: int, end_season: int, include_laps: bool, laps_only: bool) -> None:
    if not laps_only:
        ingest_races_results_qualifying(start_season, end_season)

    if include_laps or laps_only:
        ingest_laps(start_season, end_season)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, required=True)
    parser.add_argument("--end-season", type=int, required=True)
    parser.add_argument("--include-laps", action="store_true", help="Also fetch lap times.")
    parser.add_argument("--laps-only", action="store_true", help="Only fetch missing lap times.")
    args = parser.parse_args()

    main(
        start_season=args.start_season,
        end_season=args.end_season,
        include_laps=args.include_laps,
        laps_only=args.laps_only,
    )