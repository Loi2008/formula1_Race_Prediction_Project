import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROCESSED_DIR = Path("data/processed")


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    print(f"Loading: {path}")
    return pd.read_csv(path)


def load_processed_tables():
    results = safe_read_csv(PROCESSED_DIR / "results.csv")
    races = safe_read_csv(PROCESSED_DIR / "races.csv")
    qualifying = safe_read_csv(PROCESSED_DIR / "qualifying.csv")
    model_table = safe_read_csv(PROCESSED_DIR / "model_table.csv")
    openf1_meetings = safe_read_csv(PROCESSED_DIR / "openf1_meetings.csv")

    results["season"] = pd.to_numeric(results["season"], errors="coerce").astype("Int64")
    results["round"] = pd.to_numeric(results["round"], errors="coerce").astype("Int64")
    results["finish_position"] = pd.to_numeric(results["finish_position"], errors="coerce")
    results["grid"] = pd.to_numeric(results["grid"], errors="coerce")
    results["points"] = pd.to_numeric(results["points"], errors="coerce")

    races["season"] = pd.to_numeric(races["season"], errors="coerce").astype("Int64")
    races["round"] = pd.to_numeric(races["round"], errors="coerce").astype("Int64")
    races["race_date"] = pd.to_datetime(races["race_date"], errors="coerce")

    qualifying["season"] = pd.to_numeric(qualifying["season"], errors="coerce").astype("Int64")
    qualifying["round"] = pd.to_numeric(qualifying["round"], errors="coerce").astype("Int64")
    if "quali_position" in qualifying.columns:
        qualifying["quali_position"] = pd.to_numeric(qualifying["quali_position"], errors="coerce")

    model_table["season"] = pd.to_numeric(model_table["season"], errors="coerce").astype("Int64")
    model_table["round"] = pd.to_numeric(model_table["round"], errors="coerce").astype("Int64")

    if "year" in openf1_meetings.columns:
        openf1_meetings["year"] = pd.to_numeric(openf1_meetings["year"], errors="coerce").astype("Int64")
    if "meeting_key" in openf1_meetings.columns:
        openf1_meetings["meeting_key"] = pd.to_numeric(openf1_meetings["meeting_key"], errors="coerce")
    if "date_start" in openf1_meetings.columns:
        openf1_meetings["date_start"] = pd.to_datetime(openf1_meetings["date_start"], errors="coerce")

    return results, races, qualifying, model_table, openf1_meetings


def get_target_race(
    races: pd.DataFrame,
    openf1_meetings: pd.DataFrame,
    season: int,
    round_number: int
) -> dict:
    """
    First try historical races.csv.
    If not found, fall back to OpenF1 meetings for the selected year,
    ordered by date_start and mapped to round_number.
    """
    race = races[(races["season"] == season) & (races["round"] == round_number)].copy()

    if not race.empty:
        row = race.iloc[0]
        return {
            "season": season,
            "round": round_number,
            "race_name": row.get("race_name", f"Round {round_number}"),
            "race_date": row.get("race_date"),
            "circuit_id": row.get("circuit_id", "Unknown"),
            "country": row.get("country", "Unknown"),
            "source": "races",
        }

    meetings_year = openf1_meetings[openf1_meetings["year"] == season].copy()

    if meetings_year.empty:
        available = (
            races[races["season"] == season][["round", "race_name"]]
            .drop_duplicates()
            .sort_values("round")
        )
        raise ValueError(
            f"No race found for season={season}, round={round_number}. "
            f"No OpenF1 meetings found either. Available historical races:\n"
            f"{available.to_string(index=False)}"
        )

    meetings_year = meetings_year.sort_values("date_start").reset_index(drop=True)
    meetings_year["derived_round"] = meetings_year.index + 1

    meeting = meetings_year[meetings_year["derived_round"] == round_number].copy()

    if meeting.empty:
        available_rounds = meetings_year[["derived_round", "meeting_name"]].copy()
        raise ValueError(
            f"No race found for season={season}, round={round_number}. "
            f"OpenF1 available derived rounds:\n{available_rounds.to_string(index=False)}"
        )

    row = meeting.iloc[0]

    return {
        "season": season,
        "round": round_number,
        "race_name": row.get("meeting_name", f"Round {round_number}"),
        "race_date": row.get("date_start"),
        "circuit_id": row.get("circuit_short_name", "Unknown"),
        "country": row.get("country_name", "Unknown"),
        "source": "openf1_meetings",
    }


def get_historical_cutoff(df: pd.DataFrame, season: int, round_number: int) -> pd.DataFrame:
    return df[
        ((df["season"] < season)) |
        ((df["season"] == season) & (df["round"] < round_number))
    ].copy()


def infer_expected_lineup(results_hist: pd.DataFrame) -> pd.DataFrame:
    latest = (
        results_hist[["season", "round"]]
        .drop_duplicates()
        .sort_values(["season", "round"])
    )

    if latest.empty:
        raise ValueError("No historical results available to infer expected lineup.")

    latest_row = latest.iloc[-1]
    latest_season = int(latest_row["season"])
    latest_round = int(latest_row["round"])

    lineup = results_hist[
        (results_hist["season"] == latest_season) &
        (results_hist["round"] == latest_round)
    ].copy()

    lineup = lineup[
        [
            "driver_id",
            "driver_code",
            "given_name",
            "family_name",
            "constructor_id",
            "constructor_name",
            "nationality",
        ]
    ].drop_duplicates(subset=["driver_id"])

    return lineup.reset_index(drop=True)


def compute_driver_history(results_hist: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        results_hist.groupby("driver_id", as_index=False)
        .agg(
            avg_finish_position_prior=("finish_position", "mean"),
            finish_position_std_prior=("finish_position", "std"),
            avg_grid_prior=("grid", "mean"),
            avg_points_prior=("points", "mean"),
            last_points=("points", "last"),
            recent_form_last_3=("finish_position", lambda s: s.tail(3).mean()),
            recent_points_last_3=("points", lambda s: s.tail(3).mean()),
        )
    )
    grouped["driver_consistency_score"] = grouped["finish_position_std_prior"]
    return grouped


def compute_dnf_rate(results_hist: pd.DataFrame) -> pd.DataFrame:
    df = results_hist.copy()

    classified_finish = df["finish_position"].notna()
    completed_status = df["status"].fillna("").str.lower().isin(
        ["finished", "+1 lap", "+2 laps", "+3 laps"]
    )
    df["is_dnf"] = (~classified_finish & ~completed_status).astype(int)

    return (
        df.groupby("driver_id", as_index=False)
        .agg(dnf_rate_prior=("is_dnf", "mean"))
    )


def compute_constructor_history(results_hist: pd.DataFrame) -> pd.DataFrame:
    return (
        results_hist.groupby("constructor_id", as_index=False)
        .agg(
            team_points_index_prior=("points", "mean"),
            team_finish_position_prior=("finish_position", "mean"),
        )
    )


def compute_track_history(results_hist: pd.DataFrame, target_circuit_id: str) -> pd.DataFrame:
    track_hist = results_hist[results_hist["circuit_id"] == target_circuit_id].copy()

    if track_hist.empty:
        return pd.DataFrame(columns=["driver_id", "track_avg_finish_prior", "track_avg_points_prior"])

    return (
        track_hist.groupby("driver_id", as_index=False)
        .agg(
            track_avg_finish_prior=("finish_position", "mean"),
            track_avg_points_prior=("points", "mean"),
        )
    )


def compute_last_known_quali(qualifying_hist: pd.DataFrame) -> pd.DataFrame:
    qualifying_hist = qualifying_hist.sort_values(["driver_id", "season", "round"]).copy()
    last_quali = qualifying_hist.groupby("driver_id", as_index=False).tail(1)
    keep_cols = ["driver_id", "quali_position"]
    return last_quali[keep_cols].rename(columns={"quali_position": "last_known_quali_position"})


def compute_last_known_grid(results_hist: pd.DataFrame) -> pd.DataFrame:
    results_hist = results_hist.sort_values(["driver_id", "season", "round"]).copy()
    last_grid = results_hist.groupby("driver_id", as_index=False).tail(1)
    keep_cols = ["driver_id", "grid"]
    return last_grid[keep_cols].rename(columns={"grid": "last_known_grid"})


def build_future_table(season: int, round_number: int) -> pd.DataFrame:
    results, races, qualifying, model_table, openf1_meetings = load_processed_tables()

    target_race = get_target_race(races, openf1_meetings, season, round_number)
    target_circuit_id = target_race["circuit_id"]
    target_race_name = target_race["race_name"]
    target_race_date = target_race["race_date"]
    target_country = target_race["country"]

    results_with_circuit = results.merge(
        races[["season", "round", "circuit_id", "race_name", "race_date", "country"]],
        on=["season", "round", "race_name"],
        how="left"
    )

    results_hist = get_historical_cutoff(results_with_circuit, season, round_number)
    qualifying_hist = get_historical_cutoff(qualifying, season, round_number)
    model_hist = get_historical_cutoff(model_table, season, round_number)

    if results_hist.empty:
        raise ValueError("No historical data exists before the selected target race.")

    lineup = infer_expected_lineup(results_hist)

    driver_hist = compute_driver_history(results_hist)
    dnf_hist = compute_dnf_rate(results_hist)
    constructor_hist = compute_constructor_history(results_hist)
    track_hist = compute_track_history(results_hist, target_circuit_id)
    last_quali = compute_last_known_quali(qualifying_hist)
    last_grid = compute_last_known_grid(results_hist)

    latest_model_driver_features = (
        model_hist.sort_values(["driver_id", "season", "round"])
        .groupby("driver_id", as_index=False)
        .tail(1)
    )

    reuse_cols = [
        "driver_id",
        "median_lap_time_seconds",
        "mean_lap_time_seconds",
        "total_laps_logged",
        "avg_running_position",
        "avg_lap_pace_prior",
        "avg_quali_position_prior",
    ]
    reuse_cols = [c for c in reuse_cols if c in latest_model_driver_features.columns]
    latest_model_driver_features = latest_model_driver_features[reuse_cols]

    future_df = lineup.merge(driver_hist, on="driver_id", how="left")
    future_df = future_df.merge(dnf_hist, on="driver_id", how="left")
    future_df = future_df.merge(constructor_hist, on="constructor_id", how="left")
    future_df = future_df.merge(track_hist, on="driver_id", how="left")
    future_df = future_df.merge(last_quali, on="driver_id", how="left")
    future_df = future_df.merge(last_grid, on="driver_id", how="left")
    future_df = future_df.merge(latest_model_driver_features, on="driver_id", how="left")

    future_df["season"] = season
    future_df["round"] = round_number
    future_df["race_name"] = target_race_name
    future_df["race_date"] = target_race_date
    future_df["circuit_id"] = target_circuit_id
    future_df["country"] = target_country

    future_df["grid"] = future_df["last_known_grid"]
    future_df["quali_position"] = future_df["last_known_quali_position"]
    future_df["status"] = "Unknown"
    future_df["qualifying_vs_grid_gap"] = future_df["grid"] - future_df["quali_position"]

    future_df["finish_position"] = np.nan
    future_df["points"] = np.nan
    future_df["is_top3"] = np.nan
    future_df["is_winner"] = np.nan

    expected_cols = [
        "season",
        "round",
        "race_name",
        "race_date",
        "circuit_id",
        "country",
        "driver_id",
        "driver_code",
        "given_name",
        "family_name",
        "constructor_id",
        "constructor_name",
        "grid",
        "quali_position",
        "finish_position",
        "points",
        "status",
        "is_dnf",
        "median_lap_time_seconds",
        "mean_lap_time_seconds",
        "total_laps_logged",
        "avg_running_position",
        "avg_finish_position_prior",
        "finish_position_std_prior",
        "recent_form_last_3",
        "recent_points_last_3",
        "dnf_rate_prior",
        "avg_quali_position_prior",
        "avg_lap_pace_prior",
        "team_points_index_prior",
        "team_finish_position_prior",
        "track_avg_finish_prior",
        "track_avg_points_prior",
        "driver_consistency_score",
        "qualifying_vs_grid_gap",
        "is_top3",
        "is_winner",
    ]

    for col in expected_cols:
        if col not in future_df.columns:
            future_df[col] = np.nan

    future_df = future_df[expected_cols]

    numeric_cols = future_df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if future_df[col].isna().all():
            future_df[col] = 0
        else:
            future_df[col] = future_df[col].fillna(future_df[col].median())

    object_cols = future_df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        future_df[col] = future_df[col].fillna("Unknown")

    return future_df


def main(season: int, round_number: int):
    future_df = build_future_table(season, round_number)

    output_csv = PROCESSED_DIR / f"future_race_features_{season}_{round_number}.csv"
    output_parquet = PROCESSED_DIR / f"future_race_features_{season}_{round_number}.parquet"

    future_df.to_csv(output_csv, index=False)

    try:
        future_df.to_parquet(output_parquet, index=False)
        print(f"Saved future feature table to {output_parquet}")
    except Exception:
        print("Parquet save skipped (pyarrow/fastparquet not installed).")

    print(f"Saved future feature table to {output_csv}")
    print(f"Final shape: {future_df.shape}")
    print("\nPreview:")
    print(future_df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()

    main(args.season, args.round)