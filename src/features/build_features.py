from pathlib import Path

import numpy as np
import pandas as pd


PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "model_table.parquet"
OUTPUT_CSV_PATH = PROCESSED_DIR / "model_table.csv"


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    print(f"Loading: {path}")
    return pd.read_csv(path)


def prepare_base_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = safe_read_csv(PROCESSED_DIR / "results.csv")
    qualifying = safe_read_csv(PROCESSED_DIR / "qualifying.csv")
    races = safe_read_csv(PROCESSED_DIR / "races.csv")
    lap_times = safe_read_csv(PROCESSED_DIR / "jolpica_lap_times.csv")

    results["season"] = pd.to_numeric(results["season"], errors="coerce").astype("Int64")
    results["round"] = pd.to_numeric(results["round"], errors="coerce").astype("Int64")
    results["finish_position"] = pd.to_numeric(results["finish_position"], errors="coerce")
    results["grid"] = pd.to_numeric(results["grid"], errors="coerce")
    results["points"] = pd.to_numeric(results["points"], errors="coerce")
    results["fastest_lap_avg_speed"] = pd.to_numeric(results["fastest_lap_avg_speed"], errors="coerce")

    qualifying["season"] = pd.to_numeric(qualifying["season"], errors="coerce").astype("Int64")
    qualifying["round"] = pd.to_numeric(qualifying["round"], errors="coerce").astype("Int64")
    qualifying["quali_position"] = pd.to_numeric(qualifying["quali_position"], errors="coerce")

    races["season"] = pd.to_numeric(races["season"], errors="coerce").astype("Int64")
    races["round"] = pd.to_numeric(races["round"], errors="coerce").astype("Int64")
    races["race_date"] = pd.to_datetime(races["race_date"], errors="coerce")

    lap_times["season"] = pd.to_numeric(lap_times["season"], errors="coerce").astype("Int64")
    lap_times["round"] = pd.to_numeric(lap_times["round"], errors="coerce").astype("Int64")
    lap_times["lap_number"] = pd.to_numeric(lap_times["lap_number"], errors="coerce")
    lap_times["position"] = pd.to_numeric(lap_times["position"], errors="coerce")

    return results, qualifying, races, lap_times


def lap_time_to_seconds(value: str) -> float:
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if ":" in value:
        parts = value.split(":")
        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds

    try:
        return float(value)
    except ValueError:
        return np.nan


def build_lap_features(lap_times: pd.DataFrame) -> pd.DataFrame:
    lap_times = lap_times.copy()
    lap_times["lap_time_seconds"] = lap_times["lap_time"].apply(lap_time_to_seconds)

    lap_features = (
        lap_times.groupby(["season", "round", "driver_id"], as_index=False)
        .agg(
            median_lap_time_seconds=("lap_time_seconds", "median"),
            mean_lap_time_seconds=("lap_time_seconds", "mean"),
            total_laps_logged=("lap_number", "max"),
            avg_running_position=("position", "mean"),
        )
    )

    return lap_features


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_top3"] = (df["finish_position"] <= 3).astype(int)
    df["is_winner"] = (df["finish_position"] == 1).astype(int)
    return df


def add_dnf_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    classified_finish = df["finish_position"].notna()
    completed_status = df["status"].fillna("").str.lower().isin(["finished", "+1 lap", "+2 laps", "+3 laps"])

    df["is_dnf"] = (~classified_finish & ~completed_status).astype(int)

    return df


def sort_race_order(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["driver_id", "season", "round"]).reset_index(drop=True)


def add_driver_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = sort_race_order(df)

    grouped = df.groupby("driver_id", group_keys=False)

    df["avg_finish_position_prior"] = grouped["finish_position"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    df["finish_position_std_prior"] = grouped["finish_position"].transform(
        lambda s: s.shift(1).expanding().std()
    )

    df["recent_form_last_3"] = grouped["finish_position"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )

    df["recent_points_last_3"] = grouped["points"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )

    df["dnf_rate_prior"] = grouped["is_dnf"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    df["avg_quali_position_prior"] = grouped["quali_position"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    df["avg_lap_pace_prior"] = grouped["median_lap_time_seconds"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    return df


def add_constructor_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["constructor_id", "season", "round"]).reset_index(drop=True)

    grouped = df.groupby("constructor_id", group_keys=False)

    df["team_points_index_prior"] = grouped["points"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )

    df["team_finish_position_prior"] = grouped["finish_position"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )

    return df


def add_track_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["driver_id", "circuit_id", "season", "round"]).reset_index(drop=True)

    grouped = df.groupby(["driver_id", "circuit_id"], group_keys=False)

    df["track_avg_finish_prior"] = grouped["finish_position"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    df["track_avg_points_prior"] = grouped["points"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    return df


def finalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].fillna("Unknown")

    return df


def build_model_table() -> pd.DataFrame:
    results, qualifying, races, lap_times = prepare_base_tables()

    lap_features = build_lap_features(lap_times)

    model_df = results.merge(
        qualifying[["season", "round", "driver_id", "quali_position"]],
        on=["season", "round", "driver_id"],
        how="left"
    )

    model_df = model_df.merge(
        races[["season", "round", "race_name", "race_date", "circuit_id", "country"]],
        on=["season", "round", "race_name"],
        how="left"
    )

    model_df = model_df.merge(
        lap_features,
        on=["season", "round", "driver_id"],
        how="left"
    )

    model_df = add_targets(model_df)
    model_df = add_dnf_flag(model_df)
    model_df = add_driver_history_features(model_df)
    model_df = add_constructor_features(model_df)
    model_df = add_track_features(model_df)

    model_df["driver_consistency_score"] = model_df["finish_position_std_prior"]
    model_df["qualifying_vs_grid_gap"] = model_df["grid"] - model_df["quali_position"]

    keep_cols = [
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

    model_df = model_df[[c for c in keep_cols if c in model_df.columns]]
    model_df = finalize_missing_values(model_df)

    return model_df


def main() -> None:
    model_df = build_model_table()

    model_df.to_csv(OUTPUT_CSV_PATH, index=False)

    try:
        model_df.to_parquet(OUTPUT_PATH, index=False)
        print(f"Saved model table to {OUTPUT_PATH}")
    except Exception:
        print("Parquet save skipped (pyarrow/fastparquet not installed).")

    print(f"Saved model table to {OUTPUT_CSV_PATH}")
    print(f"Final shape: {model_df.shape}")
    print("\nPreview:")
    print(model_df.head())


if __name__ == "__main__":
    main()