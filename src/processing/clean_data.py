import re
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def snake_case(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text.lower()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [snake_case(col) for col in df.columns]
    return df


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        print(f"Loading: {path}")
        return pd.read_csv(path)
    print(f"Missing file: {path}")
    return None


def save_output(df: pd.DataFrame, filename: str) -> None:
    csv_path = PROCESSED_DIR / f"{filename}.csv"
    parquet_path = PROCESSED_DIR / f"{filename}.parquet"

    df.to_csv(csv_path, index=False)

    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        print(f"Parquet save skipped for {filename} (pyarrow/fastparquet not installed).")

    print(f"Saved cleaned table: {csv_path}")


def convert_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def convert_datetime(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def fill_categorical_unknown(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].fillna("Unknown")
    return df


def combine_fastf1_files(prefix: str) -> pd.DataFrame | None:
    folder = RAW_DIR / "fastf1"
    files = sorted(folder.glob(f"{prefix}_*.csv"))

    if not files:
        print(f"No FastF1 files found for prefix: {prefix}")
        return None

    frames = []
    for file in files:
        print(f"Loading: {file}")
        frames.append(pd.read_csv(file))

    combined = pd.concat(frames, ignore_index=True)
    return combined


def clean_races(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df = convert_numeric(df, ["season", "round", "lat", "long"])
    df = convert_datetime(df, ["race_date"])
    df = fill_categorical_unknown(df)

    keep_cols = [
        "season", "round", "race_name", "race_date", "race_time",
        "circuit_id", "circuit_name", "locality", "country", "lat", "long"
    ]
    df = df[[c for c in keep_cols if c in df.columns]]
    df = df.drop_duplicates(subset=["season", "round"])
    return df


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df = convert_numeric(
        df,
        [
            "season", "round", "grid", "position", "points", "laps",
            "finish_time_ms", "fastest_lap_rank", "fastest_lap_number",
            "fastest_lap_avg_speed"
        ]
    )
    df = fill_categorical_unknown(df)

    if "position" in df.columns:
        df = df.rename(columns={"position": "finish_position"})

    keep_cols = [
        "season", "round", "race_name",
        "driver_id", "driver_code", "driver_number",
        "given_name", "family_name", "date_of_birth", "nationality",
        "constructor_id", "constructor_name", "constructor_nationality",
        "grid", "finish_position", "position_text", "points", "laps", "status",
        "finish_time_ms", "finish_time_text",
        "fastest_lap_rank", "fastest_lap_number", "fastest_lap_time",
        "fastest_lap_avg_speed", "fastest_lap_speed_units"
    ]
    df = df[[c for c in keep_cols if c in df.columns]]
    df = df.drop_duplicates(subset=["season", "round", "driver_id"])
    return df


def clean_qualifying(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df = convert_numeric(df, ["season", "round", "position"])
    df = fill_categorical_unknown(df)

    if "position" in df.columns:
        df = df.rename(columns={"position": "quali_position"})

    keep_cols = [
        "season", "round", "race_name",
        "driver_id", "driver_code", "driver_number",
        "given_name", "family_name",
        "constructor_id", "constructor_name",
        "quali_position", "q1", "q2", "q3"
    ]
    df = df[[c for c in keep_cols if c in df.columns]]
    df = df.drop_duplicates(subset=["season", "round", "driver_id"])
    return df


def clean_jolpica_lap_times(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df = convert_numeric(df, ["season", "round", "lap_number", "position"])
    df = fill_categorical_unknown(df)

    keep_cols = [
        "season", "round", "race_name",
        "lap_number", "driver_id", "position", "lap_time"
    ]
    df = df[[c for c in keep_cols if c in df.columns]]
    df = df.drop_duplicates(subset=["season", "round", "driver_id", "lap_number"])
    return df


def clean_openf1_meetings(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df = fill_categorical_unknown(df)

    numeric_cols = ["meeting_key", "year", "country_key", "circuit_key"]
    date_cols = ["date_start"]

    df = convert_numeric(df, [c for c in numeric_cols if c in df.columns])
    df = convert_datetime(df, [c for c in date_cols if c in df.columns])

    df = df.drop_duplicates(subset=["meeting_key"])
    return df


def clean_openf1_sessions(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df = fill_categorical_unknown(df)

    numeric_cols = ["session_key", "meeting_key", "year", "session_type"]
    date_cols = ["date_start", "date_end"]

    df = convert_numeric(df, [c for c in numeric_cols if c in df.columns])
    df = convert_datetime(df, [c for c in date_cols if c in df.columns])

    df = df.drop_duplicates(subset=["session_key"])
    return df


def clean_openf1_drivers(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df = fill_categorical_unknown(df)

    numeric_cols = ["session_key", "meeting_key", "driver_number"]
    df = convert_numeric(df, [c for c in numeric_cols if c in df.columns])

    df = df.drop_duplicates(subset=["session_key", "driver_number"])
    return df


def clean_openf1_laps(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df = fill_categorical_unknown(df)

    numeric_cols = [
        "session_key", "meeting_key", "driver_number", "lap_number",
        "duration_sector_1", "duration_sector_2", "duration_sector_3",
        "lap_duration", "i1_speed", "i2_speed", "st_speed"
    ]
    date_cols = ["date_start"]

    df = convert_numeric(df, [c for c in numeric_cols if c in df.columns])
    df = convert_datetime(df, [c for c in date_cols if c in df.columns])

    df = df.drop_duplicates(subset=["session_key", "driver_number", "lap_number"])
    return df


def clean_openf1_weather(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df = fill_categorical_unknown(df)

    numeric_cols = [
        "session_key", "meeting_key", "air_temperature", "humidity",
        "pressure", "track_temperature", "wind_direction", "wind_speed"
    ]
    date_cols = ["date"]

    df = convert_numeric(df, [c for c in numeric_cols if c in df.columns])
    df = convert_datetime(df, [c for c in date_cols if c in df.columns])

    df = df.drop_duplicates()
    return df


def clean_openf1_position(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df = fill_categorical_unknown(df)

    numeric_cols = ["session_key", "meeting_key", "driver_number", "position"]
    date_cols = ["date"]

    df = convert_numeric(df, [c for c in numeric_cols if c in df.columns])
    df = convert_datetime(df, [c for c in date_cols if c in df.columns])

    df = df.drop_duplicates(subset=["session_key", "driver_number", "date"])
    return df


def clean_fastf1_laps(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df = fill_categorical_unknown(df)

    numeric_cols = [
        "season", "round", "drivernumber", "lapnumber", "stint",
        "tyrelife", "position", "speedi1", "speedi2", "speedfl", "speedst"
    ]
    df = convert_numeric(df, [c for c in numeric_cols if c in df.columns])

    # Keep as strings for now; feature engineering can convert later if needed
    timedelta_like = [
        "time", "laptime", "pitouttime", "pitintime",
        "sector1time", "sector2time", "sector3time"
    ]
    for col in timedelta_like:
        if col in df.columns:
            df[col] = df[col].astype(str)

    dedupe_cols = [c for c in ["season", "round", "driver", "lapnumber"] if c in df.columns]
    if dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols)
    else:
        df = df.drop_duplicates()

    return df


def clean_fastf1_weather(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df = fill_categorical_unknown(df)

    numeric_cols = [
        "season", "round", "airtemp", "humidity",
        "pressure", "rainfall", "tracktemp", "winddirection", "windspeed"
    ]
    df = convert_numeric(df, [c for c in numeric_cols if c in df.columns])

    if "time" in df.columns:
        df["time"] = df["time"].astype(str)

    dedupe_cols = [c for c in ["season", "round", "time"] if c in df.columns]
    if dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols)
    else:
        df = df.drop_duplicates()

    return df


def clean_fastf1_results(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df = fill_categorical_unknown(df)

    numeric_cols = [
        "season", "round", "position", "points", "gridposition",
        "q1", "q2", "q3"
    ]
    df = convert_numeric(df, [c for c in numeric_cols if c in df.columns])

    dedupe_cols = [c for c in ["season", "round", "abbreviation"] if c in df.columns]
    if dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols)
    else:
        df = df.drop_duplicates()

    return df


def build_dimension_tables(results_df: pd.DataFrame, races_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    drivers = (
        results_df[
            [
                "driver_id", "driver_code", "driver_number",
                "given_name", "family_name", "date_of_birth", "nationality"
            ]
        ]
        .drop_duplicates(subset=["driver_id"])
        .reset_index(drop=True)
    )

    constructors = (
        results_df[
            ["constructor_id", "constructor_name", "constructor_nationality"]
        ]
        .drop_duplicates(subset=["constructor_id"])
        .reset_index(drop=True)
    )

    circuits = (
        races_df[
            ["circuit_id", "circuit_name", "locality", "country", "lat", "long"]
        ]
        .drop_duplicates(subset=["circuit_id"])
        .reset_index(drop=True)
    )

    return drivers, constructors, circuits


def main() -> None:
    # Jolpica
    races = safe_read_csv(RAW_DIR / "jolpica" / "races.csv")
    results = safe_read_csv(RAW_DIR / "jolpica" / "results.csv")
    qualifying = safe_read_csv(RAW_DIR / "jolpica" / "qualifying.csv")
    jolpica_laps = safe_read_csv(RAW_DIR / "jolpica" / "lap_times.csv")

    # OpenF1
    meetings = safe_read_csv(RAW_DIR / "openf1" / "meetings.csv")
    sessions = safe_read_csv(RAW_DIR / "openf1" / "sessions.csv")
    openf1_drivers = safe_read_csv(RAW_DIR / "openf1" / "drivers.csv")
    openf1_laps = safe_read_csv(RAW_DIR / "openf1" / "laps.csv")
    openf1_weather = safe_read_csv(RAW_DIR / "openf1" / "weather.csv")
    openf1_position = safe_read_csv(RAW_DIR / "openf1" / "position.csv")

    # FastF1 combined files
    fastf1_laps = combine_fastf1_files("laps")
    fastf1_weather = combine_fastf1_files("weather")
    fastf1_results = combine_fastf1_files("results")

    # Clean and save
    if races is not None:
        races = clean_races(races)
        save_output(races, "races")

    if results is not None:
        results = clean_results(results)
        save_output(results, "results")

    if qualifying is not None:
        qualifying = clean_qualifying(qualifying)
        save_output(qualifying, "qualifying")

    if jolpica_laps is not None:
        jolpica_laps = clean_jolpica_lap_times(jolpica_laps)
        save_output(jolpica_laps, "jolpica_lap_times")

    if meetings is not None:
        meetings = clean_openf1_meetings(meetings)
        save_output(meetings, "openf1_meetings")

    if sessions is not None:
        sessions = clean_openf1_sessions(sessions)
        save_output(sessions, "openf1_sessions")

    if openf1_drivers is not None:
        openf1_drivers = clean_openf1_drivers(openf1_drivers)
        save_output(openf1_drivers, "openf1_drivers")

    if openf1_laps is not None:
        openf1_laps = clean_openf1_laps(openf1_laps)
        save_output(openf1_laps, "openf1_laps")

    if openf1_weather is not None:
        openf1_weather = clean_openf1_weather(openf1_weather)
        save_output(openf1_weather, "openf1_weather")

    if openf1_position is not None:
        openf1_position = clean_openf1_position(openf1_position)
        save_output(openf1_position, "openf1_position")

    if fastf1_laps is not None:
        fastf1_laps = clean_fastf1_laps(fastf1_laps)
        save_output(fastf1_laps, "fastf1_laps")

    if fastf1_weather is not None:
        fastf1_weather = clean_fastf1_weather(fastf1_weather)
        save_output(fastf1_weather, "fastf1_weather")

    if fastf1_results is not None:
        fastf1_results = clean_fastf1_results(fastf1_results)
        save_output(fastf1_results, "fastf1_results")

    if results is not None and races is not None:
        drivers, constructors, circuits = build_dimension_tables(results, races)
        save_output(drivers, "drivers")
        save_output(constructors, "constructors")
        save_output(circuits, "circuits")

    print("\nData cleaning completed successfully.")


if __name__ == "__main__":
    main()