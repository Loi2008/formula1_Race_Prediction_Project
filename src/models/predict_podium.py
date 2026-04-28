import argparse
from pathlib import Path

import joblib
import pandas as pd


DATA_PATH = Path("data/processed/model_table.csv")
MODEL_PATH = Path("artifacts/best_is_top3_model.pkl")


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Model table not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        "race_name",
        "race_date",
        "driver_id",
        "driver_code",
        "given_name",
        "family_name",
        "constructor_id",
        "constructor_name",
        "status",
        "finish_position",
        "points",
        "is_top3",
        "is_winner",
    ]

    feature_cols = [col for col in df.columns if col not in drop_cols]
    return df[feature_cols].copy()


def get_available_rounds(df: pd.DataFrame, season: int) -> list[int]:
    season_df = df[df["season"] == season].copy()
    if season_df.empty:
        return []

    rounds = (
        pd.to_numeric(season_df["round"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    return sorted(rounds)


def predict_race_podium(df: pd.DataFrame, season: int, round_number: int) -> pd.DataFrame:
    race_df = df[(df["season"] == season) & (df["round"] == round_number)].copy()

    if race_df.empty:
        available_rounds = get_available_rounds(df, season)
        raise ValueError(
            f"No rows found for season={season}, round={round_number}. "
            f"Available rounds for {season}: {available_rounds}"
        )

    model = load_model()
    X_race = prepare_features(race_df)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba().")

    race_df["top3_probability"] = model.predict_proba(X_race)[:, 1]

    podium = (
        race_df.sort_values("top3_probability", ascending=False)
        .head(3)
        .reset_index(drop=True)
    )

    podium["predicted_podium_position"] = podium.index + 1

    return podium[
        [
            "predicted_podium_position",
            "season",
            "round",
            "race_name",
            "driver_id",
            "driver_code",
            "given_name",
            "family_name",
            "constructor_name",
            "grid",
            "quali_position",
            "top3_probability",
        ]
    ]


def main(season: int, round_number: int):
    df = load_data()
    podium = predict_race_podium(df, season, round_number)

    print("\nPredicted Podium:")
    print(podium.to_string(index=False))

    output_path = Path(f"artifacts/predicted_podium_{season}_{round_number}.csv")
    podium.to_csv(output_path, index=False)
    print(f"\nSaved podium prediction to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()

    main(args.season, args.round)