import argparse
from pathlib import Path

import joblib
import pandas as pd


PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("artifacts")

FUTURE_FEATURE_TEMPLATE_PATH = PROCESSED_DIR / "model_table.csv"
TOP3_MODEL_PATH = ARTIFACTS_DIR / "best_is_top3_model.pkl"
WINNER_MODEL_PATH = ARTIFACTS_DIR / "best_is_winner_model.pkl"


def load_future_features(season: int, round_number: int) -> pd.DataFrame:
    path = PROCESSED_DIR / f"future_race_features_{season}_{round_number}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Future feature table not found: {path}. "
            f"Run build_future_features.py first."
        )

    df = pd.read_csv(path)
    print(f"Loaded future feature table: {df.shape}")
    return df


def load_training_template() -> pd.DataFrame:
    if not FUTURE_FEATURE_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Training model table not found: {FUTURE_FEATURE_TEMPLATE_PATH}")
    return pd.read_csv(FUTURE_FEATURE_TEMPLATE_PATH)


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
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
    return [col for col in df.columns if col not in drop_cols]


def align_feature_dtypes(future_df: pd.DataFrame, template_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align future feature dtypes to the same feature dtypes used in training.
    This prevents numeric columns from being treated as strings like 'Unknown'.
    """
    future_df = future_df.copy()

    template_feature_cols = get_feature_columns(template_df)
    future_feature_cols = get_feature_columns(future_df)

    missing_cols = [col for col in template_feature_cols if col not in future_feature_cols]
    extra_cols = [col for col in future_feature_cols if col not in template_feature_cols]

    if missing_cols:
        raise ValueError(f"Future feature file is missing required columns: {missing_cols}")

    if extra_cols:
        # Extra columns won't be used by the model, so drop them from feature view later.
        print(f"Dropping extra future feature columns not used in training: {extra_cols}")

    for col in template_feature_cols:
        template_dtype = template_df[col].dtype

        if pd.api.types.is_numeric_dtype(template_dtype):
            future_df[col] = pd.to_numeric(future_df[col], errors="coerce")
        else:
            future_df[col] = future_df[col].astype(str).fillna("Unknown")

    return future_df


def prepare_features(future_df: pd.DataFrame, template_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep exactly the same feature columns and order used during training.
    """
    template_feature_cols = get_feature_columns(template_df)
    X = future_df[template_feature_cols].copy()
    return X


def add_probability_column(
    df: pd.DataFrame,
    template_df: pd.DataFrame,
    model,
    prob_col_name: str
) -> pd.DataFrame:
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"Loaded model does not support predict_proba() for {prob_col_name}")

    aligned_df = align_feature_dtypes(df, template_df)
    X = prepare_features(aligned_df, template_df)

    aligned_df = aligned_df.copy()
    aligned_df[prob_col_name] = model.predict_proba(X)[:, 1]
    return aligned_df


def format_driver_name(df: pd.DataFrame) -> pd.Series:
    return df["given_name"].fillna("") + " " + df["family_name"].fillna("")


def predict_future_race(season: int, round_number: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    future_df = load_future_features(season, round_number)
    template_df = load_training_template()

    top3_model = load_model(TOP3_MODEL_PATH)
    winner_model = load_model(WINNER_MODEL_PATH)

    scored_df = add_probability_column(future_df, template_df, top3_model, "top3_probability")
    scored_df = add_probability_column(scored_df, template_df, winner_model, "winner_probability")

    scored_df["driver"] = format_driver_name(scored_df)

    ranked_df = scored_df.sort_values(
        ["top3_probability", "winner_probability"],
        ascending=False
    ).reset_index(drop=True)

    ranked_df["predicted_podium_position"] = ranked_df.index + 1

    podium_df = ranked_df.head(3).copy()

    winner_df = scored_df.sort_values(
        "winner_probability",
        ascending=False
    ).head(1).copy()

    return ranked_df, podium_df, winner_df


def save_outputs(
    ranked_df: pd.DataFrame,
    podium_df: pd.DataFrame,
    winner_df: pd.DataFrame,
    season: int,
    round_number: int
) -> None:
    ranked_path = ARTIFACTS_DIR / f"future_race_ranking_{season}_{round_number}.csv"
    podium_path = ARTIFACTS_DIR / f"future_race_podium_{season}_{round_number}.csv"
    winner_path = ARTIFACTS_DIR / f"future_race_winner_{season}_{round_number}.csv"

    ranked_df.to_csv(ranked_path, index=False)
    podium_df.to_csv(podium_path, index=False)
    winner_df.to_csv(winner_path, index=False)

    print(f"Saved full ranking to: {ranked_path}")
    print(f"Saved predicted podium to: {podium_path}")
    print(f"Saved predicted winner to: {winner_path}")


def print_outputs(ranked_df: pd.DataFrame, podium_df: pd.DataFrame, winner_df: pd.DataFrame) -> None:
    race_name = ranked_df["race_name"].iloc[0] if "race_name" in ranked_df.columns else "Unknown Race"
    season = ranked_df["season"].iloc[0]
    round_number = ranked_df["round"].iloc[0]

    print(f"\n=== Future Race Prediction: {race_name} ({season}, Round {round_number}) ===")

    print("\nPredicted Winner:")
    print(
        winner_df[
            [
                "driver",
                "constructor_name",
                "winner_probability",
                "top3_probability",
            ]
        ].to_string(index=False)
    )

    print("\nPredicted Podium:")
    print(
        podium_df[
            [
                "predicted_podium_position",
                "driver",
                "constructor_name",
                "top3_probability",
                "winner_probability",
            ]
        ].to_string(index=False)
    )

    print("\nTop 10 Driver Ranking:")
    print(
        ranked_df.head(10)[
            [
                "driver",
                "constructor_name",
                "grid",
                "quali_position",
                "top3_probability",
                "winner_probability",
            ]
        ].to_string(index=False)
    )


def main(season: int, round_number: int):
    ranked_df, podium_df, winner_df = predict_future_race(season, round_number)
    print_outputs(ranked_df, podium_df, winner_df)
    save_outputs(ranked_df, podium_df, winner_df, season, round_number)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()

    main(args.season, args.round)
