import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


DATA_PATH = Path("data/processed/model_table.csv")
ARTIFACTS_DIR = Path("artifacts")
METRICS_DIR = Path("artifacts/metrics")

METRICS_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Model table not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def time_aware_split(df: pd.DataFrame):
    seasons = sorted(df["season"].dropna().unique())
    if len(seasons) < 2:
        raise ValueError("Need at least 2 seasons for time-aware split.")

    test_season = seasons[-1]
    train_df = df[df["season"] < test_season].copy()
    test_df = df[df["season"] == test_season].copy()

    return train_df, test_df, test_season


def prepare_features(df: pd.DataFrame, target: str):
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
    X = df[feature_cols].copy()
    y = df[target].copy()
    return X, y


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    return metrics


def main(target: str):
    if target not in ["is_top3", "is_winner"]:
        raise ValueError("Target must be one of: is_top3, is_winner")

    model_path = ARTIFACTS_DIR / f"best_{target}_model.pkl"
    metrics_path = METRICS_DIR / f"{target}_evaluation.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    df = load_data()
    _, test_df, test_season = time_aware_split(df)
    X_test, y_test = prepare_features(test_df, target)

    model = joblib.load(model_path)
    metrics = evaluate_model(model, X_test, y_test)

    output = {
        "target": target,
        "test_season": int(test_season),
        "model_path": str(model_path),
        **metrics,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    print(f"\nSaved evaluation to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True, help="is_top3 or is_winner")
    args = parser.parse_args()

    main(args.target)