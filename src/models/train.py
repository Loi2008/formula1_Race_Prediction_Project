import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


DATA_PATH = Path("data/processed/model_table.csv")
ARTIFACTS_DIR = Path("artifacts")
METRICS_DIR = Path("artifacts/metrics")

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Model table not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded model table: {df.shape}")
    return df


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
    return X, y, feature_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def get_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        ),
    }


def train_and_select_best(X_train, y_train):
    preprocessor = build_preprocessor(X_train)
    models = get_models()

    trained_models = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        trained_models[model_name] = pipeline

    return trained_models


def save_training_outputs(trained_models: dict, X_test: pd.DataFrame, y_test: pd.Series, target: str, test_season: int):
    from sklearn.metrics import f1_score

    results = []

    for model_name, pipeline in trained_models.items():
        y_pred = pipeline.predict(X_test)
        score = f1_score(y_test, y_pred, zero_division=0)

        results.append({
            "model": model_name,
            "f1_score": float(score)
        })

    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    model_path = ARTIFACTS_DIR / f"best_{target}_model.pkl"
    metadata_path = METRICS_DIR / f"{target}_train_metadata.json"

    joblib.dump(best_model, model_path)

    metadata = {
        "target": target,
        "best_model": best_model_name,
        "test_season": int(test_season),
        "model_ranking": results,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nBest model: {best_model_name}")
    print(f"Saved model to: {model_path}")
    print(f"Saved training metadata to: {metadata_path}")


def main(target: str):
    if target not in ["is_top3", "is_winner"]:
        raise ValueError("Target must be one of: is_top3, is_winner")

    df = load_data()
    train_df, test_df, test_season = time_aware_split(df)

    X_train, y_train, feature_cols = prepare_features(train_df, target)
    X_test, y_test, _ = prepare_features(test_df, target)

    print(f"Training target: {target}")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Number of features before encoding: {len(feature_cols)}")

    trained_models = train_and_select_best(X_train, y_train)
    save_training_outputs(trained_models, X_test, y_test, target, test_season)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True, help="is_top3 or is_winner")
    args = parser.parse_args()

    main(args.target)