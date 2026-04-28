import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


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

    print(f"Train seasons: {sorted(train_df['season'].unique())}")
    print(f"Test season: {test_season}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    return train_df, test_df, test_season


def prepare_features(df: pd.DataFrame):
    target = "points"

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
    y = pd.to_numeric(df[target], errors="coerce").fillna(0)

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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor


def get_models():
    return {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=1.0),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost_regressor": XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
        ),
    }


def evaluate_regression(model_name: str, pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    metrics = {
        "model": model_name,
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2_score": float(r2_score(y_test, y_pred)),
    }

    return metrics


def train_points_models(X_train, y_train, X_test, y_test):
    preprocessor = build_preprocessor(X_train)
    models = get_models()

    trained_models = {}
    metrics_list = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        metrics = evaluate_regression(model_name, pipeline, X_test, y_test)

        trained_models[model_name] = pipeline
        metrics_list.append(metrics)

        print(json.dumps(metrics, indent=2))

    return trained_models, metrics_list


def save_best_model(trained_models: dict, metrics_list: list[dict], test_season: int):
    metrics_df = pd.DataFrame(metrics_list)

    # For regression, lower RMSE is better
    best_row = metrics_df.sort_values("rmse", ascending=True).iloc[0]

    best_model_name = best_row["model"]
    best_model = trained_models[best_model_name]

    model_path = ARTIFACTS_DIR / "best_points_model.pkl"
    metrics_path = METRICS_DIR / "points_metrics.json"
    metadata_path = METRICS_DIR / "points_metadata.json"

    joblib.dump(best_model, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=2)

    metadata = {
        "target": "points",
        "best_model": best_model_name,
        "selection_metric": "rmse",
        "test_season": int(test_season),
        "best_rmse": float(best_row["rmse"]),
        "best_mae": float(best_row["mae"]),
        "best_r2_score": float(best_row["r2_score"]),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nBest points model:")
    print(json.dumps(metadata, indent=2))

    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved metadata to: {metadata_path}")


def main():
    df = load_data()

    train_df, test_df, test_season = time_aware_split(df)

    X_train, y_train, feature_cols = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df)

    print(f"Target: points")
    print(f"Number of features before encoding: {len(feature_cols)}")

    trained_models, metrics_list = train_points_models(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    save_best_model(trained_models, metrics_list, test_season)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()

    main()