from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import joblib
import pandas as pd
import streamlit as st

from src.features.build_future_features import build_future_table


PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("artifacts")

MODEL_TABLE_PATH = PROCESSED_DIR / "model_table.csv"
TOP3_MODEL_PATH = ARTIFACTS_DIR / "best_is_top3_model.pkl"
WINNER_MODEL_PATH = ARTIFACTS_DIR / "best_is_winner_model.pkl"
POINTS_MODEL_PATH = ARTIFACTS_DIR / "best_points_model.pkl"
OPENF1_MEETINGS_PATH = PROCESSED_DIR / "openf1_meetings.csv"


st.set_page_config(
    page_title="F1 Race Prediction App",
    page_icon="🏎️",
    layout="wide"
)


@st.cache_data
def load_model_table() -> pd.DataFrame:
    if not MODEL_TABLE_PATH.exists():
        raise FileNotFoundError(f"Missing file: {MODEL_TABLE_PATH}")
    return pd.read_csv(MODEL_TABLE_PATH)


@st.cache_data
def load_openf1_meetings() -> pd.DataFrame:
    if not OPENF1_MEETINGS_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(OPENF1_MEETINGS_PATH)

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    if "date_start" in df.columns:
        df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")

    return df


@st.cache_resource
def load_top3_model():
    if not TOP3_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {TOP3_MODEL_PATH}")
    return joblib.load(TOP3_MODEL_PATH)


@st.cache_resource
def load_winner_model():
    if not WINNER_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {WINNER_MODEL_PATH}")
    return joblib.load(WINNER_MODEL_PATH)


@st.cache_resource
def load_points_model():
    if not POINTS_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model: {POINTS_MODEL_PATH}. "
            f"Run: python src/models/train_points.py"
        )
    return joblib.load(POINTS_MODEL_PATH)


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
    future_df = future_df.copy()
    template_feature_cols = get_feature_columns(template_df)

    for col in template_feature_cols:
        if col not in future_df.columns:
            future_df[col] = pd.NA

        template_dtype = template_df[col].dtype

        if pd.api.types.is_numeric_dtype(template_dtype):
            future_df[col] = pd.to_numeric(future_df[col], errors="coerce")
        else:
            future_df[col] = future_df[col].astype(str).fillna("Unknown")

    return future_df


def prepare_features(df: pd.DataFrame, template_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = get_feature_columns(template_df)
    return df[feature_cols].copy()


def format_driver_name(df: pd.DataFrame) -> pd.Series:
    return df["given_name"].fillna("") + " " + df["family_name"].fillna("")


def score_dataframe(
    df: pd.DataFrame,
    template_df: pd.DataFrame,
    top3_model,
    winner_model,
    points_model,
) -> pd.DataFrame:
    aligned_df = align_feature_dtypes(df, template_df)
    X = prepare_features(aligned_df, template_df)

    aligned_df = aligned_df.copy()
    aligned_df["top3_probability"] = top3_model.predict_proba(X)[:, 1]
    aligned_df["winner_probability"] = winner_model.predict_proba(X)[:, 1]
    aligned_df["predicted_points"] = points_model.predict(X)
    aligned_df["predicted_points"] = aligned_df["predicted_points"].clip(lower=0)
    aligned_df["driver"] = format_driver_name(aligned_df)

    ranked_df = aligned_df.sort_values(
        ["top3_probability", "winner_probability"],
        ascending=False
    ).reset_index(drop=True)

    ranked_df["predicted_podium_position"] = ranked_df.index + 1

    return ranked_df


def future_feature_path(season: int, round_number: int) -> Path:
    return PROCESSED_DIR / f"future_race_features_{season}_{round_number}.csv"


def load_or_build_future_feature_file(season: int, round_number: int) -> pd.DataFrame:
    path = future_feature_path(season, round_number)

    if path.exists():
        return pd.read_csv(path)

    with st.spinner(f"Building future features for {season} round {round_number}..."):
        future_df = build_future_table(season, round_number)
        future_df.to_csv(path, index=False)

        try:
            future_df.to_parquet(
                PROCESSED_DIR / f"future_race_features_{season}_{round_number}.parquet",
                index=False
            )
        except Exception:
            pass

    st.success(f"Built future feature file for {season} round {round_number}.")
    return future_df


def get_historical_actual_podium(race_df: pd.DataFrame) -> pd.DataFrame:
    actual = (
        race_df.sort_values("finish_position", ascending=True)
        .head(3)
        .reset_index(drop=True)
        .copy()
    )

    actual["actual_podium_position"] = actual.index + 1
    actual["driver"] = format_driver_name(actual)

    return actual[
        [
            "actual_podium_position",
            "driver",
            "constructor_name",
            "finish_position",
            "points",
        ]
    ]

def show_prediction_outputs(
    ranked_df: pd.DataFrame,
    probability_view: str,
    historical: bool = False,
    chart_key: str = "chart_view"
):
    selected_prob_col = (
        "top3_probability"
        if probability_view == "Top 3 Probability"
        else "winner_probability"
    )

    selected_prob_label = probability_view.lower()

    podium_df = (
        ranked_df.sort_values(selected_prob_col, ascending=False)
        .head(3)
        .reset_index(drop=True)
        .copy()
    )
    podium_df["predicted_podium_position"] = podium_df.index + 1

    winner_df = (
        ranked_df.sort_values("winner_probability", ascending=False)
        .head(1)
        .copy()
    )

    points_df = (
        ranked_df.sort_values("predicted_points", ascending=False)
        .reset_index(drop=True)
        .copy()
    )
    points_df["predicted_points_rank"] = points_df.index + 1

    race_name = (
        ranked_df["race_name"].iloc[0]
        if "race_name" in ranked_df.columns
        else "Unknown Race"
    )
    season = ranked_df["season"].iloc[0]
    round_number = ranked_df["round"].iloc[0]

    st.subheader(f"📍 {race_name} ({season}, Round {round_number})")

    if "race_date" in ranked_df.columns:
        st.caption(f"Race Date: {ranked_df['race_date'].iloc[0]}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🥇 Predicted Winner")
        winner_display = winner_df[
            [
                "driver",
                "constructor_name",
                "winner_probability",
                "top3_probability",
                "predicted_points",
            ]
        ].copy()

        winner_display["winner_probability"] = winner_display["winner_probability"].round(4)
        winner_display["top3_probability"] = winner_display["top3_probability"].round(4)
        winner_display["predicted_points"] = winner_display["predicted_points"].round(2)

        st.dataframe(winner_display, use_container_width=True)

    with col2:
        st.markdown(f"### 🏁 Predicted Podium by {probability_view}")
        podium_display = podium_df[
            [
                "predicted_podium_position",
                "driver",
                "constructor_name",
                "grid",
                "quali_position",
                "top3_probability",
                "winner_probability",
                "predicted_points",
            ]
        ].copy()

        podium_display["top3_probability"] = podium_display["top3_probability"].round(4)
        podium_display["winner_probability"] = podium_display["winner_probability"].round(4)
        podium_display["predicted_points"] = podium_display["predicted_points"].round(2)

        st.dataframe(podium_display, use_container_width=True)

    st.markdown("### 🔢 Predicted Points Ranking")
    points_display = points_df[
        [
            "predicted_points_rank",
            "driver",
            "constructor_name",
            "grid",
            "quali_position",
            "predicted_points",
            "top3_probability",
            "winner_probability",
        ]
    ].copy()

    points_display["predicted_points"] = points_display["predicted_points"].round(2)
    points_display["top3_probability"] = points_display["top3_probability"].round(4)
    points_display["winner_probability"] = points_display["winner_probability"].round(4)

    st.dataframe(points_display, use_container_width=True)

    if historical and "finish_position" in ranked_df.columns and ranked_df["finish_position"].notna().any():
        st.markdown("### 🏆 Actual Podium")
        actual_podium = get_historical_actual_podium(ranked_df)
        st.dataframe(actual_podium, use_container_width=True)

    st.markdown(f"### 📊 Full Driver Ranking by {probability_view}")

    ranking_display = ranked_df[
        [
            "driver",
            "constructor_name",
            "grid",
            "quali_position",
            "finish_position",
            "points",
            "predicted_points",
            "top3_probability",
            "winner_probability",
        ]
    ].copy()

    ranking_display["predicted_points"] = ranking_display["predicted_points"].round(2)
    ranking_display["top3_probability"] = ranking_display["top3_probability"].round(4)
    ranking_display["winner_probability"] = ranking_display["winner_probability"].round(4)

    ranking_display = (
        ranking_display.sort_values(selected_prob_col, ascending=False)
        .reset_index(drop=True)
    )

    st.dataframe(ranking_display, use_container_width=True)

    chart_choice = st.radio(
    "Chart View",
    ["Top 3 Probability", "Winner Probability", "Predicted Points"],
    horizontal=True,
    key=chart_key
    )

    if chart_choice == "Top 3 Probability":
        chart_col = "top3_probability"
    elif chart_choice == "Winner Probability":
        chart_col = "winner_probability"
    else:
        chart_col = "predicted_points"

    st.markdown(f"### 📈 Driver Chart by {chart_choice}")
    chart_df = ranked_df[["driver", chart_col]].copy().sort_values(
        chart_col,
        ascending=False
    )

    st.bar_chart(chart_df.set_index("driver"))

    st.markdown("### ℹ️ Prediction Views")
    st.write(
        f"""
        The current ranking view is showing **{selected_prob_label}**.

        - **Top 3 Probability** ranks drivers by chance of finishing on the podium.
        - **Winner Probability** ranks drivers by chance of winning the race.
        - **Predicted Points** estimates how many points each driver may score.
        """
    )


def get_available_historical_races(model_table: pd.DataFrame) -> pd.DataFrame:
    return (
        model_table[["season", "round", "race_name", "race_date"]]
        .drop_duplicates()
        .sort_values(["season", "round"])
        .reset_index(drop=True)
    )

def get_available_future_rounds(openf1_meetings: pd.DataFrame, season: int) -> list[int]:
    if openf1_meetings.empty or "year" not in openf1_meetings.columns:
        return []

    season_df = openf1_meetings[openf1_meetings["year"] == season].copy()

    if season_df.empty:
        return []

    season_df = season_df.sort_values("date_start").reset_index(drop=True)
    season_df["derived_round"] = season_df.index + 1

    return season_df["derived_round"].astype(int).tolist()


def get_future_round_labels(openf1_meetings: pd.DataFrame, season: int) -> dict[int, str]:
    if openf1_meetings.empty or "year" not in openf1_meetings.columns:
        return {}

    season_df = openf1_meetings[openf1_meetings["year"] == season].copy()

    if season_df.empty:
        return {}

    season_df = season_df.sort_values("date_start").reset_index(drop=True)
    season_df["derived_round"] = season_df.index + 1

    labels = {}
    for _, row in season_df.iterrows():
        labels[int(row["derived_round"])] = row.get(
            "meeting_name",
            f"Round {int(row['derived_round'])}"
        )

    return labels


def render_2025_tab(
    model_table,
    top3_model,
    winner_model,
    points_model,
    probability_view,
):
    st.header("🏁 2025 Race Prediction")
    st.write("Analyze completed 2025 races and compare predictions with actual results.")

    historical_races = get_available_historical_races(model_table)
    historical_races = historical_races[historical_races["season"] == 2025].copy()

    if historical_races.empty:
        st.warning("No 2025 race data found.")
        return

    race_options = {
        int(row["round"]): f"Round {int(row['round'])} - {row['race_name']}"
        for _, row in historical_races.iterrows()
    }

    selected_round_2025 = st.selectbox(
        "Select 2025 Race",
        list(race_options.keys()),
        format_func=lambda x: race_options[x],
        key="race_2025"
    )

    race_df = model_table[
        (model_table["season"] == 2025)
        & (model_table["round"] == selected_round_2025)
    ].copy()

    if race_df.empty:
        st.warning("No historical data found for the selected 2025 race.")
        return

    ranked_df = score_dataframe(
        race_df,
        model_table,
        top3_model,
        winner_model,
        points_model,
    )

    show_prediction_outputs(
    ranked_df,
    probability_view=probability_view,
    historical=True,
    chart_key="chart_view_2025"
    )


def render_2026_tab(
    model_table,
    openf1_meetings,
    top3_model,
    winner_model,
    points_model,
    probability_view,
):
    st.header("🔮 2026 Upcoming Race Prediction")
    st.write("Predict upcoming 2026 races using the future feature pipeline.")

    selected_season = 2026
    available_rounds = get_available_future_rounds(
        openf1_meetings,
        selected_season
    )
    round_labels = get_future_round_labels(
        openf1_meetings,
        selected_season
    )

    if not available_rounds:
        st.warning("No 2026 meetings found in openf1_meetings.csv.")
        return

    selected_round_2026 = st.selectbox(
        "Select 2026 Race",
        available_rounds,
        format_func=lambda x: f"Round {x} - {round_labels.get(x, f'Round {x}')}",
        key="race_2026"
    )

    try:
        future_df = load_or_build_future_feature_file(
            selected_season,
            selected_round_2026
        )
    except Exception as e:
        st.error(f"Could not build or load future features: {e}")
        return

    ranked_df = score_dataframe(
        future_df,
        model_table,
        top3_model,
        winner_model,
        points_model,
    )

    show_prediction_outputs(
    ranked_df,
    probability_view=probability_view,
    historical=False,
    chart_key="chart_view_2026"
    )

    st.info(
        """
        2026 predictions are based on the future feature pipeline.
        If a future feature file does not exist, the app builds it automatically.

        Current predictions mainly use:
        - historical driver form
        - team performance
        - track history
        - last known qualifying/grid proxies
        """
    )


def main():
    st.title("🏎️ Formula 1 Race Prediction App")

    st.markdown(
        """
        A production-style Formula 1 prediction application for race outcomes,
        podium probability, winner probability, and predicted points.
        """
    )

    try:
        model_table = load_model_table()
        openf1_meetings = load_openf1_meetings()
        top3_model = load_top3_model()
        winner_model = load_winner_model()
        points_model = load_points_model()
    except Exception as e:
        st.error(f"Startup error: {e}")
        st.stop()

    probability_view = st.sidebar.radio(
        "Probability View",
        ["Top 3 Probability", "Winner Probability"]
    )

    st.sidebar.markdown("---")
    st.sidebar.write("Use the tabs on the home page to switch between 2025 and 2026 predictions.")

    tab_2025, tab_2026 = st.tabs(["🏁 2025 Prediction", "🔮 2026 Prediction"])

    with tab_2025:
        render_2025_tab(
            model_table=model_table,
            top3_model=top3_model,
            winner_model=winner_model,
            points_model=points_model,
            probability_view=probability_view,
        )

    with tab_2026:
        render_2026_tab(
            model_table=model_table,
            openf1_meetings=openf1_meetings,
            top3_model=top3_model,
            winner_model=winner_model,
            points_model=points_model,
            probability_view=probability_view,
        )


if __name__ == "__main__":
    main()