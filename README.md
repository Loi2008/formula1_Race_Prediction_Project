## Formula 1 Race Outcome Prediction

### End-to-End Machine Learning Pipeline | MLOps + Sports Analytics
An end-to-end machine learning system that predicts Formula 1 race winners, podium finishes, and driver points using multi-source motorsport data. The project demonstrates production-style ML workflows including data ingestion, feature engineering, model training, evaluation, and deployment via an interactive dashboard.

### Key Highlights
- Built a full ML pipeline from raw data ingestion to live predictions
- Integrated multiple real-world APIs (FastF1, OpenF1, Jolpica)
- Engineered race-level and driver-level features from lap telemetry, weather, and historical results

### Trained models to predict:
- Race winner
- Top 3 finishers
- Driver points outcomes

Developed an interactive Streamlit dashboard for visualization
Structured project using modular, production-style architecture

### Skills Demonstrated
- Machine Learning: Classification, model evaluation, feature engineering
- Data Engineering: ETL pipelines, data cleaning, schema alignment
- Python Ecosystem: Pandas, NumPy, Scikit-learn
- MLOps Practices: Modular code, artifacts tracking, reproducibility
- Visualization & Apps: Streamlit
- Data Sources: APIs, CSV/Parquet pipelines

### Architecture Overview
```
Raw Data → Ingestion → Cleaning → Feature Engineering → Model Training → Evaluation → Predictions → Dashboard
```

### Project Structure (Simplified)
```
FORMULA1_RACE_PREDICTION/
│
├── app/
│   ├── assets/                  # Images and static assets
│   └── streamlit_app.py        # Streamlit dashboard
│
├── artifacts/
│   ├── metrics/                # Model evaluation metrics (JSON)
│   ├── *.pkl                   # Trained models
│   ├── future_*                # Future race predictions
│   └── predicted_*             # Historical predictions
│
├── config/                     # Configuration files (if used)
│
├── data/
│   ├── processed/              # Cleaned + feature-engineered datasets
│   └── raw/                    # Raw ingested datasets
│       ├── fastf1/
│       ├── openf1/
│       └── jolpica/
│
├── notebooks/                  # Exploratory notebooks
│
├── sql/                        # SQL scripts (if applicable)
│
├── src/
│   ├── features/
│   │   ├── build_features.py
│   │   └── build_future_features.py
│   │
│   ├── ingestion/
│   │   ├── ingest_fastf1.py
│   │   ├── ingest_openf1.py
│   │   └── ingest_jolpica.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── train_points.py
│   │   ├── predict_podium.py
│   │   ├── predict_future_race.py
│   │   └── evaluate.py
│   │
│   ├── processing/
│   │   └── clean_data.py
│   │
│   └── utils/
│
├── tests/                      # Unit tests
│
├── requirements.txt
└── README.md
```

### End-to-End Workflow
1. Data Ingestion
    - Collects race data from multiple APIs
    - Stores raw datasets for reproducibility
2. Data Processing
    - Cleans and standardizes datasets
    - Handles missing values and inconsistencies
3. Feature Engineering
    - Builds driver performance metrics
    - Extracts lap-time trends and race conditions
Generates future race features for prediction
4. Model Training
    - Trains multiple models for:
    - Winner classification
    - Podium prediction
    - Points prediction
5. Evaluation
    - Tracks performance using stored metrics
    - Outputs structured JSON reports
6. Prediction
    - Generates predictions for historical and future races
7. Visualization
    - Streamlit dashboard for interactive exploration

### Sample Outputs
- Predicted race winners and podiums
- Driver rankings for upcoming races
- Model performance metrics (accuracy, etc.)

### Stored in:
```
artifacts/
```

### Running the Project
#### Setup
```
git clone https://github.com/Loi2008/formula1_Race_Prediction_Project.git
cd formula1_race_prediction
pip install -r requirements.txt
```

#### Run Pipeline
```
python src/ingestion/ingest_fastf1.py
python src/processing/clean_data.py
python src/features/build_features.py
python src/models/train.py
```

#### Launch Dashboard
```
streamlit run app/streamlit_app.py
```

The URL
```
https://formula1racepredictionproject-bpswtijvr94sgxs3lxvc5p.streamlit.app/
```

### What Makes This Project Stand Out
- Real-world, messy data integration
- End-to-end ML lifecycle (not just modeling)
- Modular, scalable codebase
- Combines sports analytics + ML engineering
- Ready for deployment extensions

### Further Improvements
- Real-time race prediction pipeline
- Dockerized deployment
- Cloud integration (AWS/GCP)
