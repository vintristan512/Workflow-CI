import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import logging
import argparse
import joblib

logging.basicConfig(level=logging.WARN)

# === PATH CONFIG ===
MLFLOW_TRACKING_URI = "file:mlruns"

EXPERIMENT_NAME = "Klasifikasi Diabetes - Basic Run"
RANDOM_SEED = 42
TARGET_COLUMN = "Outcome"
DATA_PATH = "diabetes_preprocessing.csv"


def run_basic_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load data
    try:
        df_clean = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        logging.warning(f"❌ Data tidak ditemukan di {DATA_PATH}")
        return

    # Split
    X = df_clean.drop(columns=[TARGET_COLUMN])
    y = df_clean[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Autolog
    mlflow.sklearn.autolog()

    model = LogisticRegression(random_state=RANDOM_SEED)
    print("🚀 Memulai pelatihan model dengan MLflow...")

    with mlflow.start_run(run_name="Basic_Logistic_Regression"):
        model.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"✅ Akurasi Test: {test_score:.4f}")

        # Log model ke MLflow
        mlflow.sklearn.log_model(model, "model_artefact")

        # Simpan juga sebagai artefak manual
        artifact_dir = os.path.join(os.path.dirname(__file__), "artifact")
        os.makedirs(artifact_dir, exist_ok=True)

        model_path = os.path.join(artifact_dir, "model.joblib")
        joblib.dump(model, model_path)

        mlflow.log_artifact(model_path)
        print(f"📦 Model tersimpan di {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="diabetes_preprocessing.csv")
    args = parser.parse_args()

    DATA_PATH = args.data_path
    run_basic_model()

