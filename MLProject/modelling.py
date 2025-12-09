import argparse
import os
import mlflow
from mlflow import MlflowClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# ==========================
# 1. MLflow Tracking Setup
# ==========================

# Pastikan path SQLite absolut (FIX PENTING)
tracking_db_path = os.path.abspath("mlflow.db")
mlflow.set_tracking_uri("sqlite:///" + tracking_db_path)

# Pastikan experiment tidak hilang
EXPERIMENT_NAME = "Klasifikasi Diabetes - CI"
client = MlflowClient()

if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

# ==========================
# 2. Fungsi training
# ==========================

def run_basic(data_path):
    print("🚀 Start MLflow training (CI-stable)")

    # Load data
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    accuracy = model.score(X_test_scaled, y_test)
    
    # ======================
    # FIX RUN-ID ERROR
    # ======================
    run = client.create_run(experiment_id=experiment_id)
    mlflow.start_run(run_id=run.info.run_id)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)

    # Save model
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.joblib")
    mlflow.log_artifact("model/model.joblib")

    mlflow.end_run()
    print(f"🎉 Training selesai. Accuracy: {accuracy}")

# ==========================
# 3. Main
# ==========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    run_basic(args.data_path)
