# modelling.py - CI-stable version
import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import joblib

TARGET_COLUMN = "Outcome"

def run_basic(data_path):
    print("🚀 Start MLflow training (CI-stable)")

    # 1) Gunakan SQLite backend supaya MLflow tidak pakai file store yang kadang kacau di CI
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Klasifikasi Diabetes - CI")

    # 2) Load data (pastikan file ada di MLProject/)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")
    df = pd.read_csv(data_path)

    # 3) Split
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4) Train simple model
    model = LogisticRegression(max_iter=1000)

    with mlflow.start_run(run_name="Basic_Logistic_Regression"):
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        mlflow.log_metric("accuracy", float(score))
        mlflow.sklearn.log_model(model, "model")

        print(f"Akurasi test: {score:.4f}")

        # 5) Simpan artefak juga agar GitHub Action bisa upload
        artifact_dir = os.path.join(os.path.dirname(__file__), "artifact")
        os.makedirs(artifact_dir, exist_ok=True)
        model_path = os.path.join(artifact_dir, "model.joblib")
        joblib.dump(model, model_path)
        print(f"Model tersimpan: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="diabetes_preprocessing.csv")
    args = parser.parse_args()
    run_basic(args.data_path)
