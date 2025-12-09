import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import argparse

TARGET_COLUMN = "Outcome"

def run_basic():
    print("🚀 Start MLflow training")

    # Gunakan SQLite supaya CI stabil
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ci-experiment")

    df = pd.read_csv("diabetes_preprocessing.csv")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()

    with mlflow.start_run():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        mlflow.log_metric("accuracy", score)
        mlflow.sklearn.log_model(model, "model")

        print("Akurasi:", score)

if __name__ == "__main__":
    run_basic()
