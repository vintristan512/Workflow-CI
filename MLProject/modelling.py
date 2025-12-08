# modelling.py - KODE FINAL FIXED

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import logging
import argparse
import joblib   # <-- TAMBAHKAN

logging.basicConfig(level=logging.WARN)

# --- KONFIGURASI PATH DAN MLOPS ---

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MLFLOW_TRACKING_URI = "file:" + os.path.join(PROJECT_ROOT, "mlruns") 

EXPERIMENT_NAME = "Klasifikasi Diabetes - Basic Run"
RANDOM_SEED = 42
TARGET_COLUMN = 'Outcome'
DATA_PATH = 'diabetes_preprocessing.csv'

# --- FUNGSI UTAMA MODELLING DASAR ---
def run_basic_model():
    # Set MLflow Tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # [1] Load Data
    try:
        df_clean = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        logging.warning(f"Data bersih tidak ditemukan di {DATA_PATH}. Pastikan file ada.")
        return

    # [2] Split data
    X = df_clean.drop(columns=[TARGET_COLUMN])
    y = df_clean[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # [3] Enable autolog
    mlflow.sklearn.autolog()

    # [4] Train model
    model = LogisticRegression(random_state=RANDOM_SEED)

    print("Memulai pelatihan model dasar dengan Autolog...")

    with mlflow.start_run(run_name="Basic_Logistic_Regression"):
        
        model.fit(X_train, y_train) 
        test_score = model.score(X_test, y_test) 
        
        print(f"Pelatihan Selesai. Akurasi Test: {test_score:.4f}")
        
        # Logging model secara eksplisit
        mlflow.sklearn.log_model(model, "model_artefact")

        # --- 🔥 FIX: SIMPAN ARTEFAK DI PATH YANG PASTI TERDETEKSI OLEH CI ---
        artifact_dir = os.path.join(os.getcwd(), "artifact")
        os.makedirs(artifact_dir, exist_ok=True)
    
        model_path = os.path.join(artifact_dir, "model.joblib")
        joblib.dump(model, model_path)
    
        mlflow.log_artifact(model_path)
        print(f"Model artefak tersimpan di: {model_path}")
    
        print(f"Run berhasil dilog ke: {mlflow.get_tracking_uri()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="diabetes_preprocessing.csv")
    args = parser.parse_args()

    DATA_PATH = args.data_path
    run_basic_model()


