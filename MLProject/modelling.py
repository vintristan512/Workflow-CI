# modelling.py - KODE FINAL FIXED

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import logging
import argparse


logging.basicConfig(level=logging.WARN)

# --- KONFIGURASI PATH DAN MLOPS ---

# 1. Mendapatkan path absolut ke folder root proyek (D:\SMSML_Kevin-Tristan)
# Ini menjamin MLflow selalu menulis ke root proyek, terlepas dari di mana script dijalankan.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MLFLOW_TRACKING_URI = "file:" + os.path.join(PROJECT_ROOT, "mlruns") 

EXPERIMENT_NAME = "Klasifikasi Diabetes - Basic Run"
RANDOM_SEED = 42
TARGET_COLUMN = 'Outcome'
DATA_PATH = 'diabetes_preprocessing.csv' # Asumsi data berada di folder Membangun_Model

# --- FUNGSI UTAMA MODELLING DASAR ---
def run_basic_model():
    # Set MLflow Tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # [1] Muat Data Bersih
    try:
        # Load data dari folder yang sama
        df_clean = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        logging.warning(f"Data bersih tidak ditemukan di {DATA_PATH}. Pastikan file ada di folder Membangun_Model.")
        return

    # [2] Split data bersih
    X = df_clean.drop(columns=[TARGET_COLUMN])
    y = df_clean[TARGET_COLUMN]
    
    # OUTPUT DARI SPLIT DISIMPAN DI VARIABEL INI
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # [3] ENABLE AUTOLOG (Wajib Basic Kriteria 2)
    mlflow.sklearn.autolog() 

    # [4] Pelatihan Model
    model = LogisticRegression(random_state=RANDOM_SEED)

    print("Memulai pelatihan model dasar dengan Autolog...")
    
    with mlflow.start_run(run_name="Basic_Logistic_Regression"):
        
        # MODEL FIT MENGGUNAKAN VARIABEL YANG BARU DIDEFINISIKAN
        model.fit(X_train, y_train) 
        
        test_score = model.score(X_test, y_test) 
        
        print(f"Pelatihan Selesai. Akurasi Test: {test_score:.4f}")
        
        # Logging model secara eksplisit
        mlflow.sklearn.log_model(model, "model_artefact")
        
        print(f"Run berhasil dilog ke: {mlflow.get_tracking_uri()}") 
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="diabetes_preprocessing.csv")
    args = parser.parse_args()

    DATA_PATH = args.data_path
    run_basic_model()
