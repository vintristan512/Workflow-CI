# modelling.py - Kriteria 2: Basic (Autolog) - KODE FINAL
import pandas as pd
import numpy as np
import os
import sys
import logging

# [1] IMPORT DEPENDENSI KRITERIA 1 (Preprocessing)
try:
    # Mengarahkan Python untuk mencari automate_Kevin_Tristan.py di folder Kriteria 1
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Eksperimen_SML_Kevin Tristan', 'preprocessing')) 
    from automate_Kevin_Tristan import get_preprocessor
except ImportError:
    logging.warning("Gagal mengimpor automate_Kevin-Tristan. Pastikan file berada di path yang benar.")
    sys.exit(1)


# [2] IMPORT MLOPS TOOLS
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# --- KONFIGURASI UMUM ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000" 
EXPERIMENT_NAME = "Klasifikasi Diabetes - Basic Run"
RANDOM_SEED = 42

logging.basicConfig(level=logging.WARN)

# --- FUNGSI UTAMA MODELLING DASAR ---
def run_basic_model():
    # Set MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # [1] Muat Data Mentah (PATH AKHIR YANG DIJAMIN BENAR)
    RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'Eksperimen_SML_Kevin Tristan', 'diabetes_raw.csv')
    try:
        df_raw = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        logging.warning(f"Data mentah tidak ditemukan di {RAW_DATA_PATH}.")
        return

    # [2] Split data mentah
    X_raw = df_raw.drop(columns=['Outcome'])
    y_raw = df_raw['Outcome']
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=RANDOM_SEED, stratify=y_raw
    )
    
    # [3] ENABLE AUTOLOG (Wajib Basic Kriteria 2)
    mlflow.sklearn.autolog() 

    # [4] Membuat Pipeline dan Pelatihan
    preprocessor = get_preprocessor()
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=RANDOM_SEED))
    ])

    print("Memulai pelatihan model dasar dengan Autolog...")
    
    with mlflow.start_run(run_name="Basic_Logistic_Regression"):
        model_pipeline.fit(X_train_raw, y_train_raw)
        
        test_score = model_pipeline.score(X_test_raw, y_test_raw)
        print(f"Pelatihan Selesai. Akurasi Test: {test_score:.4f}")


if __name__ == '__main__':
    run_basic_model()
