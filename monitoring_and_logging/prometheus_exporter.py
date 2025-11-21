import time
import random
import requests
import json
import pandas as pd
import os
from prometheus_client import start_http_server, Gauge, Counter, Histogram

print("Memulai Prometheus Exporter (Mode Data Asli)...")

# --- PATH KE DATA ASLI ---
# Pastikan path ini benar!
DATA_PATH =".././preprocessing/namadataset_preprocessing/wine_processed.csv"
# --- DEKLARASI METRIK ---
PREDICTION_COUNT = Counter('wine_prediction_total', 'Total prediksi yang dibuat')
PREDICTION_LATENCY = Histogram('wine_prediction_latency_seconds', 'Latensi prediksi (detik)')
LAST_ALCOHOL = Gauge('wine_feature_last_alcohol', 'Nilai fitur alcohol terakhir')
PREDICTION_RESULT = Counter('wine_prediction_result_total', 'Total prediksi berdasarkan kelas', ['class'])
HTTP_ERRORS = Counter('wine_http_errors_total', 'Total HTTP error saat prediksi')
FEATURE_GAUGES = {
    "fixed acidity": Gauge('wine_feature_fixed_acidity', 'Nilai fitur fixed acidity'),
    "volatile acidity": Gauge('wine_feature_volatile_acidity', 'Nilai fitur volatile acidity'),
    "citric acid": Gauge('wine_feature_citric_acid', 'Nilai fitur citric acid'),
    "residual sugar": Gauge('wine_feature_residual_sugar', 'Nilai fitur residual sugar'),
    "chlorides": Gauge('wine_feature_chlorides', 'Nilai fitur chlorides'),
    "free sulfur dioxide": Gauge('wine_feature_free_sulfur_dioxide', 'Nilai fitur free sulfur dioxide'),
    "total sulfur dioxide": Gauge('wine_feature_total_sulfur_dioxide', 'Nilai fitur total sulfur dioxide'),
    "density": Gauge('wine_feature_density', 'Nilai fitur density'),
    "pH": Gauge('wine_feature_pH', 'Nilai fitur pH'),
    "sulphates": Gauge('wine_feature_sulphates', 'Nilai fitur sulphates'),
    "alcohol": Gauge('wine_feature_alcohol', 'Nilai fitur alcohol')
}

# Endpoint model (TANPA DOCKER)
MODEL_ENDPOINT = "http://localhost:5001/invocations" 
COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

def load_real_data(path):
    """Memuat data wine bersih dari CSV."""
    try:
        df = pd.read_csv(path)
        df_features = df.drop("quality_category", axis=1)
        df_features = df_features[COLUMNS] 
        print(f"Berhasil memuat {len(df_features)} baris data fitur asli dari {path}")
        return df_features
    except Exception as e:
        print(f"Error saat memuat data: {e}")
        return None

def get_real_features(df_features):
    """Mengambil satu baris acak dari data asli."""
    random_row = df_features.sample(n=1)
    features_dict = random_row.iloc[0].to_dict()

    for key, gauge in FEATURE_GAUGES.items():
        gauge.set(features_dict.get(key, 0))

    LAST_ALCOHOL.set(features_dict.get("alcohol", 0))
    return [list(features_dict.values())]

def predict_and_log(df_features):
    """Mengirim request ke model dan mencatat metrik."""
    features = get_real_features(df_features) 

    payload = {
        "dataframe_split": {
            "columns": COLUMNS,
            "data": features
        }
    }

    try:
        with PREDICTION_LATENCY.time():
            response = requests.post(MODEL_ENDPOINT, json=payload, headers={"Content-Type": "application/json"})

        if response.status_code == 200:
            PREDICTION_COUNT.inc()
            result = response.json()
            prediction_value = result['predictions'][0]
            prediction_class = "Baik" if prediction_value == 1 else "Buruk"
            PREDICTION_RESULT.labels(prediction_class).inc()
            print(f"Prediksi sukses (data asli): {prediction_class}")
        else:
            HTTP_ERRORS.inc()
            print(f"HTTP Error: {response.status_code} - {response.text}")

    except requests.exceptions.ConnectionError:
        HTTP_ERRORS.inc()
        print("Error: Tidak bisa terhubung ke model server di port 5001.")

if __name__ == '__main__':
    df_wine_features = load_real_data(DATA_PATH)

    if df_wine_features is not None:
        start_http_server(8000)
        print("Exporter berjalan di http://localhost:8000")
        print("Mengirim data inferensi ASLI ke model...")

        while True:
            predict_and_log(df_wine_features)
            time.sleep(random.uniform(1, 5))
    else:
        print("Exporter gagal dimulai karena data wine tidak dapat dimuat.")