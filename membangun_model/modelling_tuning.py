import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import warnings
import os

# Abaikan peringatan agar output bersih
warnings.filterwarnings('ignore')

def load_data(path):
    """Memuat data bersih."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File data tidak ditemukan di: {path}")
    return pd.read_csv(path)

def main():
    print("Memulai run 'Skilled' dengan manual log dan tuning (Dataset: Heart Disease)...")

    # 1. Memuat dataset
    # Pastikan path ini sesuai dengan output preprocessing kamu
    data_path = './heart_disease/heart_processed.csv'
    
    try:
        df = load_data(data_path)
    except Exception as e:
        print(f"Error: {e}")
        print("Pastikan script preprocessing sudah dijalankan.")
        return

    # 2. Memisahkan fitur (X) dan target (y)
    # PENTING: Kolom target Heart Disease adalah 'target'
    target_col = 'target'
    
    if target_col not in df.columns:
        print(f"Error: Kolom target '{target_col}' tidak ditemukan di dataset.")
        return

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # 3. Membagi data training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Memulai MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # --- CHECKLIST 1: Menerapkan Hyperparameter Tuning ---
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000] # Diperbesar agar konvergen
        }

        base_model = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(estimator=base_model, 
                                   param_grid=param_grid, 
                                   cv=5, 
                                   scoring='accuracy', 
                                   n_jobs=-1)

        print("Menjalankan GridSearchCV...")
        grid_search.fit(X_train, y_train)

        # Dapatkan model dan parameter terbaik
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print(f"Best Parameters: {best_params}")

        # --- CHECKLIST 2: Log Parameter Secara Manual ---
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("tuning_cv_folds", 5)

        # 5. Evaluasi model terbaik
        print("Mengevaluasi model terbaik...")
        preds = best_model.predict(X_test)

        # Hitung metrik
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        print(f"Accuracy: {acc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-Score: {f1}")

        # --- CHECKLIST 3: Log Metrik Secara Manual ---
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        # Menambahkan score terbaik saat validasi silang
        mlflow.log_metric("best_cv_score", grid_search.best_score_)

        # 6. Log Model Secara Manual
        print("Menyimpan model ke MLflow...")
        mlflow.sklearn.log_model(best_model, "model")

    print("\nRun 'Skilled' selesai.")
    print(f"Cek run di MLflow UI dengan ID: {run_id}")

if __name__ == "__main__":
    main()