import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def load_data(filepath):
    """Memuat data dari file CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan di: {filepath}")
    
    # Membaca data, asumsikan separator koma untuk Heart Disease dataset
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """Melakukan preprocessing: cleaning, encoding, dan scaling."""
    df_processed = df.copy()

    print("1. Handling Missing Values...")
    # Mengisi nilai kosong dengan rata-rata (hanya kolom numerik)
    # numeric_only=True digunakan untuk menghindari error jika ada kolom string
    numeric_means = df_processed.mean(numeric_only=True)
    df_processed.fillna(numeric_means, inplace=True)

    print("2. Encoding Data Kategorikal...")
    # Daftar kolom yang biasanya kategorikal di dataset Heart Disease
    # cp: chest pain type, restecg: resting ecg results, slope: slope of the peak exercise, thal: thalassemia
    # sex, fbs, exang, ca juga bisa dianggap kategori tapi kadang dibiarkan biner/ordinal
    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    
    # Tambahkan kolom bertipe object (teks) jika ada
    object_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    # Gabungkan list kolom yang perlu di-encode
    cols_to_encode = list(set(object_cols + [c for c in categorical_cols if c in df_processed.columns]))
    
    if cols_to_encode:
        print(f"   Melakukan One-Hot Encoding pada: {cols_to_encode}")
        df_processed = pd.get_dummies(df_processed, columns=cols_to_encode, drop_first=True)

    print("3. Scaling Data Numerik...")
    scaler = StandardScaler()
    
    # Kolom numerik kontinu yang perlu disamakan skalanya
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    # Pastikan kolom tersebut ada di dataset
    existing_num_cols = [col for col in numerical_cols if col in df_processed.columns]
    
    if existing_num_cols:
        print(f"   Melakukan Standard Scaling pada: {existing_num_cols}")
        df_processed[existing_num_cols] = scaler.fit_transform(df_processed[existing_num_cols])

    return df_processed

def save_data(df, output_path):
    """Menyimpan data yang sudah diproses ke file CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data berhasil disimpan di: {output_path}")

def main():
    # Lokasi file input (mentah) dan output (bersih)
    # Pastikan file 'heart.csv' sudah ada di folder data/raw/
    raw_data_path = '.././heart_disease/heart_disease_uci.csv'
    processed_data_path = 'data/processed/heart_processed.csv'

    print("=== Memulai Preprocessing Otomatis ===")
    
    try:
        # 1. Load Data
        df = load_data(raw_data_path)
        print(f"Data awal dimuat: {df.shape}")
        
        # 2. Preprocess Data
        df_clean = preprocess_data(df)
        print(f"Data setelah diproses: {df_clean.shape}")
        
        # 3. Save Data
        save_data(df_clean, processed_data_path)
        print("=== Preprocessing Selesai ===")
        
    except Exception as e:
        print(f"\nTERJADI KESALAHAN: {e}")
        print("Pastikan file 'heart.csv' sudah di-upload ke folder 'data/raw/'.")

if __name__ == "__main__":
    main()