import pandas as pd
import os
import argparse

# --- PATH DEFINITIONS ---
# Menggunakan os.path.abspath(__file__) untuk mendapatkan path absolut ke skrip ini
# Ini membuatnya berfungsi baik secara lokal maupun di GitHub Actions
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Jika __file__ tidak terdefinisi (misalnya, dijalankan di lingkungan interaktif tertentu)
    BASE_DIR = os.path.abspath(os.getcwd())

# Path ke root direktori proyek (satu level di atas folder 'preprocessing')
ROOT_DIR = os.path.dirname(BASE_DIR) 

# Path ke data mentah
RAW_DATA_PATH = os.path.join(ROOT_DIR, 'wine_quality', 'winequality-red.csv')

# Path untuk menyimpan data bersih
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'namadataset_preprocessing')
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'wine_processed.csv')


def load_data(path):
    """Memuat data dari file CSV."""
    print(f"Memuat data mentah dari: {path}")
    if not os.path.exists(path):
        print(f"Error: File tidak ditemukan di {path}")
        return None
    
    # Dataset ini menggunakan titik koma (;) sebagai pemisah
    return pd.read_csv(path, sep=';')

def preprocess_data(df):
    """
    Menerapkan preprocessing pada data:
    1. Membuat target biner 'quality_category'.
    2. Menghapus kolom 'quality' asli.
    """
    print("Memulai preprocessing data...")
    
    df_processed = df.copy()
    
    # 1. Membuat kolom target biner ('quality_category')
    # Kualitas > 5 dianggap 'baik' (1), sisanya 'buruk' (0)
    df_processed['quality_category'] = df_processed['quality'].apply(lambda x: 1 if x > 5 else 0)

    # 2. Menghapus kolom 'quality' asli
    df_processed = df_processed.drop('quality', axis=1)
    
    print("Preprocessing selesai.")
    return df_processed

def save_data(df, path):
    """Menyimpan dataframe yang telah diproses ke CSV."""
    
    # Pastikan direktori (folder) untuk menyimpan file ada
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    print(f"Menyimpan data bersih ke: {path}")
    df.to_csv(path, index=False)
    print("Data bersih berhasil disimpan.")

def main(args):
    """Fungsi utama untuk menjalankan alur kerja."""
    print("Menjalankan skrip preprocessing otomatis...")
    
    # Gunakan path dari argumen jika ada, jika tidak, gunakan path default
    raw_path = args.input if args.input else RAW_DATA_PATH
    processed_path = args.output if args.output else PROCESSED_DATA_PATH
    
    df_raw = load_data(raw_path)
    
    if df_raw is not None:
        df_clean = preprocess_data(df_raw)
        save_data(df_clean, processed_path)
        print("Skrip preprocessing otomatis selesai.")
    else:
        print("Skrip dihentikan karena data mentah tidak ditemukan.")

if __name__ == "__main__":
    # Menambahkan argparse untuk fleksibilitas (opsional tapi bagus)
    parser = argparse.ArgumentParser(description="Script Preprocessing Data Wine.")
    parser.add_argument('-i', '--input', type=str, help=f"Path ke file data mentah. Default: {RAW_DATA_PATH}")
    parser.add_argument('-o', '--output', type=str, help=f"Path untuk menyimpan data bersih. Default: {PROCESSED_DATA_PATH}")
    
    args = parser.parse_args()
    main(args)