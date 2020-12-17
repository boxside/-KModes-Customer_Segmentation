import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes

import pickle
from pathlib import Path

# Data Baru
data = [{
    'Customer_ID': 'CUST-100',
    'Nama Pelanggan': 'Joko',
    'Jenis Kelamin': 'Pria',
    'Umur': 45,
    'Profesi': 'Wiraswasta',
    'Tipe Residen': 'Cluster',
    'NilaiBelanjaSetahun': 8230000

}]

# Membuat Data Frame
new_df = pd.DataFrame(data)

# Melihat Data
print(new_df)

def data_preprocess(data):
    # Konversi Kategorikal data
    kolom_kategorikal = ['Jenis Kelamin', 'Profesi', 'Tipe Residen']

    df_encode = data[kolom_kategorikal].copy()

    ## Jenis Kelamin
    df_encode['Jenis Kelamin'] = df_encode['Jenis Kelamin'].map({
        'Pria': 0,
        'Wanita': 1
    })

    ## Profesi
    df_encode['Profesi'] = df_encode['Profesi'].map({
        'Ibu Rumah Tangga': 0,
        'Mahasiswa': 1,
        'Pelajar': 2,
        'Professional': 3,
        'Wiraswasta': 4
    })

    ## Tipe Residen
    df_encode['Tipe Residen'] = df_encode['Tipe Residen'].map({
        'Cluster': 0,
        'Sector': 1
    })

    # Standardisasi Numerical Data
    kolom_numerik = ['Umur', 'NilaiBelanjaSetahun']
    df_std = data[kolom_numerik].copy()

    ## Standardisasi Kolom Umur
    df_std['Umur'] = (df_std['Umur'] - 37.5) / 14.7

    ## Standardisasi Kolom Nilai Belanja Setahun
    df_std['NilaiBelanjaSetahun'] = (df_std['NilaiBelanjaSetahun'] - 7069874.8) / 2590619.0

    # Menggabungkan Kategorikal dan numerikal data
    df_model = df_encode.merge(df_std, left_index=True,
                               right_index=True, how='left')

    return df_model


# Menjalankan fungsi
new_df_model = data_preprocess(new_df)

print(new_df_model)
