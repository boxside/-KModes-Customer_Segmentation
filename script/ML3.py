import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes

import pickle
from pathlib import Path

# import dataset
df = pd.read_csv('https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/customer_segments.txt', sep="\t")

# menampilkan data
print(df.head())
print(df.info())


sns.set(style='white')
plt.clf()

# Fungsi untuk membuat plot
def observasi_num(features):
	fig, axs = plt.subplots(2, 2, figsize=(10, 9))
	for i, kol in enumerate(features):
		sns.boxplot(df[kol], ax = axs[i][0])
		sns.distplot(df[kol], ax = axs[i][1])
		axs[i][0].set_title('mean = %.2f\n median = %.2f\n std = %.2f'%(df[kol].mean(), df[kol].median(), df[kol].std()))
		plt.setp(axs)
		plt.tight_layout()
		plt.show()

# Memanggil fungsi untuk membuat Plot untuk data numerik
kolom_numerik = ['Umur','NilaiBelanjaSetahun']
observasi_num(kolom_numerik)

# Menyiapkan kolom kategorikal
kolom_kategorikal = ['Jenis Kelamin', 'Profesi', 'Tipe Residen']

# Membuat canvas
fig, axs = plt.subplots(3, 1, figsize=(7, 10))

# Membuat plot untuk setiap kolom kategorikal
for i, kol in enumerate(kolom_kategorikal):
    # Membuat Plot
    sns.countplot(df[kol], order=df[kol].value_counts().index, ax=axs[i])
    axs[i].set_title('\nCount Plot %s\n' % (kol), fontsize=15)

    # Memberikan anotasi
    for p in axs[i].patches:
        axs[i].annotate(format(p.get_height(), '.0f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center',
                        va='center',
                        xytext=(0, 10),
                        textcoords='offset points')

        # Setting Plot
    sns.despine(right=True, top=True, left=True)
    axs[i].axes.yaxis.set_visible(False)
    plt.setp(axs)
    plt.tight_layout()

# Tampilkan plot
plt.show()

#standarized data

kolom_numerik = ['Umur','NilaiBelanjaSetahun']

# Statistik sebelum Standardisasi
print('Statistik Sebelum Standardisasi\n')
print(df[kolom_numerik ].describe().round(1))

# Standardisasi
df_std = StandardScaler().fit_transform(df[kolom_numerik])

# Membuat DataFrame
df_std = pd.DataFrame(data=df_std, index=df.index, columns=df[kolom_numerik].columns)

# Menampilkan contoh isi data dan summary statistic
print('Contoh hasil standardisasi\n')
print(df_std.head())

print('Statistik hasil standardisasi\n')
print(df_std.describe().round(0))

# Inisiasi nama kolom kategorikal
kolom_kategorikal = ['Jenis Kelamin', 'Profesi', 'Tipe Residen']

# Membuat salinan data frame
df_encode = df[kolom_kategorikal].copy()

# Melakukan labelEncoder untuk semua kolom kategorikal
for col in kolom_kategorikal:
    df_encode[col] = LabelEncoder().fit_transform(df_encode[col])

# Menampilkan data
print(df_encode.head())

# Menggabungkan data frame
df_model= df_encode.merge(df_std, left_index = True, right_index=True, how= 'left')
print(df_model.head())



# Melakukan Iterasi untuk Mendapatkan nilai Cost
cost = {}
for k in range(2, 10):
    kproto = KPrototypes(n_clusters=k, random_state=75)
    kproto.fit_predict(df_model, categorical=[0, 1, 2])
    cost[k] = kproto.cost_

# Memvisualisasikan Elbow Plot
sns.pointplot(x=list(cost.keys()), y=list(cost.values()))
plt.show()

kproto = KPrototypes(n_clusters=5, random_state = 75)
kproto = kproto.fit(df_model, categorical=[0,1,2])

#Save Model
pickle.dump(kproto, open('cluster.pkl', 'wb'))

# Menentukan segmen tiap pelanggan
clusters = kproto.predict(df_model, categorical=[0, 1, 2])
print('segmen pelanggan: {}\n'.format(clusters))

# Menggabungkan data awal dan segmen pelanggan
df_final = df.copy()
df_final['cluster'] = clusters
print(df_final.head())

# Menampilkan data pelanggan berdasarkan cluster nya
for i in range (0,5):
    print('\nPelanggan cluster: {}\n'.format(i))
    print(df_final[df_final['cluster']== i])

# Data Numerical
kolom_numerik = ['Umur', 'NilaiBelanjaSetahun']

for i in kolom_numerik:
    plt.figure(figsize=(6, 4))
    ax = sns.boxplot(x='cluster', y=i, data=df_final)
    plt.title('\nBox Plot {}\n'.format(i), fontsize=12)
    plt.show()

# Data Kategorikal
kolom_categorical = ['Jenis Kelamin', 'Profesi', 'Tipe Residen']

for i in kolom_categorical:
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df_final, x='cluster', hue=i)
    plt.title('\nCount Plot {}\n'.format(i), fontsize=12)
    ax.legend(loc="upper center")
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center',
                    va='center',
                    xytext=(0, 10),
                    textcoords='offset points')

    sns.despine(right=True, top=True, left=True)
    ax.axes.yaxis.set_visible(False)
    plt.show()

# Mapping nama kolom
df_final['Segmen'] = df_final['cluster'].map({
    0: 'Diamond Young Member',
    1: 'Diamond Senior Member',
    2: 'Silver Member',
    3: 'Gold Young Member',
    4: 'Gold Senior Member'
})

print(df_final.info())
print(df_final.head())


