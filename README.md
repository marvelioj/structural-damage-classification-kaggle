# Structural Damage Classification

## Gambaran Umum

Proyek ini memprediksi tingkat kerusakan bangunan (`damage_grade`, kelas 1-5) akibat gempa Gorkha, Nepal (25 April 2015, Mw 7,8). Data berasal dari survei pasca-gempa Central Bureau of Statistics (CBS) Nepal dan berisi karakteristik struktural bangunan sebelum gempa serta kerusakan aktualnya. Proyek ini adalah Tugas 1 Supervised Machine Learning dengan metrik **macro F1-score**; model neural network tidak digunakan.

## Struktur Proyek

Struktur direktori berikut menggambarkan susunan proyek yang digunakan:

``` text
.
├── README.md
├── .gitignore
├── requirements.txt
├── data/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_submission.ipynb
├── scripts/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
├── models/
├── submissions/
└── reports/
    └── figures/
```

- `data/`: Berisi dataset yang diunduh dari Kaggle dan disimpan secara lokal.
- `notebooks/`: Berisi eksplorasi data, eksperimen pemodelan, dan pembuatan submission.
- `scripts/`: Berisi kode yang dapat digunakan kembali untuk preprocessing, feature engineering, pelatihan, dan prediksi.
- `models/`: Berisi pipeline yang telah dilatih serta parameter penyesuaian kelas.
- `submissions/`: Berisi berkas prediksi yang dibuat untuk Kaggle.
- `reports/figures/`: Berisi visualisasi yang dihasilkan selama evaluasi model.

## Dataset

Kompetisi: [structural-damage-classification](https://www.kaggle.com/competitions/structural-damage-classification). Klik "Join Competition" dan terima aturannya terlebih dahulu, lalu unduh data dengan salah satu cara berikut:

- Manual: tab **Data** > **Download All**, ekstrak ke folder `data/`.
- Kaggle API: `kaggle competitions download -c structural-damage-classification -p data` lalu ekstrak zip-nya di `data/`.

Folder `data/` harus berisi `train.csv`, `test.csv`, dan `sample_submission.csv`. Berkas CSV tidak di-commit (lihat `.gitignore`).

## Instalasi

Proyek ini dikembangkan menggunakan Python 3.12.

Buat dan aktifkan virtual environment dari direktori utama repositori:

``` bash
python -m venv .venv
```

Di Windows, aktifkan dengan:

``` powershell
.venv\Scripts\activate
```

Instal dependensi yang diperlukan:

``` bash
python -m pip install -r requirements.txt
```

Pastikan seluruh berkas dataset sudah ditempatkan di dalam direktori `data/` sebelum menjalankan script.

## Alur Kerja

### 1\. Eksplorasi Data

Notebook `notebooks/01_eda.ipynb` digunakan untuk mengeksplorasi dataset, termasuk distribusi tingkat kerusakan, karakteristik bangunan, serta kemungkinan hubungan antara fitur dan variabel target.

### 2\. Preprocessing dan Feature Engineering

Preprocessing dan feature engineering diimplementasikan dalam beberapa modul Python yang dapat digunakan kembali.

- `scripts/preprocess.py` menangani parsing target, pembuatan fitur dasar, persiapan data kategorikal, dan `ColumnTransformer`.
- `scripts/feature_engineering.py` membuat fitur tambahan berdasarkan bangunan tetangga dan material konstruksi, termasuk fitur label propagation.
- `scripts/train.py` menggabungkan transformasi dan classifier ke dalam scikit-learn `Pipeline`.

`Pipeline` memastikan bahwa preprocessing dan pelatihan model berjalan sebagai satu rangkaian. Dengan demikian, transformasi dapat di-fit hanya pada data pelatihan di setiap fold ketika cross-validation dilakukan, sehingga risiko data leakage dari proses preprocessing dapat dikurangi.

Fitur yang bergantung pada label dibuat menggunakan pendekatan out-of-fold selama pelatihan untuk mengurangi target leakage. Fitur tersebut tidak boleh dibuat dengan menggunakan label dari baris yang sedang diprediksi.

### 3\. Pelatihan dan Evaluasi Model

Jalankan pipeline pelatihan dari direktori utama repositori:

``` bash
python -m scripts.train
```

Alur pelatihan melakukan cross-validation, mengevaluasi prediksi menggunakan macro F1-score, menyesuaikan offset keputusan kelas, melakukan fitting pipeline final pada seluruh data pelatihan yang tersedia, lalu menyimpan artefak model.

Classifier yang digunakan adalah **LightGBM**, yaitu model gradient boosting. Proyek ini tidak menggunakan neural network.

Notebook `notebooks/02_modeling.ipynb` mendokumentasikan proses evaluasi dan eksperimen pemodelan.

### 4\. Prediksi dan Pembuatan Submission

Setelah proses pelatihan selesai, buat prediksi dengan menjalankan:

``` bash
python -m scripts.predict
```

Script tersebut memuat artefak model yang telah disimpan dan menghasilkan:

``` text
submissions/submission.csv
```

Berkas keluaran mengikuti urutan baris dan struktur kolom pada `sample_submission.csv`. Berkas tersebut dapat diunggah ke Kaggle untuk memperoleh skor leaderboard.

Notebook untuk proses submission tersedia di `notebooks/03_submission.ipynb`.

## Hasil

| Evaluasi | Skor |
| --- | --- |
| OOF macro F1, argmax (pipeline) | 0,5854 |
| OOF macro F1, dengan offset kelas (pipeline) | 0,5908 |
| Public / Private, pipeline (`scripts.predict`) | 0,59283 / 0,59637 |
| Public / Private, notebook awal (submission terbaik) | 0,59617 / 0,59905 |

Submission terbaik di leaderboard berasal dari notebook awal (OOF 0,5921) yang menghitung fitur berbasis label dengan array fold manual. Setelah kode dirapikan menjadi `Pipeline` dan `ColumnTransformer`, hasilnya turun sekitar 0,003 pada leaderboard dan 0,001 pada OOF. Dugaan penyebabnya (belum diuji satu per satu): prediksi test kini berasal dari satu model, bukan rata-rata lima model fold; offset disetel pada model 80% data tetapi dipakai pada model 100% data; jumlah pohon dibuat tetap (230) tanpa early stopping; dan fitur propagasi pada data latih dihitung dari graf tanpa baris test.

## Interpretasi Model

- **Ketidakseimbangan kelas.** Grade 5 mencakup sekitar 36% data, Grade 1 hanya sekitar 10%. Karena metriknya macro F1, model memakai class weight dan offset per kelas pada log-probabilitas. Offset menaikkan OOF macro F1 dari 0,5854 menjadi 0,5908 (+0,0054), artinya sebagian kelas minoritas memang kurang diprediksi oleh argmax biasa.
- **Lokasi lewat `building_id`.** Pada EDA, grade bangunan dengan ID bersebelahan berkorelasi (r sekitar 0,58). Karena itu dibuat fitur tetangga (rata-rata, simpangan baku, dan proporsi grade dari 5 bangunan berlabel terdekat) dan fitur label propagation yang menyebarkan label lewat graf ID sehingga jangkauannya lebih jauh.
- **`weakest_link`.** Setiap kategori material diurutkan berdasarkan rata-rata kerusakan, dan skor sebuah bangunan adalah peringkat tertinggi dari kelima kolom material (bangunan hanya sekuat komponen terlemahnya). Hasil kalinya dengan rata-rata grade tetangga menggambarkan bangunan rapuh yang berada di area dengan guncangan kuat.
- **Tanpa kebocoran label.** Semua fitur berbasis label dihitung out-of-fold di dalam `Pipeline` (`fit_transform` memakai lipatan internal, `transform` memakai seluruh label latih), sehingga label sebuah baris tidak pernah dipakai untuk fiturnya sendiri.
- **Co-training ditolak.** Percobaan menambahkan prediksi test sebagai seed propagasi menaikkan OOF tetapi menurunkan skor public, tanda model mulai mempercayai prediksinya sendiri. Versi tersebut tidak dipakai.
- **Feature importance dan confusion matrix:** (diisi dari `reports/figures`, lihat di bawah.)

Hubungan prediktif ini tidak membuktikan hubungan sebab-akibat.

## Reproduksibilitas

Untuk menjalankan kembali seluruh alur kerja:

1. Unduh dataset dan simpan berkas CSV yang diperlukan di dalam `data/`.
2. Instal dependensi dari `requirements.txt`.
3. Jalankan `python -m scripts.train` untuk melatih model dan menyimpan artefaknya.
4. Jalankan `python -m scripts.predict` untuk menghasilkan berkas submission.
5. Unggah `submissions/submission.csv` ke Kaggle untuk memperoleh skor leaderboard.

Artefak model yang telah disimpan mungkin tidak disertakan dalam version control karena ukurannya. Jika artefak tersebut tidak tersedia, jalankan script pelatihan terlebih dahulu sebelum membuat prediksi.

## Keterbatasan

- Performa model pada validasi dapat berbeda dari performanya pada leaderboard Kaggle.
- Fitur berbasis tetangga dan label propagation bergantung pada hubungan yang tersedia melalui identifier bangunan serta data pelatihan.
- Adjusted class offsets dibuat berdasarkan prediksi validasi dan perlu dibandingkan dengan baseline tanpa penyesuaian.
- Model hanya mengidentifikasi hubungan prediktif dalam dataset; model tidak membuktikan pengaruh kausal karakteristik bangunan tertentu terhadap kerusakan akibat gempa.
- Skor pipeline akhir sedikit di bawah notebook awal (selisih sekitar 0,003 pada leaderboard).
- Offset kelas disetel pada prediksi OOF yang sama dengan yang dipakai untuk evaluasi, sehingga skor OOF dengan offset sedikit optimistis.

## Referensi

- [Kaggle Structural Damage Classification](<https://www.kaggle.com/competitions/structural-damage-classification>)
- [Dokumentasi LightGBM](<https://lightgbm.readthedocs.io/>)
- [Dokumentasi Scikit-learn](<https://scikit-learn.org/stable/>)