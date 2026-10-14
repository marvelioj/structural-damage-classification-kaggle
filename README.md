# Klasifikasi Kerusakan Bangunan Akibat Gempa

## Gambaran Umum

Proyek ini mengembangkan model machine learning untuk memprediksi tingkat kerusakan bangunan akibat gempa Nepal tahun 2015. Model menggunakan karakteristik bangunan, material konstruksi, dan informasi terkait lokasi untuk mengklasifikasikan setiap bangunan ke dalam salah satu dari lima tingkat kerusakan.

Dataset yang digunakan berasal dari kompetisi Kaggle **Richter's Predictor: Modeling Earthquake Damage**.

**Tujuan:** Memprediksi `damage_grade` untuk setiap bangunan pada dataset pengujian dan menghasilkan berkas submission Kaggle yang valid.

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

Unduh dataset kompetisi dari [halaman kompetisi Kaggle](<https://www.kaggle.com/competitions/richter-s-predictor-modeling-earthquake-damage>).

Simpan berkas berikut di dalam direktori `data/`:

``` text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

Data pelatihan berisi karakteristik bangunan dan variabel target `damage_grade`. Data pengujian berisi bangunan yang tingkat kerusakannya harus diprediksi. Berkas `sample_submission.csv` menunjukkan format keluaran yang diwajibkan.

Dataset tidak disertakan dalam repositori ini. Berkas CSV dataset dikecualikan melalui `.gitignore` agar berkas berukuran besar tidak ikut di-commit.

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

Metrik evaluasi utama yang digunakan adalah **macro F1-score**, yang memberikan bobot yang sama kepada F1-score dari setiap tingkat kerusakan.

| Evaluasi | Skor |
| --- | ---: |
| Out-of-fold macro F1 dengan argmax | \[Isi dengan skor aktual\] |
| Out-of-fold macro F1 dengan adjusted offsets | \[Isi dengan skor aktual\] |
| Skor public leaderboard Kaggle | \[Isi dengan skor aktual\] |
| Skor private leaderboard Kaggle | Menunggu hasil akhir leaderboard |

Skor out-of-fold (OOF) dihitung dari prediksi untuk sampel pelatihan yang tidak digunakan saat model pada fold terkait dilatih. Oleh karena itu, skor OOF memberikan estimasi validasi tanpa mengevaluasi setiap sampel menggunakan model yang dilatih langsung dengan sampel tersebut.

Skor adjusted offset mengukur dampak perubahan threshold keputusan kelas berdasarkan prediksi out-of-fold. Hasilnya perlu dibandingkan dengan prediksi argmax tanpa penyesuaian untuk menentukan apakah penyesuaian tersebut benar-benar meningkatkan macro F1-score.

## Interpretasi Model

Model menggabungkan karakteristik bangunan dengan fitur tambahan yang dirancang untuk menangkap kerentanan struktur dan pola kerusakan lokal.

- **Karakteristik bangunan:** Fitur seperti usia bangunan, jumlah lantai, luas plinth, dan tinggi bangunan menggambarkan kondisi fisik yang mungkin berkaitan dengan performa struktur.
- **Material konstruksi:** Informasi mengenai fondasi, atap, lantai dasar, dan superstruktur digunakan sebagai indikator konstruksi serta kemungkinan kerentanan struktur.
- **Fitur berbasis tetangga:** Identifier bangunan yang berdekatan digunakan untuk memperoleh ringkasan tingkat kerusakan lokal dan merepresentasikan pola spasial.
- **Fitur weakest-link:** Fitur yang berkaitan dengan material digabungkan menjadi indikator kerentanan struktur, kemudian dikombinasikan dengan interaksi antara indikator tersebut dan rata-rata tingkat kerusakan lingkungan sekitar.
- **Fitur label propagation:** Hubungan antarbangunan digunakan untuk menghasilkan fitur tambahan yang berkaitan dengan kelas. Selama pelatihan, fitur ini dibuat secara out-of-fold untuk mengurangi leakage.

Fitur-fitur tersebut membantu model memanfaatkan informasi yang lebih luas daripada atribut bangunan asli saja. Namun, hubungan prediktif antarfitur tidak secara otomatis membuktikan hubungan sebab-akibat.

Visualisasi evaluasi yang dihasilkan di `reports/figures/` dapat digunakan untuk membantu interpretasi performa model. Confusion matrix, misalnya, dapat menunjukkan tingkat kerusakan yang sering tertukar dengan tingkat lainnya.

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

## Referensi

- [Kaggle — Richter's Predictor: Modeling Earthquake Damage](<https://www.kaggle.com/competitions/richter-s-predictor-modeling-earthquake-damage>)
- [Dokumentasi LightGBM](<https://lightgbm.readthedocs.io/>)
- [Dokumentasi Scikit-learn](<https://scikit-learn.org/stable/>)