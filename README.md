# Klasifikasi Citra Dokumen Pindaian (RVL-CDIP, 16 kelas)

Tugas 2 Supervised Machine Learning 2026. Model menebak jenis dokumen (surat, memo, formulir, invoice, dan seterusnya)
dari gambar hasil pindaian. Metrik lomba: **Categorical Accuracy**.

| | |
|---|---|
| Skor public leaderboard | **0,95** |
| Pendekatan | ensemble DiT-large + DiT-base (fine-tune) + Donut, semuanya PyTorch dari Hugging Face Hub |
| Waktu jalan | sekitar 1,3 sampai 1,5 jam di Kaggle GPU T4 x2 |

## Daftar isi

1. [Struktur repository](#struktur-repository)
2. [Unduh data](#unduh-data)
3. [Instalasi](#instalasi)
4. [Cara penggunaan](#cara-penggunaan)
5. [Workflow analisis](#workflow-analisis)
6. [Laporan interpretasi](#laporan-interpretasi)
7. [Referensi](#referensi)

## Struktur repository

```
.
├── README.md                  # dokumen ini: cara pakai + laporan interpretasi
├── requirements.txt           # dependensi Python
├── .gitignore                 # data/, checkpoint/, outputs/ tidak di-commit
├── data/                      # data lomba (diunduh sendiri, tidak di-commit)
│   └── README.md
├── checkpoint/                # bobot model terbaik per anggota (dibuat saat training, tidak di-commit)
│   └── README.md
├── notebooks/
│   └── klasifikasi_dokumen_rvlcdip.ipynb   # alur analisis tahap 1 sampai 10
├── reports/                   # grafik dan ringkasan angka yang dikutip README (dibuat oleh notebook)
├── scripts/
│   └── run_pipeline.py        # jalankan seluruh pipeline dari baris perintah
└── src/                       # kode inti, setiap fungsi dan kelas ber-docstring
    ├── __init__.py
    ├── config.py              # konfigurasi (variabel lingkungan DOC_*) dan daftar anggota ensemble per profil
    ├── data.py                # muat CSV, scan integritas TIFF, deteksi duplikat (MD5 + pHash), fold berbasis grup
    ├── dataset.py             # preprocessing, augmentasi dokumen, Dataset dan DataLoader
    ├── models.py              # DiT (transformers), Donut, backbone timm
    ├── train.py               # loop training, prediksi, TTA, experiment ledger
    ├── ensemble.py            # noise floor, uji berpasangan, pemilihan kombinasi ensemble, submission
    ├── visualize.py           # grafik EDA dan evaluasi, Grad-CAM
    └── ocr.py                 # anggota teks opsional (Tesseract + TF-IDF)
```

Folder `outputs/` (cache PNG, logit tiap anggota, `submission.csv`) dibuat otomatis saat pipeline berjalan dan di-`.gitignore`.

## Unduh data

Data **tidak** disimpan di repo ini (sekitar 4,6 GB, 40 ribu file TIFF).

**Opsi 1, Kaggle API (lokal):**

1. Buka link lomba dari dosen dan klik *Join Competition* / terima aturan lomba.
2. Buat API token di Kaggle (*Settings → API → Create New Token*), simpan `kaggle.json` ke `~/.kaggle/` lalu `chmod 600 ~/.kaggle/kaggle.json`.
3. Unduh dan ekstrak ke `data/`. `<slug-lomba>` adalah bagian URL lomba `kaggle.com/competitions/<slug-lomba>`:

```bash
kaggle competitions download -c <slug-lomba> -p data/
unzip -q data/<slug-lomba>.zip -d data/
```

Hasil akhirnya harus seperti di `data/README.md` (`data/train.csv`, `data/test.csv`, `data/sample_submission.csv`,
`data/train/train/<kelas>/*.tif`, `data/test/test/*.tif`). Kode juga menerima tata letak `data/train/<kelas>/*.tif`.

**Opsi 2, Kaggle Notebook:** tidak perlu mengunduh. Tambahkan data lomba lewat *Add Input*; lokasi `train.csv`
dicari otomatis di `/kaggle/input`.

## Instalasi

```bash
git clone https://github.com/USERNAME/NAMA-REPO.git
cd NAMA-REPO
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Butuh Python 3.10+ dan GPU CUDA untuk training penuh. Bobot model diunduh otomatis dari Hugging Face Hub saat pertama dipakai.

## Cara penggunaan

### A. Notebook (disarankan, berisi analisis lengkap)

**Di Kaggle**

1. *New Notebook → File → Import Notebook*, pilih `notebooks/klasifikasi_dokumen_rvlcdip.ipynb`.
2. *Add Input*: data lomba.
3. *Settings*: Accelerator **GPU T4 x2**, Internet **ON**.
4. Ganti `REPO_URL` di sel pertama dengan URL repo ini (notebook meng-clone repo untuk mengambil folder `src/`).
   Cara lain: unggah folder repo sebagai Kaggle Dataset lalu tambahkan sebagai input.
5. *Run All*. `submission.csv` muncul di `/kaggle/working/`, grafik dan ringkasan di `/kaggle/working/reports/`.

**Lokal**

```bash
jupyter lab notebooks/klasifikasi_dokumen_rvlcdip.ipynb
```

### B. Baris perintah

```bash
python scripts/run_pipeline.py                         # profil default rvlcdip_ft
python scripts/run_pipeline.py --batas-jam 1.0         # anggota baru tidak dimulai setelah 1 jam
python scripts/run_pipeline.py --tanpa-donut           # hanya DiT (sekitar 45 menit)
python scripts/run_pipeline.py --profil dokumen        # DiT self-supervised + EfficientNetV2-S
```

### C. Uji asap (CPU, tanpa internet, beberapa menit)

```bash
DOC_SMOKE=1 DOC_PRETRAINED=0 python scripts/run_pipeline.py
```

Data dipotong kecil dan model diganti versi mungil berbobot acak, untuk memastikan seluruh alur berjalan sampai
`submission.csv` terbentuk.

### Konfigurasi

| variabel | default | arti |
|---|---|---|
| `DOC_PROFIL` | `rvlcdip_ft` | `rvlcdip_ft` (DiT + Donut fine-tuned RVL-CDIP), `dokumen`, atau `lengkap` |
| `DOC_FT_EPOCHS` | `2` | epoch fine-tune DiT; `0` = checkpoint apa adanya |
| `DOC_DONUT` | `1` | `0` mematikan anggota Donut |
| `DOC_DONUT_H` | `2560` | tinggi input Donut; `1920` kira-kira 2 kali lebih cepat |
| `DOC_BATAS_JAM` | `2.0` | anggota baru tidak dimulai setelah batas ini |
| `DOC_OCR` | `0` | `1` menambah anggota teks OCR (lambat, 3 sampai 5 jam) |
| `DOC_DATA_ROOT` | otomatis | folder berisi `train.csv` |
| `DOC_WORKDIR` | root repo / `/kaggle/working` | induk folder `outputs/`, `checkpoint/`, `reports/` |
| `DOC_SMOKE`, `DOC_PRETRAINED` | `0`, `1` | uji asap |

## Workflow analisis

Notebook mengikuti tahapan praktikum. Setiap tahap memanggil fungsi di `src/` lalu menafsirkan hasilnya.

1. **Pemahaman data & problem statement**: 32.003 gambar latih, 7.994 gambar test, 16 kelas seimbang, TIFF grayscale setinggi 1.000 px.
2. **Data cleaning**: setiap TIFF dibuka sekali untuk cek integritas, MD5, pHash, dan cache PNG. Duplikat persis dan near-duplicate
   dikelompokkan dengan union-find; kelompok kembar berlabel berbeda dibuang dari data latih; fold dibagi dengan
   `StratifiedGroupKFold` supaya gambar kembar tidak bocor antara train dan holdout. Duplikat train dan test hanya dilaporkan.
3. **EDA**: distribusi kelas, ukuran, proporsi tinta, sebaran dan korelasi metadata, profil tinta atas-bawah dan kiri-kanan
   per kelas (sidik jari tata letak), contoh per kelas, rata-rata halaman per kelas, train vs test, contoh kembar berlabel konflik.
4. **Preprocessing & augmentasi**: cache PNG, view halaman penuh dan header, augmentasi pindaian (rotasi kecil, kontras, blur,
   JPEG, derau) **tanpa flip**; noise floor (standard error holdout) sebagai batas selisih yang bermakna.
6. **Konstruksi model**: DiT-large dan DiT-base fine-tuned RVL-CDIP dilatih lagi 2 epoch (AdamW, layer-wise LR decay,
   label smoothing, AMP, DataParallel 2 GPU), Donut dipakai tanpa training.
7. **Model selection**: semua kombinasi anggota, greedy ensemble selection, split-half, dan uji berpasangan bootstrap.
8. **Evaluasi**: classification report, confusion matrix, pasangan kelas yang paling sering tertukar, precision/recall/F1
   per kelas, akurasi top-k, kalibrasi (reliability diagram, ECE), akurasi vs cakupan.
9. **Visualisasi**: kurva training, akurasi per kelas per anggota, keragaman anggota (prediksi berbeda dan korelasi kesalahan),
   kenaikan akurasi per kelas dari ensemble, t-SNE ruang peluang, contoh salah prediksi, peta atensi DiT, Grad-CAM (anggota CNN),
   distribusi kelas prediksi test.
10. **Optimalisasi**: TTA tanpa flip, bobot ensemble, anggota OCR opsional, submission + sub-diff.

## Laporan interpretasi

> Angka di bagian ini diambil dari run Kaggle profil `rvlcdip_ft` (T4 x2). Rincian lengkap tiap run tersimpan di
> `reports/ringkasan_hasil.json` dan `reports/ledger.csv`.

### 1. Hasil utama

| model | akurasi holdout | keterangan |
|---|---|---|
| DiT-large fine-tuned RVL-CDIP (+2 epoch) | lihat `reports/ledger.csv` | model citra terkuat |
| Donut fine-tuned RVL-CDIP (tanpa training) | lihat `reports/ledger.csv` | membaca teks langsung dari piksel |
| DiT-base fine-tuned RVL-CDIP (+2 epoch) | lihat `reports/ledger.csv` | pembeda kapasitas |
| **Ensemble akhir** | lihat `reports/ringkasan_hasil.json` | **public leaderboard 0,95** |

Sebagai pembanding, model yang dilatih dari bobot ImageNet atau self-supervised pada percobaan awal mencapai:
EfficientNetV2-S 448 px 0,891, ConvNeXt-S 384 px 0,878, DiT-base self-supervised 0,875, dan EfficientNetV2-S header 0,823.
Lompatan ke 0,95 datang dari dua hal: checkpoint yang sudah mengenal 16 kelas RVL-CDIP, dan Donut yang membaca teks.

![Rata-rata halaman per kelas](reports/eda_rata2_halaman.png)

![Profil tinta per kelas](reports/eda_profil_tinta.png)

Profil tinta memperlihatkan bahwa sebagian kelas punya tata letak yang khas (advertisement dan file_folder gelap di seluruh
halaman, news_article dan scientific_publication padat berkolom), sedangkan letter, memo, email, dan form punya profil hampir sama.
Kelas-kelas terakhir ini hanya bisa dibedakan dari isi teks, dan inilah alasan Donut ditambahkan.

### 2. Temuan data cleaning

* 1 gambar latih tidak bisa dibaca dan dibuang.
* 9.275 pasangan near-duplicate di train. 1.098 gambar berada di kelompok kembar dengan label berbeda; ini bukti nyata derau
  label RVL-CDIP dan gambarnya dibuang dari data latih (30.904 gambar tersisa: 24.816 train, 6.088 holdout).
* 25 gambar test identik (MD5) dan 119 gambar test identik secara visual (pHash) dengan gambar train. Angka ini hanya
  dilaporkan; labelnya tidak disalin ke submission.
* Model yang hanya memakai metadata (ukuran file, lebar, proporsi tinta) diuji di tahap 2 notebook untuk memastikan tidak ada
  sinyal non-visual yang membocorkan label; nama file test sudah dianonimkan sehingga tidak bisa dipakai.

### 3. Mengapa model ini bekerja

* **Teks adalah pembeda utama.** Di literatur, model yang hanya melihat gambar mentok di sekitar 92 sampai 94% pada RVL-CDIP
  (DiT-large 92,7%), sedangkan model yang juga membaca teks mencapai 95 sampai 97% (Donut 95,3%, LayoutLMv3 95,4%).
  Donut dipilih karena membaca teks dari piksel tanpa OCR, sehingga muat dalam batas waktu.
* **Tata letak dan teks saling melengkapi.** DiT melihat halaman 224 px sehingga menangkap tata letak (kolom, tabel,
  kop, area kosong), sedangkan Donut membaca halaman 2560 px sehingga menangkap kata kunci (MEMORANDUM, To/From, INVOICE).
  Ensemble paling untung kalau kesalahan anggotanya berbeda; perbandingan akurasi per kelas tiap anggota ada di
  `reports/akurasi_per_kelas.png`.
* **Fine-tune singkat dengan LR kecil.** Checkpoint sudah berada di ruang kelas yang sama, jadi 2 epoch dengan LR 1e-5 sampai
  2e-5 dan layer-wise LR decay cukup untuk menyesuaikan dengan partisi lomba tanpa merusak representasi.

### 4. Analisis kesalahan

![Confusion matrix](reports/confusion_matrix.png)

![Akurasi per kelas](reports/akurasi_per_kelas.png)


* Pasangan yang paling sering tertukar adalah pasangan yang memang ambigu secara definisi: form dan questionnaire,
  scientific_report dan scientific_publication, letter dan memo, budget dan invoice (lihat `reports/confusion_matrix.png`).
* Banyak salah prediksi dengan keyakinan tinggi (`reports/salah_prediksi.png`) adalah label yang meragukan. Larson dkk. (2023)
  memperkirakan sekitar 8,1% label RVL-CDIP salah, misalnya letter yang sebenarnya memo dan invoice yang sebenarnya budget.
  Artinya skor di atas sekitar 95% sebagian besar berasal dari dokumen yang punya kembaran atau templat di data latih.

![Salah prediksi dengan keyakinan tertinggi](reports/salah_prediksi.png)

* **Keragaman anggota** (`reports/keragaman_anggota.png`, `reports/gain_ensemble.png`): kenaikan ensemble datang dari pasangan
  anggota yang sering berbeda pendapat dan kesalahannya tidak berkorelasi.
* **Kalibrasi** (`reports/kalibrasi.png`): prediksi salah umumnya punya keyakinan lebih rendah, sehingga halaman dengan keyakinan
  rendah bisa diteruskan ke pemeriksaan manual (`reports/akurasi_vs_cakupan.png`).
* **t-SNE** (`reports/tsne_holdout.png`): salah prediksi berkumpul di perbatasan gugus kelas yang ambigu.
* **Peta atensi DiT** (`reports/atensi_dit.png`): model memperhatikan isi dokumen (kop, header, tabel, blok teks).

![Keragaman anggota](reports/keragaman_anggota.png)

![t-SNE holdout](reports/tsne_holdout.png)

### 5. Keterbatasan dan catatan kejujuran

* Checkpoint RVL-CDIP dilatih pada 320 ribu gambar RVL-CDIP, yang kemungkinan mencakup sebagian gambar lomba. Karena itu akurasi
  holdout cenderung optimis; skor leaderboard adalah ukuran yang lebih jujur. Penggunaan checkpoint ini diizinkan panitia.
* Holdout dibagi berbasis grup kembar, tetapi kembaran yang lolos dari ambang pHash masih bisa ada.
* Selisih kurang dari sekitar 0,6 poin di holdout (2 x standard error) dianggap belum bermakna kecuali lolos uji berpasangan.

### 6. Langkah lanjutan

* Fine-tune Donut beberapa ratus langkah pada data lomba.
* DiT pada resolusi lebih tinggi (384 px) dengan interpolasi posisi.
* Anggota multimodal citra + teks + tata letak (LayoutLMv3) kalau OCR bisa dipercepat.

## Referensi

* Harley, A. W., Ufkes, A., & Derpanis, K. G. (2015). Evaluation of deep convolutional nets for document image classification and retrieval. *ICDAR*.
* Li, J., Xu, Y., Lv, T., Cui, L., Zhang, C., & Wei, F. (2022). DiT: Self-supervised pre-training for document image transformer. *ACM MM*. arXiv:2203.02378.
* Kim, G., Hong, T., Yim, M., et al. (2022). OCR-free document understanding transformer (Donut). *ECCV*. arXiv:2111.15664.
* Larson, S., Lim, G., & Leach, K. (2023). On evaluation of document classification using RVL-CDIP. *EACL*. arXiv:2306.12550.
* Huang, Y., Lv, T., Cui, L., Lu, Y., & Wei, F. (2022). LayoutLMv3: Pre-training for document AI with unified text and image masking. *ACM MM*.
* Das, A., Roy, S., Bhattacharya, U., & Parui, S. K. (2018). Document image classification with intra-domain transfer learning and stacked generalization of deep convolutional neural networks. *ICPR*.
* Caruana, R., Niculescu-Mizil, A., Crew, G., & Ksikes, A. (2004). Ensemble selection from libraries of models. *ICML*.
