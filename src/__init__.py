"""Paket klasifikasi citra dokumen pindaian (RVL-CDIP, 16 kelas) untuk Tugas 2 Supervised Machine Learning.

Modul:
    config     konfigurasi eksperimen dan daftar anggota ensemble per profil
    data       muat data, scan integritas, deteksi duplikat, pembagian fold
    dataset    preprocessing, augmentasi, Dataset dan DataLoader
    models     DiT (transformers), Donut, dan backbone timm dari Hugging Face Hub
    train      training, prediksi, TTA, experiment ledger
    ensemble   noise floor, pemilihan kombinasi ensemble, submission
    visualize  grafik EDA dan evaluasi, Grad-CAM
    ocr        anggota teks opsional (Tesseract + TF-IDF)
"""

__version__ = "1.0.0"
