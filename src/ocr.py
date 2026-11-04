"""Anggota teks opsional: OCR Tesseract lalu TF-IDF kata + karakter dan Logistic Regression."""
from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

from .config import pastikan_paket
from .data import Konteks


def ocr_satu(path: str) -> str:
    """Baca teks satu halaman TIFF asli dengan Tesseract (mode segmentasi halaman otomatis)."""
    import pytesseract
    return pytesseract.image_to_string(Image.open(path).convert("L"), config="--psm 3")


def jalankan_ocr(ctx: Konteks) -> dict:
    """OCR semua halaman train dan test; hasil di-cache ke ``artefak/ocr_<h>.parquet``.

    Thread dipakai (bukan proses) karena pytesseract sudah memanggil proses tesseract terpisah, dan fork setelah
    PyTorch dimuat bisa macet.

    Returns:
        Peta ``path -> teks``.
    """
    pastikan_paket("pytesseract")
    if shutil.which("tesseract") is None:
        subprocess.run("apt-get install -y -q tesseract-ocr", shell=True, check=False)
    f = ctx.cfg.artefak / f"ocr_{ctx.cfg.cache_h}.parquet"
    ocr = pd.read_parquet(f) if f.exists() else pd.DataFrame({"path": [], "teks": []})
    semua = pd.concat([ctx.train_df[["path"]], ctx.test_df[["path"]]]).drop_duplicates("path")
    sisa = semua[~semua["path"].isin(set(ocr["path"]))]
    if len(sisa):
        os.environ["OMP_THREAD_LIMIT"] = "1"
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
            teks = list(tqdm(ex.map(ocr_satu, sisa["path"]), total=len(sisa), desc="OCR"))
        ocr = pd.concat([ocr, pd.DataFrame({"path": sisa["path"].values, "teks": teks})])
        ocr.to_parquet(f)
    return dict(zip(ocr["path"], ocr["teks"]))


def latih_anggota_ocr(ctx: Konteks) -> float:
    """Latih TF-IDF + Logistic Regression di atas teks OCR, simpan logit ke ``artefak/ocr_tfidf/``.

    Returns:
        Akurasi holdout anggota teks.
    """
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    peta = jalankan_ocr(ctx)
    df = ctx.train_df
    t_tr = [peta[p] for p in df["path"].values[ctx.tr_idx]]
    t_va = [peta[p] for p in df["path"].values[ctx.va_idx]]
    t_te = [peta[p] for p in ctx.test_df["path"].values]
    vw = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300000, sublinear_tf=True)
    Xtr = hstack([vw.fit_transform(t_tr), vc.fit_transform(t_tr)]).tocsr()
    Xva = hstack([vw.transform(t_va), vc.transform(t_va)]).tocsr()
    Xte = hstack([vw.transform(t_te), vc.transform(t_te)]).tocsr()
    lr = LogisticRegression(C=8, max_iter=2000).fit(Xtr, df["label"].values[ctx.tr_idx])
    n = ctx.num_classes
    pv, pt = np.zeros((len(ctx.va_idx), n)), np.zeros((len(ctx.test_df), n))
    pv[:, lr.classes_] = lr.predict_proba(Xva)
    pt[:, lr.classes_] = lr.predict_proba(Xte)
    d = ctx.cfg.artefak / "ocr_tfidf"
    d.mkdir(exist_ok=True)
    np.save(d / "val_logits.npy", np.log(pv + 1e-9))
    np.save(d / "test_logits.npy", np.log(pt + 1e-9))
    return accuracy_score(ctx.y_va, pv.argmax(1))
