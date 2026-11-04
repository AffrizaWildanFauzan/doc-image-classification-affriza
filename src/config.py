"""Konfigurasi eksperimen: lokasi folder, sakelar fitur, dan daftar anggota ensemble per profil.

Semua nilai bisa diubah lewat variabel lingkungan ``DOC_*`` sehingga notebook dan skrip CLI
memakai konfigurasi yang sama tanpa mengedit kode.
"""
from __future__ import annotations

import importlib
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# Urutan label checkpoint microsoft/dit-*-finetuned-rvlcdip (config.id2label).
RVL_ID2LABEL = ["letter", "form", "email", "handwritten", "advertisement", "scientific report",
                "scientific publication", "specification", "file folder", "news article", "budget", "invoice",
                "presentation", "questionnaire", "resume", "memo"]

# Id token kelas di naver-clova-ix/donut-base-finetuned-rvlcdip (added_tokens.json).
DONUT_TOKEN = {"<s_rvlcdip>": 57539, "<s_class>": 57537, "<advertisement/>": 57524, "<budget/>": 57525,
               "<email/>": 57526, "<file_folder/>": 57527, "<form/>": 57528, "<handwritten/>": 57529,
               "<invoice/>": 57530, "<letter/>": 57531, "<memo/>": 57532, "<news_article/>": 57533,
               "<presentation/>": 57534, "<questionnaire/>": 57535, "<resume/>": 57536,
               "<scientific_publication/>": 57541, "<scientific_report/>": 57542, "<specification/>": 57543}


def _env_bool(nama: str, default: str) -> bool:
    """Baca variabel lingkungan bernilai ``"1"``/``"0"`` sebagai boolean."""
    return os.environ.get(nama, default) == "1"


def _folder_kerja_default() -> Path:
    """Folder kerja bawaan: ``/kaggle/working`` di Kaggle, selain itu folder ``outputs/`` di root repo."""
    if os.environ.get("DOC_WORKDIR"):
        return Path(os.environ["DOC_WORKDIR"])
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    return Path(__file__).resolve().parents[1]


@dataclass
class Konfigurasi:
    """Seluruh hiperparameter dan lokasi folder untuk satu kali jalan pipeline.

    Atribut penting:
        profil: ``"rvlcdip_ft"`` (default, DiT + Donut yang sudah di-finetune RVL-CDIP),
            ``"dokumen"`` (DiT self-supervised + CNN) atau ``"lengkap"`` (CNN/transformer ImageNet).
        smoke: uji asap CPU dengan data dan model mungil.
        pretrained: ``False`` hanya untuk uji asap tanpa internet (bobot acak).
        kerja: folder induk untuk ``outputs/``, ``checkpoint/`` dan ``reports/``.
    """

    smoke: bool = False
    profil: str = "rvlcdip_ft"
    seed: int = 42
    n_folds: int = 5
    holdout_fold: int = 0
    header_frac: float = 0.33
    phash_max_hamming: int = 4
    drop_konflik: bool = True
    izin_dit: bool = True
    pakai_ocr: bool = False
    pakai_tta: bool = True
    pretrained: bool = True
    batas_jam: float = 2.0
    lewati_jika_ada: bool = True
    ft_epochs: int = 2
    pakai_donut: bool = True
    donut_h: int = 2560
    pakai_dit_large: bool = False
    pakai_effnet: bool = True
    data_root: Path | None = None
    kerja: Path = field(default_factory=_folder_kerja_default)
    mulai: float = field(default_factory=time.time)

    @classmethod
    def dari_env(cls) -> "Konfigurasi":
        """Bangun konfigurasi dari variabel lingkungan ``DOC_*`` (lihat README bagian Konfigurasi)."""
        return cls(
            smoke=_env_bool("DOC_SMOKE", "0"),
            profil=os.environ.get("DOC_PROFIL", "rvlcdip_ft"),
            izin_dit=_env_bool("DOC_IZIN_DIT", "1"),
            pakai_ocr=_env_bool("DOC_OCR", "0"),
            pretrained=_env_bool("DOC_PRETRAINED", "1"),
            batas_jam=float(os.environ.get("DOC_BATAS_JAM", "2.0")),
            ft_epochs=int(os.environ.get("DOC_FT_EPOCHS", "2")),
            pakai_donut=_env_bool("DOC_DONUT", "1"),
            donut_h=int(os.environ.get("DOC_DONUT_H", "2560")),
            pakai_dit_large=_env_bool("DOC_DIT_LARGE", "0"),
            pakai_effnet=_env_bool("DOC_EFFNET", "1"),
            data_root=Path(os.environ["DOC_DATA_ROOT"]) if os.environ.get("DOC_DATA_ROOT") else None,
        )

    # ---- turunan ----
    @property
    def cache_h(self) -> int:
        """Tinggi halaman (px) setelah di-cache sebagai PNG; rasio aspek dipertahankan."""
        return 256 if self.smoke else (768 if self.profil == "lengkap" else 512)

    @property
    def device(self) -> str:
        """``"cuda"`` kalau ada GPU, selain itu ``"cpu"``."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def num_workers(self) -> int:
        """Jumlah worker DataLoader (0 saat uji asap agar mudah di-debug)."""
        return 0 if self.smoke else min(4, os.cpu_count() or 1)

    @property
    def dasar(self) -> Path:
        """Folder induk keluaran; uji asap dipisah ke subfolder ``smoke_run``."""
        return self.kerja / "smoke_run" if self.smoke else self.kerja

    @property
    def outputs(self) -> Path:
        """Folder hasil antara yang tidak di-commit (cache, logit, submission)."""
        return self.dasar / "outputs"

    @property
    def cache_dir(self) -> Path:
        """Folder cache PNG hasil decode TIFF."""
        return self.outputs / f"cache_{self.cache_h}"

    @property
    def artefak(self) -> Path:
        """Folder logit validasi/test per anggota, ledger, dan hasil scan."""
        return self.outputs / "artefak"

    @property
    def checkpoint_dir(self) -> Path:
        """Folder bobot model terbaik per anggota (di-``.gitignore``)."""
        return self.dasar / "checkpoint"

    @property
    def reports_dir(self) -> Path:
        """Folder ringkasan kecil (gambar, ledger, konfigurasi akhir) yang di-commit dan dikutip README."""
        return self.dasar / "reports"

    @property
    def submission_path(self) -> Path:
        """Lokasi ``submission.csv``; di Kaggle harus langsung di ``/kaggle/working``."""
        return self.kerja / "submission.csv" if not self.smoke else self.dasar / "submission.csv"

    def siapkan_folder(self) -> None:
        """Buat semua folder keluaran kalau belum ada."""
        for d in (self.cache_dir, self.artefak, self.checkpoint_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)

    def jam_berjalan(self) -> float:
        """Lama pipeline berjalan (jam) sejak konfigurasi dibuat."""
        return (time.time() - self.mulai) / 3600


def daftar_anggota(cfg: Konfigurasi) -> list[dict]:
    """Kembalikan anggota ensemble yang aktif untuk profil di ``cfg``.

    Setiap anggota adalah dict: ``name``, ``backend`` (``timm``/``hf``/``donut``), ``hub`` (id Hugging Face Hub),
    ``view`` (``full``/``header``), ``size`` (tinggi, lebar), ``epochs``, ``bs``, ``lr`` dan opsi lain
    (``accum``, ``ckpt``, ``layer_decay``, ``head_rvl``).
    """
    lengkap = [
        dict(name="cnxs_full384", backend="timm", hub="timm/convnext_small.fb_in22k_ft_in1k_384",
             view="full", size=(384, 384), epochs=12, bs=32, lr=2e-4, aktif=True),
        dict(name="effv2s_full448", backend="timm", hub="timm/tf_efficientnetv2_s.in21k_ft_in1k",
             view="full", size=(448, 448), epochs=12, bs=32, lr=4e-4, aktif=True),
        dict(name="effv2s_head", backend="timm", hub="timm/tf_efficientnetv2_s.in21k_ft_in1k",
             view="header", size=(256, 576), epochs=10, bs=32, lr=4e-4, aktif=True),
        dict(name="swinv2b_full384", backend="timm", hub="timm/swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
             view="full", size=(384, 384), epochs=10, bs=8, accum=2, ckpt=True, lr=1e-4, aktif=True),
        dict(name="dit_base_full224", backend="hf", hub="microsoft/dit-base",
             view="full", size=(224, 224), epochs=10, bs=32, lr=5e-5, aktif=cfg.izin_dit),
    ]
    dokumen = [
        dict(name="dit_large_full224", backend="hf", hub="microsoft/dit-large", view="full", size=(224, 224),
             epochs=6, bs=32, ckpt=True, lr=1e-4, layer_decay=0.85, aktif=cfg.izin_dit and cfg.pakai_dit_large),
        dict(name="dit_base_full224", backend="hf", hub="microsoft/dit-base", view="full", size=(224, 224),
             epochs=5, bs=64, lr=3e-4, layer_decay=0.75, aktif=cfg.izin_dit),
        dict(name="effv2s_full384", backend="timm", hub="timm/tf_efficientnetv2_s.in21k_ft_in1k", view="full",
             size=(384, 384), epochs=6, bs=32, lr=4e-4, aktif=cfg.pakai_effnet or not cfg.izin_dit),
        dict(name="cnxt_full384", backend="timm", hub="timm/convnext_tiny.in12k_ft_in1k_384", view="full",
             size=(384, 384), epochs=7, bs=32, lr=2e-4, aktif=not cfg.izin_dit),
    ]
    rvlcdip_ft = [
        dict(name="dit_large_rvlft", backend="hf", hub="microsoft/dit-large-finetuned-rvlcdip", head_rvl=True,
             view="full", size=(224, 224), epochs=cfg.ft_epochs, bs=32, ckpt=True, lr=1e-5, layer_decay=0.9, aktif=True),
        dict(name="donut_rvlcdip", backend="donut", hub="naver-clova-ix/donut-base-finetuned-rvlcdip", view="full",
             size=(cfg.donut_h, cfg.donut_h * 3 // 4), epochs=0, bs=8, lr=0.0, aktif=cfg.pakai_donut),
        dict(name="dit_base_rvlft", backend="hf", hub="microsoft/dit-base-finetuned-rvlcdip", head_rvl=True,
             view="full", size=(224, 224), epochs=cfg.ft_epochs, bs=64, lr=2e-5, layer_decay=0.8, aktif=True),
    ]
    anggota = {"rvlcdip_ft": rvlcdip_ft, "dokumen": dokumen, "lengkap": lengkap}[cfg.profil]
    if cfg.smoke:   # model mungil supaya bisa jalan di CPU dalam hitungan menit
        ft = cfg.profil == "rvlcdip_ft"
        anggota = [
            dict(name="r18_full", backend="timm", hub="timm/resnet18.a1_in1k", view="full", size=(160, 160),
                 epochs=1, bs=16, lr=1e-3, aktif=True),
            dict(name="atto_head", backend="timm", hub="timm/convnext_atto.d2_in1k", view="header", size=(96, 192),
                 epochs=1, bs=8, accum=2, ckpt=True, lr=1e-3, aktif=True),
            dict(name="dit_base_rvlft", backend="hf", hub="microsoft/dit-base-finetuned-rvlcdip", head_rvl=True,
                 view="full", size=(224, 224), epochs=cfg.ft_epochs, bs=8, lr=2e-5, layer_decay=0.8, aktif=ft),
            dict(name="donut_rvlcdip", backend="donut", hub="naver-clova-ix/donut-base-finetuned-rvlcdip",
                 view="full", size=(320, 240), epochs=0, bs=4, lr=0.0, aktif=ft and cfg.pakai_donut),
        ]
    return [m for m in anggota if m["aktif"]]


def seed_everything(seed: int) -> None:
    """Samakan seed ``random``, NumPy dan PyTorch (CPU dan semua GPU) agar hasil bisa diulang."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pastikan_paket(modul: str, pip_name: str | None = None) -> None:
    """Pasang paket lewat pip kalau ``modul`` belum bisa di-import (berguna di Kaggle)."""
    try:
        importlib.import_module(modul)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or modul], check=False)
