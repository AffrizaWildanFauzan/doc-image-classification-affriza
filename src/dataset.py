"""Preprocessing, augmentasi khusus dokumen pindaian, Dataset PyTorch, dan pembuat DataLoader."""
from __future__ import annotations

import io
import random

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

HEADER_FRAC = 0.33   # potongan header = 33% bagian atas halaman


class AcakJPEG:
    """Augmentasi artefak kompresi JPEG (mensimulasikan faks/pindaian berkualitas rendah)."""

    def __init__(self, p: float = 0.3, q: tuple[int, int] = (30, 90)):
        """``p`` peluang diterapkan, ``q`` rentang kualitas JPEG."""
        self.p, self.q = p, q

    def __call__(self, im: Image.Image) -> Image.Image:
        """Kompres lalu dekompres gambar PIL grayscale."""
        if random.random() > self.p:
            return im
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=random.randint(*self.q))
        buf.seek(0)
        return Image.open(buf).convert("L")


class AcakBlur:
    """Augmentasi blur Gaussian ringan (mensimulasikan pindaian tidak fokus)."""

    def __init__(self, p: float = 0.2):
        """``p`` peluang diterapkan."""
        self.p = p

    def __call__(self, im: Image.Image) -> Image.Image:
        """Blur dengan radius acak 0,3 sampai 1,0 px."""
        return im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0))) if random.random() < self.p else im


class AcakNoise:
    """Augmentasi derau Gaussian pada tensor (mensimulasikan bintik pindaian)."""

    def __init__(self, p: float = 0.2, std: float = 0.04):
        """``p`` peluang diterapkan, ``std`` simpangan baku derau pada skala 0..1."""
        self.p, self.std = p, std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Tambah derau lalu potong ke rentang 0..1."""
        return (x + torch.randn_like(x) * self.std).clamp(0, 1) if random.random() < self.p else x


def baca_cache(path: str) -> Image.Image:
    """Buka PNG cache sebagai gambar grayscale (mode ``L``)."""
    return Image.open(path).convert("L")


def ambil_view(im: Image.Image, view: str, header_frac: float = HEADER_FRAC) -> Image.Image:
    """Kembalikan seluruh halaman (``view="full"``) atau potongan bagian atas halaman (``view="header"``)."""
    if view == "header":
        w, h = im.size
        im = im.crop((0, 0, w, max(1, int(h * header_frac))))
    return im


def _ke_tiga_kanal(x: torch.Tensor) -> torch.Tensor:
    """Ulang tensor 1 kanal menjadi 3 kanal agar cocok dengan bobot pretrained RGB."""
    return x.expand(3, -1, -1)


def buat_transform(size: tuple[int, int], mean, std, train: bool) -> T.Compose:
    """Bangun pipeline transform.

    Training memakai augmentasi yang masuk akal untuk dokumen: rotasi/geser/skala/shear kecil, kecerahan dan kontras,
    blur, JPEG, dan derau. Tidak ada flip karena membalik teks.
    """
    langkah = [T.Resize(size, interpolation=T.InterpolationMode.BILINEAR, antialias=True)]
    if train:
        langkah += [T.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.93, 1.05), shear=2, fill=255),
                    T.ColorJitter(brightness=0.2, contrast=0.25), AcakBlur(), AcakJPEG()]
    langkah += [T.ToTensor()]
    if train:
        langkah += [AcakNoise()]
    langkah += [T.Lambda(_ke_tiga_kanal), T.Normalize(mean, std)]
    return T.Compose(langkah)


def buat_transform_tta(size: tuple[int, int], mean, std, skala: float) -> T.Compose:
    """Transform TTA tanpa flip: halaman diperbesar ``skala`` kali lalu dipotong tengah ke ``size``."""
    besar = (int(size[0] * skala), int(size[1] * skala))
    return T.Compose([T.Resize(besar, antialias=True), T.CenterCrop(size), T.ToTensor(),
                      T.Lambda(_ke_tiga_kanal), T.Normalize(mean, std)])


class DocDataset(Dataset):
    """Dataset halaman dokumen dari PNG cache; label ``-1`` untuk data test."""

    def __init__(self, paths, labels, view: str, tfm):
        """``paths`` lokasi PNG cache, ``labels`` array label atau ``None``, ``view`` full/header, ``tfm`` transform."""
        self.paths, self.labels, self.view, self.tfm = list(paths), labels, view, tfm

    def __len__(self) -> int:
        """Jumlah gambar."""
        return len(self.paths)

    def __getitem__(self, i: int):
        """Kembalikan ``(tensor_gambar, label)``."""
        im = ambil_view(baca_cache(self.paths[i]), self.view)
        y = -1 if self.labels is None else int(self.labels[i])
        return self.tfm(im), y


class DonutDataset(Dataset):
    """Dataset untuk Donut: dibaca dari TIFF asli (resolusi penuh), diproses oleh ``DonutImageProcessor``."""

    def __init__(self, paths, proc):
        """``paths`` lokasi TIFF asli, ``proc`` image processor Donut."""
        self.paths, self.proc = list(paths), proc

    def __len__(self) -> int:
        """Jumlah gambar."""
        return len(self.paths)

    def __getitem__(self, i: int):
        """Kembalikan ``(pixel_values, 0)``."""
        im = Image.open(self.paths[i]).convert("RGB")
        return self.proc(im, return_tensors="pt").pixel_values[0], 0


def buat_loader(anggota: dict, df: pd.DataFrame, train: bool, mean, std, num_workers: int, device: str,
                tta: float | None = None) -> DataLoader:
    """Bangun DataLoader untuk satu anggota ensemble.

    Hanya loader training yang memakai ``persistent_workers``; loader evaluasi/TTA yang persistent menumpuk
    proses worker dan bisa menghabiskan RAM Kaggle.
    """
    if tta is not None:
        tfm = buat_transform_tta(anggota["size"], mean, std, tta)
    else:
        tfm = buat_transform(anggota["size"], mean, std, train)
    labels = df["label"].values if "label" in df else None
    ds = DocDataset(df["cache"].values, labels, anggota["view"], tfm)
    return DataLoader(ds, batch_size=anggota["bs"] if train else anggota["bs"] * 2, shuffle=train, drop_last=train,
                      num_workers=num_workers, pin_memory=device == "cuda",
                      persistent_workers=num_workers > 0 and train)
