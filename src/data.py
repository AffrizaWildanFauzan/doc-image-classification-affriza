"""Memuat data lomba, memindai integritas file, mendeteksi duplikat, dan membagi fold berbasis grup kembar."""
from __future__ import annotations

import hashlib
import itertools
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm

from .config import Konfigurasi

Image.MAX_IMAGE_PIXELS = None


@dataclass
class Konteks:
    """Data yang dipakai bersama oleh tahap training, evaluasi dan ensemble.

    Atribut:
        cfg: konfigurasi eksperimen.
        train_df: data latih setelah cleaning, dengan kolom ``path``, ``cache``, ``label``, ``grp``, ``fold``.
        test_df: data test dengan kolom ``id``, ``path``, ``cache``.
        classes: peta label integer -> nama kelas (urutan lomba, 0..15).
        tr_idx, va_idx: indeks baris ``train_df`` untuk train dan holdout.
    """

    cfg: Konfigurasi
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    classes: dict
    tr_idx: np.ndarray
    va_idx: np.ndarray

    @property
    def y_va(self) -> np.ndarray:
        """Label holdout."""
        return self.train_df["label"].values[self.va_idx]

    @property
    def num_classes(self) -> int:
        """Jumlah kelas (16)."""
        return len(self.classes)

    @property
    def nama_kelas(self) -> list[str]:
        """Nama kelas berurutan sesuai label 0..15."""
        return [self.classes[c] for c in range(self.num_classes)]


def cari_root(cfg: Konfigurasi) -> Path:
    """Cari folder yang berisi ``train.csv`` dan ``test.csv``.

    Urutan pencarian: ``cfg.data_root``, folder ``data/`` di repo, folder kerja, lalu semua dataset di ``/kaggle/input``.
    """
    repo = Path(__file__).resolve().parents[1]
    kandidat = [cfg.data_root] if cfg.data_root else []
    kandidat += [repo / "data", repo, Path.cwd(), Path.cwd().parent]
    if Path("/kaggle/input").exists():
        kandidat += [p.parent for p in Path("/kaggle/input").rglob("train.csv")]
    for c in kandidat:
        if c and (c / "train.csv").exists() and (c / "test.csv").exists():
            return c
    raise FileNotFoundError("train.csv tidak ditemukan. Unduh data ke folder data/ atau set DOC_DATA_ROOT.")


def resolve_path(root: Path, rel: str) -> str | None:
    """Ubah path relatif di CSV menjadi path file yang benar-benar ada.

    CSV menulis ``train/<kelas>/x.tif``, sedangkan unduhan Kaggle/GitHub menyimpan ``train/train/<kelas>/x.tif``.
    Kedua tata letak dicoba; ``None`` kalau file tidak ditemukan.
    """
    rel = Path(rel)
    for p in (root / rel, root / rel.parts[0] / rel):
        if p.exists():
            return str(p)
    return None


def muat_csv(cfg: Konfigurasi) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Baca ``train.csv``, ``test.csv``, ``sample_submission.csv`` dan cek konsistensinya.

    Returns:
        ``(train_df, test_df, sample_submission, classes)`` dengan kolom ``path`` berisi lokasi file TIFF.
    """
    root = cari_root(cfg)
    train_df = pd.read_csv(root / "train.csv")
    test_df = pd.read_csv(root / "test.csv")
    sub = pd.read_csv(root / "sample_submission.csv")
    classes = (train_df[["label", "class_name"]].drop_duplicates().sort_values("label")
               .set_index("label")["class_name"].to_dict())
    assert train_df["id"].is_unique and test_df["id"].is_unique
    assert train_df.groupby("label")["class_name"].nunique().max() == 1
    assert sorted(classes) == list(range(len(classes)))
    assert (sub["id"].values == test_df["id"].values).all()
    train_df["path"] = [resolve_path(root, f) for f in tqdm(train_df["filename"], desc="cek path train")]
    test_df["path"] = [resolve_path(root, f) for f in tqdm(test_df["filename"], desc="cek path test")]
    if cfg.smoke:   # pada uji asap hanya sebagian kecil gambar yang tersedia
        train_df = train_df.dropna(subset=["path"]).reset_index(drop=True)
        test_df = test_df.dropna(subset=["path"]).reset_index(drop=True)
    else:
        assert train_df["path"].notna().all() and test_df["path"].notna().all(), "ada file yang tidak ditemukan"
    print(f"folder data: {root} | train {len(train_df)} | test {len(test_df)}")
    return train_df, test_df, sub, classes


def scan_satu(path: str, cache_dir: Path, cache_h: int) -> dict:
    """Baca satu TIFF sekali jalan: metadata, statistik tinta, MD5, pHash, dan cache PNG setinggi ``cache_h`` px.

    Kegagalan baca tidak menghentikan proses; kolom ``ok`` bernilai ``False`` dan pesan disimpan di ``err``.
    """
    out = dict(path=path, ok=False)
    try:
        raw = Path(path).read_bytes()
        out["md5"] = hashlib.md5(raw).hexdigest()
        out["file_kb"] = len(raw) / 1024
        im = Image.open(path)
        out.update(mode=im.mode, w=im.size[0], h=im.size[1], n_frames=getattr(im, "n_frames", 1),
                   compression=str(im.info.get("compression")), dpi=str(im.info.get("dpi")))
        g = im.convert("L")
        a = np.asarray(g, dtype=np.uint8)
        out["mean"] = float(a.mean())
        out["ink"] = float((a < 128).mean())        # proporsi piksel gelap (tinta)
        out["phash"] = str(imagehash.phash(g, hash_size=8))
        cache_path = Path(cache_dir) / (hashlib.md5(path.encode()).hexdigest() + ".png")
        if not cache_path.exists():
            nw = max(1, round(g.size[0] * cache_h / g.size[1]))
            g.resize((nw, cache_h), Image.BILINEAR).save(cache_path, optimize=False, compress_level=1)
        out["cache"] = str(cache_path)
        out["ok"] = True
    except Exception as e:  # noqa: BLE001 - file rusak dicatat, bukan dihentikan
        out["err"] = repr(e)
    return out


def scan_semua(cfg: Konfigurasi, paths: list[str], nama: str) -> pd.DataFrame:
    """Jalankan :func:`scan_satu` paralel untuk semua ``paths``; hasil disimpan ke parquet agar sesi berikutnya cepat."""
    f = cfg.artefak / f"scan_{nama}_{cfg.cache_h}.parquet"
    if f.exists():
        s = pd.read_parquet(f)
        # cache PNG dari sesi lain (misalnya output Kaggle lama) belum tentu ada di sesi ini, jadi dicek dulu
        cache_ada = s["cache"].dropna().head(20).map(lambda c: Path(c).exists()).all()
        if set(s["path"]) >= set(paths) and cache_ada:
            return s.set_index("path").loc[list(paths)].reset_index()
    fungsi = partial(scan_satu, cache_dir=cfg.cache_dir, cache_h=cfg.cache_h)
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() or 1)) as ex:
        res = list(tqdm(ex.map(fungsi, paths, chunksize=32), total=len(paths), desc=f"scan {nama}"))
    s = pd.DataFrame(res)
    if "err" not in s:
        s["err"] = None
    s.to_parquet(f)
    return s


def cari_pasangan(hashes, max_ham: int = 4, max_group: int = 300) -> list[tuple[int, int, int]]:
    """Cari pasangan pHash 64-bit dengan jarak Hamming <= ``max_ham`` memakai multi-index hashing.

    Hash dipotong menjadi ``max_ham + 1`` bagian; menurut prinsip sarang merpati, dua hash yang cukup dekat
    pasti identik di minimal satu potongan, jadi hanya kandidat yang berbagi potongan yang dibandingkan.

    Returns:
        Daftar ``(i, j, jarak)`` dengan ``i < j``.
    """
    H = np.array([int(h, 16) for h in hashes], dtype=np.uint64)
    n_pot = max_ham + 1
    batas = np.linspace(0, 64, n_pot + 1).astype(int)
    pasangan = set()
    for i in range(n_pot):
        lo, hi = batas[i], batas[i + 1]
        kunci = (H >> np.uint64(lo)) & np.uint64((1 << (hi - lo)) - 1)
        for idx in pd.Series(np.arange(len(H))).groupby(kunci).apply(list):
            if 1 < len(idx) <= max_group:
                for a, b in itertools.combinations(idx, 2):
                    pasangan.add((a, b) if a < b else (b, a))
    return [(a, b, d) for a, b in pasangan if (d := bin(int(H[a] ^ H[b])).count("1")) <= max_ham]


class UnionFind:
    """Struktur disjoint-set untuk mengelompokkan gambar kembar (transitif: A~B dan B~C berarti satu grup)."""

    def __init__(self, n: int):
        """Buat ``n`` himpunan tunggal."""
        self.p = list(range(n))

    def find(self, x: int) -> int:
        """Kembalikan wakil grup ``x`` (dengan path halving)."""
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        """Gabungkan grup ``a`` dan ``b``."""
        self.p[self.find(a)] = self.find(b)


def kelompokkan_duplikat(train_df: pd.DataFrame, max_ham: int = 4) -> tuple[pd.DataFrame, int]:
    """Tambah kolom ``grp`` (id grup kembar), ``dup`` dan ``konflik`` (kembar dengan label berbeda) ke ``train_df``.

    Duplikat persis (MD5 sama) dan near-duplicate (pHash dekat) digabung. Halaman hampir kosong dikecualikan
    karena pHash-nya mirip walau isinya berbeda.

    Returns:
        ``(train_df, jumlah_pasangan_near_duplicate)``.
    """
    train_df = train_df.copy()
    ok = train_df["ok"].values
    kosong = train_df["ink"].fillna(0).values < 0.005
    idx_valid = np.where(ok & ~kosong)[0]
    pairs = [(idx_valid[a], idx_valid[b], d) for a, b, d in cari_pasangan(train_df["phash"].values[idx_valid], max_ham)]
    uf = UnionFind(len(train_df))
    for a, b, _ in pairs:
        uf.union(a, b)
    for _, g in train_df[ok].groupby("md5"):
        for a, b in zip(g.index[:-1], g.index[1:]):
            uf.union(a, b)
    train_df["grp"] = [uf.find(i) for i in range(len(train_df))]
    train_df["dup"] = train_df.groupby("grp")["id"].transform("size") > 1
    train_df["konflik"] = train_df.groupby("grp")["label"].transform("nunique") > 1
    return train_df, len(pairs)


def laporan_duplikat_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Hitung gambar test yang MD5/pHash-nya identik dengan gambar train.

    Hanya untuk laporan (memahami jarak CV dan papan); label train TIDAK pernah disalin ke test.
    """
    return dict(md5_identik=int(test_df["md5"].isin(set(train_df["md5"].dropna())).sum()),
                phash_identik=int(test_df["phash"].isin(set(train_df["phash"].dropna())).sum()),
                n_test=len(test_df))


def bersihkan_dan_bagi(cfg: Konfigurasi, train_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Buang gambar rusak dan (opsional) kembar berlabel konflik, lalu bagi fold dengan StratifiedGroupKFold.

    Grup kembar tidak pernah terbelah antara train dan holdout, sehingga akurasi holdout tidak terlalu optimis.

    Returns:
        ``(train_df_bersih, tr_idx, va_idx)``.
    """
    df = train_df[train_df["ok"]].copy()
    if cfg.drop_konflik:
        df = df[~df["konflik"]]
    df = df.reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds if not cfg.smoke else 4, shuffle=True, random_state=cfg.seed)
    df["fold"] = -1
    for f, (_, va) in enumerate(sgkf.split(df, df["label"], groups=df["grp"])):
        df.loc[va, "fold"] = f
    tr_idx = np.where(df["fold"] != cfg.holdout_fold)[0]
    va_idx = np.where(df["fold"] == cfg.holdout_fold)[0]
    assert not set(df["grp"].values[tr_idx]) & set(df["grp"].values[va_idx]), "grup kembar bocor antar split"
    return df, tr_idx, va_idx


def pulihkan_artefak_kaggle(cfg: Konfigurasi) -> None:
    """Salin logit anggota dan ledger dari output sesi Kaggle sebelumnya (ditambahkan sebagai input) agar tidak dilatih ulang.

    Bekerja untuk output notebook versi lama (``artefak/``) maupun versi ini (``outputs/artefak/``).
    """
    if not Path("/kaggle/input").exists():
        return
    for lama in Path("/kaggle/input").rglob("artefak/*/test_logits.npy"):
        tujuan = cfg.artefak / lama.parent.name
        if not (tujuan / "test_logits.npy").exists():
            shutil.copytree(lama.parent, tujuan, dirs_exist_ok=True)
            print("dipulihkan dari sesi sebelumnya:", lama.parent.name)
    for f in Path("/kaggle/input").rglob("artefak/ledger.csv"):
        if not (cfg.artefak / f.name).exists():
            shutil.copy(f, cfg.artefak / f.name)


def siapkan_konteks(cfg: Konfigurasi, verbose: bool = True) -> tuple[Konteks, dict]:
    """Jalankan seluruh tahap data (muat, scan, duplikat, cleaning, fold) dan kembalikan :class:`Konteks`.

    Returns:
        ``(konteks, info)``; ``info`` berisi data mentah dan ringkasan cleaning untuk dilaporkan di notebook.
    """
    cfg.siapkan_folder()
    pulihkan_artefak_kaggle(cfg)
    train_df, test_df, sub, classes = muat_csv(cfg)
    train_df = train_df.merge(scan_semua(cfg, list(train_df["path"]), "train"), on="path", how="left")
    test_df = test_df.merge(scan_semua(cfg, list(test_df["path"]), "test"), on="path", how="left")
    assert test_df["ok"].all(), "ada gambar test rusak, cek manual"
    train_raw, n_pairs = kelompokkan_duplikat(train_df, cfg.phash_max_hamming)
    train_bersih, tr_idx, va_idx = bersihkan_dan_bagi(cfg, train_raw)
    info = dict(train_raw=train_raw, sample_submission=sub, n_pairs=n_pairs,
                n_rusak=int((~train_raw["ok"]).sum()), n_konflik=int(train_raw["konflik"].sum()),
                duplikat_test=laporan_duplikat_test(train_raw, test_df))
    if verbose:
        print(f"rusak: {info['n_rusak']} | pasangan near-duplicate: {n_pairs} | kembar berlabel konflik: "
              f"{info['n_konflik']} | train {len(tr_idx)} | holdout {len(va_idx)}")
    return Konteks(cfg, train_bersih, test_df, classes, tr_idx, va_idx), info
