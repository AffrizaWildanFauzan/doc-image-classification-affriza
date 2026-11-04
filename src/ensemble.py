"""Noise floor, pemilihan kombinasi ensemble (exhaustive, greedy Caruana, split-half), dan penulisan submission."""
from __future__ import annotations

import itertools
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def softmax_np(z: np.ndarray) -> np.ndarray:
    """Softmax baris demi baris yang stabil secara numerik."""
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def se_akurasi(p: float, n: int) -> float:
    """Standard error akurasi ``p`` pada ``n`` sampel (pendekatan binomial)."""
    return math.sqrt(p * (1 - p) / n)


def bandingkan(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 42) -> dict:
    """Uji berpasangan selisih akurasi ``b - a`` dengan interval bootstrap 95%.

    Returns:
        dict ``selisih``, ``lo``, ``hi``, dan ``nyata`` (``True`` kalau interval tidak memuat 0).
    """
    rng = np.random.default_rng(seed)
    d = (pred_b == y).astype(float) - (pred_a == y).astype(float)
    boots = d[rng.integers(0, len(y), size=(n_boot, len(y)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(selisih=float(d.mean()), lo=float(lo), hi=float(hi), nyata=bool(lo > 0 or hi < 0))


def muat_pool(artefak: Path, n_val: int, n_test: int, pakai_tta: bool = False) -> dict:
    """Muat peluang holdout dan test semua anggota dari ``artefak/<nama>/``.

    Kalau ``pakai_tta`` dan file TTA ada, versi TTA yang dipakai. Anggota yang ukuran logitnya tidak cocok dilewati.

    Returns:
        ``{nama: {"val": prob_holdout, "test": prob_test}}``.
    """
    pool = {}
    for d in sorted(p for p in Path(artefak).iterdir() if p.is_dir()):
        suf = "_tta" if pakai_tta and (d / "val_logits_tta.npy").exists() else ""
        fv, ft = d / f"val_logits{suf}.npy", d / f"test_logits{suf}.npy"
        if fv.exists() and ft.exists():
            v, t = np.load(fv), np.load(ft)
            if len(v) == n_val and len(t) == n_test:
                pool[d.name] = dict(val=softmax_np(v), test=softmax_np(t))
    return pool


def tabel_anggota(pool: dict, y: np.ndarray) -> pd.DataFrame:
    """Akurasi holdout tiap anggota beserta interval kepercayaan 95%, diurutkan dari yang terbaik."""
    baris = []
    for k, v in pool.items():
        acc = accuracy_score(y, v["val"].argmax(1))
        se = se_akurasi(acc, len(y))
        baris.append(dict(anggota=k, val_acc=acc, ci95_bawah=acc - 1.96 * se, ci95_atas=acc + 1.96 * se))
    return pd.DataFrame(baris).sort_values("val_acc", ascending=False).reset_index(drop=True)


def acc_kombinasi(nama, pool: dict, y: np.ndarray, idx=None, w=None) -> float:
    """Akurasi rata-rata (berbobot ``w``) peluang anggota ``nama`` pada indeks holdout ``idx``."""
    idx = np.arange(len(y)) if idx is None else idx
    w = np.ones(len(nama)) if w is None else np.asarray(w, float)
    p = sum(wi * pool[n]["val"][idx] for wi, n in zip(w, nama))
    return accuracy_score(y[idx], p.argmax(1))


def semua_kombinasi(pool: dict, y: np.ndarray, idx=None) -> pd.DataFrame:
    """Nilai semua kombinasi anggota (rata-rata peluang), diurutkan dari akurasi tertinggi lalu jumlah anggota terkecil."""
    hasil = []
    for r in range(1, len(pool) + 1):
        for komb in itertools.combinations(pool, r):
            hasil.append(dict(kombinasi=" + ".join(komb), n=r, acc=acc_kombinasi(komb, pool, y, idx)))
    return pd.DataFrame(hasil).sort_values(["acc", "n"], ascending=[False, True]).reset_index(drop=True)


def greedy_selection(pool: dict, y: np.ndarray, idx=None, langkah: int = 20) -> tuple[Counter, float]:
    """Greedy ensemble selection (Caruana dkk., 2004) dengan pengembalian: anggota boleh dipilih berulang.

    Returns:
        ``(bobot, akurasi)``; bobot adalah jumlah kali tiap anggota dipilih.
    """
    idx = np.arange(len(y)) if idx is None else idx
    pilih, best_acc, p_sum = [], -1.0, 0
    for _ in range(langkah):
        acc, n = max((accuracy_score(y[idx], (p_sum + pool[n]["val"][idx]).argmax(1)), n) for n in pool)
        if acc < best_acc - 1e-12:
            break
        pilih.append(n)
        p_sum = p_sum + pool[n]["val"][idx]
        best_acc = acc
    return Counter(pilih), best_acc


def split_half(pool: dict, y: np.ndarray, seed: int = 42) -> pd.DataFrame:
    """Estimasi jujur: pilih kombinasi terbaik di separuh holdout lalu nilai di separuh lainnya (dua arah)."""
    perm = np.random.default_rng(seed).permutation(len(y))
    A, B = perm[: len(perm) // 2], perm[len(perm) // 2:]
    hasil = []
    for pilih_di, nilai_di in ((A, B), (B, A)):
        k = semua_kombinasi(pool, y, pilih_di)
        terbaik = k.iloc[0]["kombinasi"].split(" + ")
        tunggal = k.query("n == 1").iloc[0]["kombinasi"].split(" + ")
        hasil.append(dict(kombinasi=" + ".join(terbaik), acc_kombinasi=acc_kombinasi(terbaik, pool, y, nilai_di),
                          tunggal=tunggal[0], acc_tunggal=acc_kombinasi(tunggal, pool, y, nilai_di)))
    return pd.DataFrame(hasil)


def peluang(pool: dict, bobot: dict, bagian: str = "val") -> np.ndarray:
    """Jumlah peluang berbobot ``bobot`` (``{nama: bobot}``) pada bagian ``val`` atau ``test``."""
    return sum(w * pool[n][bagian] for n, w in bobot.items())


def pilih_bobot_final(pool: dict, y: np.ndarray, seed: int = 42) -> tuple[dict, dict]:
    """Pilih bobot akhir: rata-rata kombinasi terbaik, diganti bobot greedy hanya kalau kenaikannya nyata.

    Returns:
        ``(bobot_final, info)``; ``info`` berisi akurasi dan hasil uji berpasangan untuk dilaporkan.
    """
    komb = semua_kombinasi(pool, y).iloc[0]["kombinasi"].split(" + ")
    bobot_g, acc_g = greedy_selection(pool, y)
    p_rata = peluang(pool, {n: 1 for n in komb})
    p_bobot = peluang(pool, dict(bobot_g))
    u = bandingkan(p_rata.argmax(1), p_bobot.argmax(1), y, seed=seed)
    final = dict(bobot_g) if (u["nyata"] and u["selisih"] > 0) else {n: 1 for n in komb}
    info = dict(kombinasi_rata=komb, acc_rata=accuracy_score(y, p_rata.argmax(1)), bobot_greedy=dict(bobot_g),
                acc_greedy=acc_g, uji=u, acc_final=accuracy_score(y, peluang(pool, final).argmax(1)))
    return final, info


def tulis_submisi(test_ids, p_test: np.ndarray, path: Path, n_kelas: int = 16) -> pd.DataFrame:
    """Tulis ``submission.csv`` berformat ``id,label`` (label integer 0..15) dan kembalikan DataFrame-nya."""
    sub = pd.DataFrame({"id": np.asarray(test_ids), "label": p_test.argmax(1).astype(int)})
    assert sub["label"].between(0, n_kelas - 1).all() and list(sub.columns) == ["id", "label"]
    sub.to_csv(path, index=False)
    return sub


def beda_submisi(sub: pd.DataFrame, folder: Path) -> list[tuple[str, int]]:
    """Bandingkan ``sub`` dengan submission lama di ``folder`` (sub-diff); kembalikan jumlah label yang berbeda."""
    hasil = []
    for f in sorted(Path(folder).glob("*.csv")):
        lama = pd.read_csv(f).set_index("id").loc[sub["id"], "label"].values
        hasil.append((f.name, int((lama != sub["label"].values).sum())))
    return hasil
