"""Visualisasi EDA dan hasil model. Setiap fungsi bisa menyimpan gambar ke ``reports/`` untuk dikutip README."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix

from .dataset import ambil_view, baca_cache, buat_transform


def _selesai(fig, simpan: Path | None) -> None:
    """Rapikan tata letak, simpan ke ``simpan`` (kalau diberikan), lalu tampilkan."""
    fig.tight_layout()
    if simpan is not None:
        Path(simpan).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(simpan, dpi=110, bbox_inches="tight")
    plt.show()


def plot_distribusi(train_df: pd.DataFrame, classes: dict, simpan: Path | None = None) -> pd.Series:
    """Plot jumlah gambar per kelas, lebar gambar, dan ukuran file. Mengembalikan jumlah per kelas."""
    fig, ax = plt.subplots(1, 3, figsize=(17, 4))
    vc = train_df["class_name"].value_counts().reindex(list(classes.values()))
    ax[0].barh(vc.index, vc.values, color="#4C72B0")
    ax[0].invert_yaxis()
    ax[0].set_title("Jumlah gambar per kelas")
    ax[1].hist(train_df["w"], bins=40, color="#55A868")
    ax[1].set_title("Lebar gambar (tinggi hampir selalu 1000 px)")
    ax[2].hist(train_df["file_kb"], bins=50, color="#C44E52")
    ax[2].set_title("Ukuran file (KB)")
    _selesai(fig, simpan)
    return vc


def plot_box_per_kelas(train_df: pd.DataFrame, classes: dict, simpan: Path | None = None) -> None:
    """Boxplot proporsi tinta dan lebar gambar per kelas."""
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    order = list(classes.values())
    sns.boxplot(data=train_df, y="class_name", x="ink", order=order, ax=ax[0], fliersize=1)
    ax[0].set_title("Proporsi piksel gelap (tinta) per kelas")
    sns.boxplot(data=train_df, y="class_name", x="w", order=order, ax=ax[1], fliersize=1)
    ax[1].set_title("Lebar gambar per kelas")
    _selesai(fig, simpan)


def plot_contoh_per_kelas(train_df: pd.DataFrame, classes: dict, n_per: int = 4, seed: int = 42,
                          simpan: Path | None = None) -> None:
    """Tampilkan ``n_per`` contoh halaman untuk setiap kelas."""
    n = len(classes)
    fig, axes = plt.subplots(n, n_per, figsize=(n_per * 2.2, n * 2.6))
    for c in range(n):
        sub = train_df[train_df["label"] == c]
        sub = sub.sample(min(n_per, len(sub)), random_state=seed)
        for j in range(n_per):
            a = axes[c, j]
            a.axis("off")
            if j < len(sub):
                a.imshow(baca_cache(sub["cache"].iloc[j]), cmap="gray")
            if j == 0:
                a.set_title(f"{c}: {classes[c]}", fontsize=9, loc="left")
    _selesai(fig, simpan)


def plot_rata2_halaman(train_df: pd.DataFrame, classes: dict, n_maks: int = 150, seed: int = 42,
                       simpan: Path | None = None) -> None:
    """Rata-rata piksel halaman per kelas: memperlihatkan pola tata letak khas tiap kelas."""
    fig, axes = plt.subplots(2, 8, figsize=(18, 6))
    for c, a in zip(range(len(classes)), axes.ravel()):
        ps = train_df.loc[train_df["label"] == c, "cache"]
        ps = ps.sample(min(n_maks, len(ps)), random_state=seed)
        m = np.mean([np.asarray(baca_cache(p).resize((96, 128))) / 255.0 for p in ps], axis=0)
        a.imshow(m, cmap="gray", vmin=0.4, vmax=1)
        a.set_title(classes[c], fontsize=8)
        a.axis("off")
    fig.suptitle("Rata-rata halaman per kelas")
    _selesai(fig, simpan)


def plot_train_vs_test(train_df: pd.DataFrame, test_df: pd.DataFrame, simpan: Path | None = None) -> pd.DataFrame:
    """Bandingkan contoh gambar train dan test secara visual; kembalikan rata-rata statistik keduanya."""
    fig, axes = plt.subplots(2, 10, figsize=(20, 5.5))
    for a, p in zip(axes[0], train_df["cache"].sample(min(10, len(train_df)), random_state=1)):
        a.imshow(baca_cache(p), cmap="gray")
        a.set_title("train", fontsize=8)
    for a, p in zip(axes[1], test_df["cache"].sample(min(10, len(test_df)), random_state=1)):
        a.imshow(baca_cache(p), cmap="gray")
        a.set_title("test", fontsize=8)
    for a in axes.ravel():
        a.axis("off")
    _selesai(fig, simpan)
    kol = ["mean", "ink", "w", "file_kb"]
    return pd.DataFrame({"train": train_df[kol].mean(), "test": test_df[kol].mean()}).round(3)


def plot_augmentasi(path_cache: str, simpan: Path | None = None) -> None:
    """Tampilkan satu halaman asli, lima hasil augmentasi, dan view header."""
    tf = buat_transform((384, 300), (0.0,) * 3, (1.0,) * 3, train=True)
    asli = baca_cache(path_cache)
    fig, axes = plt.subplots(1, 7, figsize=(19, 4), gridspec_kw=dict(width_ratios=[1] * 6 + [1.6]))
    axes[0].imshow(asli, cmap="gray")
    axes[0].set_title("asli")
    for a in axes[1:6]:
        a.imshow(tf(asli)[0], cmap="gray")
        a.set_title("augmentasi")
    axes[6].imshow(ambil_view(asli, "header"), cmap="gray")
    axes[6].set_title("view header")
    for a in axes:
        a.axis("off")
    _selesai(fig, simpan)


def plot_confusion(y: np.ndarray, pred: np.ndarray, classes: dict, simpan: Path | None = None) -> np.ndarray:
    """Confusion matrix ternormalisasi per baris (anotasi = jumlah). Mengembalikan matriks jumlah."""
    n = len(classes)
    nama = [classes[c] for c in range(n)]
    cm = confusion_matrix(y, pred, labels=list(range(n)))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(cmn, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1, cbar_kws={"label": "proporsi per baris"},
                xticklabels=nama, yticklabels=nama)
    plt.xlabel("prediksi")
    plt.ylabel("label asli")
    plt.title("Confusion matrix ensemble (holdout)")
    _selesai(fig, simpan)
    return cm


def pasangan_tertukar(cm: np.ndarray, classes: dict, top: int = 12) -> pd.DataFrame:
    """Daftar pasangan kelas (asli -> prediksi) yang paling sering tertukar."""
    n = len(classes)
    salah = [(classes[i], classes[j], int(cm[i, j])) for i in range(n) for j in range(n) if i != j and cm[i, j] > 0]
    return pd.DataFrame(sorted(salah, key=lambda t: -t[2])[:top], columns=["asli", "prediksi", "jumlah"])


def plot_kurva_training(artefak: Path, simpan: Path | None = None) -> None:
    """Plot loss training dan akurasi holdout per epoch dari ``artefak/<nama>/riwayat.csv``."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    for d in sorted(Path(artefak).iterdir()):
        if (d / "riwayat.csv").exists():
            r = pd.read_csv(d / "riwayat.csv")
            ax[0].plot(r["epoch"], r["loss"], marker="o", label=d.name)
            ax[1].plot(r["epoch"], r["val_acc"], marker="o", label=d.name)
    ax[0].set_title("loss training")
    ax[1].set_title("akurasi holdout")
    ax[1].legend(fontsize=8)
    _selesai(fig, simpan)


def plot_akurasi_per_kelas(pool: dict, pred_ens: np.ndarray, y: np.ndarray, classes: dict,
                           simpan: Path | None = None) -> pd.DataFrame:
    """Heatmap akurasi per kelas untuk tiap anggota dan ensemble. Mengembalikan tabelnya."""
    n = len(classes)
    pk = pd.DataFrame({k: [(v["val"].argmax(1)[y == c] == c).mean() for c in range(n)] for k, v in pool.items()},
                      index=[classes[c] for c in range(n)])
    pk["ENSEMBLE"] = [(pred_ens[y == c] == c).mean() for c in range(n)]
    fig = plt.figure(figsize=(1.6 * pk.shape[1] + 4, 7))
    sns.heatmap(pk, annot=True, fmt=".2f", cmap="RdYlGn", vmin=max(0, pk.values.min() - 0.02), vmax=1)
    plt.title("Akurasi per kelas di holdout")
    _selesai(fig, simpan)
    return pk


def plot_salah_prediksi(paths_cache, y: np.ndarray, pred: np.ndarray, conf: np.ndarray, classes: dict,
                        n: int = 16, simpan: Path | None = None) -> None:
    """Tampilkan gambar salah prediksi dengan keyakinan model tertinggi (banyak di antaranya label berderau)."""
    salah = np.where(pred != y)[0]
    pilih = salah[np.argsort(-conf[salah])][:n]
    if not len(pilih):
        print("tidak ada salah prediksi")
        return
    fig, axes = plt.subplots(2, 8, figsize=(20, 7))
    for a, i in zip(axes.ravel(), pilih):
        a.imshow(baca_cache(paths_cache[i]), cmap="gray")
        a.set_title(f"asli: {classes[y[i]]}\npred: {classes[pred[i]]} ({conf[i]:.2f})", fontsize=8)
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle("Salah prediksi dengan keyakinan tertinggi")
    _selesai(fig, simpan)


def gradcam(model: torch.nn.Module, x: torch.Tensor, target: int) -> np.ndarray:
    """Peta Grad-CAM (0..1) untuk model CNN timm pada kelas ``target``; ``x`` berukuran ``(1, 3, H, W)``."""
    feats = model.forward_features(x)
    feats.retain_grad()
    out = model.forward_head(feats)
    model.zero_grad()
    out[0, target].backward()
    cam = F.relu((feats.grad.mean((2, 3), keepdim=True) * feats).sum(1))[0]
    return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()


def plot_gradcam(model, anggota: dict, mean, std, paths_cache, labels, classes: dict,
                 simpan: Path | None = None) -> None:
    """Tampilkan Grad-CAM di atas halaman untuk beberapa contoh (satu per kelas)."""
    tfm = buat_transform(anggota["size"], mean, std, train=False)
    h, w = anggota["size"]
    fig, axes = plt.subplots(1, len(paths_cache), figsize=(2.6 * len(paths_cache), 4))
    for a, p, lab in zip(np.atleast_1d(axes), paths_cache, labels):
        im = ambil_view(baca_cache(p), anggota["view"])
        cam = gradcam(model, tfm(im).unsqueeze(0).requires_grad_(True), int(lab))
        a.imshow(im.resize((w, h)), cmap="gray")
        a.imshow(np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h))), cmap="jet", alpha=0.4)
        a.set_title(classes[int(lab)], fontsize=8)
        a.axis("off")
    fig.suptitle(f"Grad-CAM {anggota['name']}")
    _selesai(fig, simpan)


# ======================================================================
# Visualisasi lanjutan: tata letak, duplikat, ruang fitur, kalibrasi, keragaman anggota
# ======================================================================

def plot_scatter_meta(train_df: pd.DataFrame, classes: dict, n_per: int = 150, seed: int = 42,
                      simpan: Path | None = None) -> None:
    """Sebaran lebar halaman vs proporsi tinta dan ukuran file vs proporsi tinta, diwarnai per kelas."""
    sub = pd.concat([g.sample(min(n_per, len(g)), random_state=seed) for _, g in train_df.groupby("label")])
    fig, ax = plt.subplots(1, 2, figsize=(17, 6))
    pal = sns.color_palette("tab20", len(classes))
    for c in range(len(classes)):
        s = sub[sub["label"] == c]
        ax[0].scatter(s["w"], s["ink"], s=8, alpha=0.6, color=pal[c], label=classes[c])
        ax[1].scatter(s["file_kb"], s["ink"], s=8, alpha=0.6, color=pal[c])
    ax[0].set_xlabel("lebar (px)"); ax[0].set_ylabel("proporsi tinta"); ax[0].set_title("Lebar vs tinta")
    ax[1].set_xlabel("ukuran file (KB)"); ax[1].set_xscale("log"); ax[1].set_title("Ukuran file vs tinta")
    ax[0].legend(fontsize=7, ncol=2, markerscale=2)
    _selesai(fig, simpan)


def plot_korelasi_meta(train_df: pd.DataFrame, simpan: Path | None = None) -> pd.DataFrame:
    """Heatmap korelasi Spearman antar fitur metadata (lebar, tinggi, ukuran file, intensitas, tinta)."""
    kol = ["w", "h", "file_kb", "mean", "ink"]
    kor = train_df[kol].astype(float).corr(method="spearman")
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(kor, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1)
    plt.title("Korelasi Spearman metadata")
    _selesai(fig, simpan)
    return kor


def profil_tinta(path_cache: str, n_bin: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Proyeksi tinta per baris dan per kolom halaman (``n_bin`` titik, 0..1): tanda tangan tata letak."""
    a = 1 - np.asarray(baca_cache(path_cache).resize((n_bin, n_bin)), dtype=np.float32) / 255.0
    return a.mean(1), a.mean(0)


def plot_profil_tinta(train_df: pd.DataFrame, classes: dict, n_per: int = 80, seed: int = 42,
                      simpan: Path | None = None) -> None:
    """Heatmap rata-rata profil tinta vertikal (atas ke bawah) dan horizontal (kiri ke kanan) per kelas."""
    n = len(classes)
    vert, hor = np.zeros((n, 100)), np.zeros((n, 100))
    for c in range(n):
        ps = train_df.loc[train_df["label"] == c, "cache"]
        prof = [profil_tinta(p) for p in ps.sample(min(n_per, len(ps)), random_state=seed)]
        vert[c] = np.mean([v for v, _ in prof], 0)
        hor[c] = np.mean([h for _, h in prof], 0)
    nama = [classes[c] for c in range(n)]
    fig, ax = plt.subplots(1, 2, figsize=(17, 6))
    sns.heatmap(vert, ax=ax[0], cmap="magma_r", yticklabels=nama, xticklabels=20, cbar_kws={"label": "tinta"})
    ax[0].set_xlabel("posisi vertikal halaman (0 = atas, 99 = bawah)")
    ax[0].set_title("Profil tinta atas ke bawah")
    sns.heatmap(hor, ax=ax[1], cmap="magma_r", yticklabels=nama, xticklabels=20, cbar_kws={"label": "tinta"})
    ax[1].set_xlabel("posisi horizontal halaman (0 = kiri, 99 = kanan)")
    ax[1].set_title("Profil tinta kiri ke kanan")
    _selesai(fig, simpan)


def plot_pasangan_konflik(train_df: pd.DataFrame, classes: dict, n: int = 6, simpan: Path | None = None) -> int:
    """Tampilkan pasangan gambar kembar yang labelnya berbeda (bukti derau label). Mengembalikan jumlah grup konflik."""
    grup = [g for _, g in train_df[train_df["konflik"]].groupby("grp")]
    if not grup:
        print("tidak ada grup kembar berlabel konflik")
        return 0
    pilih = grup[:n]
    fig, axes = plt.subplots(2, len(pilih), figsize=(2.8 * len(pilih), 7), squeeze=False)
    for j, g in enumerate(pilih):
        g = g.drop_duplicates("label").head(2)
        for i in range(2):
            a = axes[i, j]
            a.axis("off")
            if i < len(g):
                a.imshow(baca_cache(g["cache"].iloc[i]), cmap="gray")
                a.set_title(f"grup {j + 1}: {classes[int(g['label'].iloc[i])]}", fontsize=8)
    fig.suptitle("Gambar kembar dengan label berbeda (dibuang dari data latih)")
    _selesai(fig, simpan)
    return len(grup)


def plot_tsne(prob: np.ndarray, y: np.ndarray, classes: dict, seed: int = 42, n_maks: int = 4000,
              simpan: Path | None = None) -> np.ndarray:
    """Proyeksi t-SNE 2D dari vektor peluang ensemble di holdout, diwarnai label asli; salah prediksi ditandai x."""
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))[:n_maks]
    z = np.log(prob[idx] / prob[idx].sum(1, keepdims=True) + 1e-6)
    emb = TSNE(n_components=2, perplexity=min(30, max(5, len(idx) // 5)), init="pca", random_state=seed).fit_transform(z)
    pred = prob[idx].argmax(1)
    pal = sns.color_palette("tab20", len(classes))
    fig = plt.figure(figsize=(12, 9))
    for c in range(len(classes)):
        m = y[idx] == c
        plt.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.6, color=pal[c], label=classes[c])
        cx, cy = np.median(emb[m], 0) if m.any() else (np.nan, np.nan)
        plt.text(cx, cy, classes[c], fontsize=8, weight="bold")
    salah = pred != y[idx]
    plt.scatter(emb[salah, 0], emb[salah, 1], s=14, marker="x", color="black", linewidths=0.6, label="salah prediksi")
    plt.legend(fontsize=7, ncol=2, markerscale=2, loc="best")
    plt.title("t-SNE peluang ensemble di holdout")
    plt.xticks([]); plt.yticks([])
    _selesai(fig, simpan)
    return emb


def tabel_topk(prob: np.ndarray, y: np.ndarray, ks=(1, 2, 3)) -> pd.Series:
    """Akurasi top-k: proporsi gambar yang label aslinya ada di k tebakan teratas."""
    urut = np.argsort(-prob, 1)
    return pd.Series({f"top-{k}": float((urut[:, :k] == y[:, None]).any(1).mean()) for k in ks})


def plot_kalibrasi(prob: np.ndarray, y: np.ndarray, n_bin: int = 10, simpan: Path | None = None) -> float:
    """Reliability diagram dan histogram keyakinan (benar vs salah). Mengembalikan Expected Calibration Error."""
    p = prob / prob.sum(1, keepdims=True)
    conf, pred = p.max(1), p.argmax(1)
    benar = pred == y
    batas = np.linspace(0, 1, n_bin + 1)
    ece, xs, ys, ns = 0.0, [], [], []
    for lo, hi in zip(batas[:-1], batas[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean()); ys.append(benar[m].mean()); ns.append(m.sum())
            ece += m.mean() * abs(conf[m].mean() - benar[m].mean())
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot([0, 1], [0, 1], "--", color="gray", label="kalibrasi sempurna")
    ax[0].plot(xs, ys, marker="o", label=f"ensemble (ECE {ece:.3f})")
    ax[0].set_xlabel("keyakinan rata-rata"); ax[0].set_ylabel("akurasi"); ax[0].set_title("Reliability diagram")
    ax[0].legend()
    ax[1].hist(conf[benar], bins=30, alpha=0.7, label="benar", color="#55A868")
    ax[1].hist(conf[~benar], bins=30, alpha=0.7, label="salah", color="#C44E52")
    ax[1].set_yscale("log"); ax[1].set_xlabel("keyakinan"); ax[1].set_title("Sebaran keyakinan")
    ax[1].legend()
    _selesai(fig, simpan)
    return float(ece)


def plot_cakupan(prob: np.ndarray, y: np.ndarray, simpan: Path | None = None) -> pd.DataFrame:
    """Akurasi vs cakupan: kalau hanya gambar paling yakin yang dijawab, seberapa akurat? (analisis risiko-cakupan)."""
    p = prob / prob.sum(1, keepdims=True)
    urut = np.argsort(-p.max(1))
    benar = (p.argmax(1) == y)[urut]
    cak = np.arange(1, len(y) + 1) / len(y)
    akur = np.cumsum(benar) / np.arange(1, len(y) + 1)
    fig = plt.figure(figsize=(7, 5))
    plt.plot(cak, akur)
    plt.xlabel("cakupan (proporsi gambar yang dijawab)"); plt.ylabel("akurasi")
    plt.title("Akurasi vs cakupan"); plt.grid(alpha=0.3)
    _selesai(fig, simpan)
    return pd.DataFrame({"cakupan": [0.5, 0.8, 0.9, 0.95, 1.0],
                         "akurasi": [akur[max(0, int(c * len(y)) - 1)] for c in (0.5, 0.8, 0.9, 0.95, 1.0)]})


def plot_keragaman_anggota(pool: dict, y: np.ndarray, simpan: Path | None = None) -> pd.DataFrame:
    """Keragaman anggota ensemble: proporsi prediksi berbeda dan korelasi kesalahan untuk setiap pasangan anggota.

    Ensemble paling untung kalau anggotanya sering berbeda pendapat dan kesalahannya tidak berkorelasi.
    """
    nama = list(pool)
    pred = {k: v["val"].argmax(1) for k, v in pool.items()}
    salah = {k: (pred[k] != y).astype(float) for k in nama}
    beda = pd.DataFrame([[float((pred[a] != pred[b]).mean()) for b in nama] for a in nama], index=nama, columns=nama)
    kor = pd.DataFrame([[float(np.corrcoef(salah[a], salah[b])[0, 1]) if salah[a].std() and salah[b].std() else np.nan
                         for b in nama] for a in nama], index=nama, columns=nama)
    fig, ax = plt.subplots(1, 2, figsize=(6 + 1.6 * len(nama), 4 + 0.6 * len(nama)))
    sns.heatmap(beda, annot=True, fmt=".3f", cmap="Purples", ax=ax[0])
    ax[0].set_title("Proporsi prediksi berbeda")
    sns.heatmap(kor, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, ax=ax[1])
    ax[1].set_title("Korelasi kesalahan")
    _selesai(fig, simpan)
    return beda


def plot_metrik_per_kelas(y: np.ndarray, pred: np.ndarray, classes: dict, simpan: Path | None = None) -> pd.DataFrame:
    """Diagram batang precision, recall, dan F1 per kelas, diurutkan dari F1 terendah."""
    from sklearn.metrics import precision_recall_fscore_support
    n = len(classes)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(range(n)), zero_division=0)
    df = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=[classes[c] for c in range(n)])
    df = df.sort_values("f1")
    fig = plt.figure(figsize=(12, 7))
    df[["precision", "recall", "f1"]].plot.barh(ax=plt.gca(), width=0.8)
    plt.xlim(max(0, df[["precision", "recall", "f1"]].values.min() - 0.05), 1)
    plt.title("Precision, recall, F1 per kelas (holdout)")
    _selesai(fig, simpan)
    return df


def plot_gain_ensemble(pool: dict, pred_ens: np.ndarray, y: np.ndarray, classes: dict,
                       simpan: Path | None = None) -> pd.Series:
    """Selisih akurasi per kelas antara ensemble dan anggota tunggal terbaik (poin persen)."""
    from sklearn.metrics import accuracy_score
    terbaik = max(pool, key=lambda k: accuracy_score(y, pool[k]["val"].argmax(1)))
    pt = pool[terbaik]["val"].argmax(1)
    n = len(classes)
    gain = pd.Series({classes[c]: 100 * ((pred_ens[y == c] == c).mean() - (pt[y == c] == c).mean())
                      for c in range(n) if (y == c).any()}).sort_values()
    fig = plt.figure(figsize=(9, 6))
    gain.plot.barh(color=["#C44E52" if v < 0 else "#55A868" for v in gain.values])
    plt.axvline(0, color="black", lw=0.8)
    plt.xlabel(f"poin akurasi (ensemble - {terbaik})")
    plt.title("Kenaikan akurasi per kelas dari ensemble")
    _selesai(fig, simpan)
    return gain


def plot_distribusi_prediksi(sub: pd.DataFrame, train_df: pd.DataFrame, classes: dict,
                             simpan: Path | None = None) -> pd.DataFrame:
    """Bandingkan proporsi kelas prediksi test dengan proporsi kelas di train (cek bias prediksi)."""
    n = len(classes)
    df = pd.DataFrame({"train": train_df["label"].value_counts(normalize=True).reindex(range(n), fill_value=0).values,
                       "prediksi test": sub["label"].value_counts(normalize=True).reindex(range(n), fill_value=0).values},
                      index=[classes[c] for c in range(n)])
    fig = plt.figure(figsize=(10, 6))
    df.plot.barh(ax=plt.gca(), width=0.8)
    plt.axvline(1 / n, color="gray", ls="--", lw=0.8)
    plt.title("Proporsi kelas: train vs prediksi test")
    _selesai(fig, simpan)
    return df


def peta_atensi_dit(model, x: torch.Tensor) -> np.ndarray:
    """Peta atensi token [CLS] lapisan terakhir DiT (rata-rata semua head), dibentuk ulang ke grid patch (0..1)."""
    inti = model.m if hasattr(model, "m") else model
    try:
        inti.config._attn_implementation = "eager"
    except Exception:  # noqa: BLE001 - versi transformers lama tidak punya atribut ini
        pass
    with torch.no_grad():
        out = inti(pixel_values=x, output_attentions=True)
    att = out.attentions[-1][0].mean(0)[0, 1:]          # atensi CLS ke semua patch
    g = int(round(att.numel() ** 0.5))
    a = att[: g * g].reshape(g, g).float().cpu().numpy()
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


def plot_atensi_dit(model, anggota: dict, paths_cache, labels, classes: dict, simpan: Path | None = None) -> None:
    """Tampilkan peta atensi DiT di atas halaman untuk beberapa contoh (bagian yang diperhatikan model)."""
    tfm = buat_transform(anggota["size"], (0.5,) * 3, (0.5,) * 3, train=False)
    h, w = anggota["size"]
    fig, axes = plt.subplots(1, len(paths_cache), figsize=(2.6 * len(paths_cache), 4))
    for a, p, lab in zip(np.atleast_1d(axes), paths_cache, labels):
        im = baca_cache(p)
        dev = next(model.parameters()).device
        peta = peta_atensi_dit(model, tfm(im).unsqueeze(0).to(dev))
        a.imshow(im.resize((w, h)), cmap="gray")
        a.imshow(np.array(Image.fromarray((peta * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)),
                 cmap="jet", alpha=0.45)
        a.set_title(classes[int(lab)], fontsize=8)
        a.axis("off")
    fig.suptitle(f"Peta atensi [CLS] lapisan terakhir {anggota['name']}")
    _selesai(fig, simpan)
