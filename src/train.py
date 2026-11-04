"""Training, prediksi (termasuk TTA), dan pencatatan eksperimen untuk setiap anggota ensemble."""
from __future__ import annotations

import gc
import math
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import seed_everything
from .data import Konteks
from .dataset import DonutDataset, buat_loader
from .ensemble import softmax_np
from .models import buat_model, grup_parameter

BATAS_RAM_PERSEN = 88


def info_memori(device: str) -> str:
    """Ringkasan pemakaian RAM dan memori GPU untuk log."""
    import psutil
    vm = psutil.virtual_memory()
    teks = f"RAM {vm.used / 1e9:.1f}/{vm.total / 1e9:.0f} GB"
    if device == "cuda":
        teks += " | GPU " + ", ".join(f"{torch.cuda.memory_allocated(i) / 1e9:.1f} GB"
                                      for i in range(torch.cuda.device_count()))
    return teks


def cek_ram() -> None:
    """Hentikan anggota dengan ``MemoryError`` sebelum Kaggle membunuh kernel karena RAM habis."""
    import psutil
    if psutil.virtual_memory().percent > BATAS_RAM_PERSEN:
        raise MemoryError(f"RAM {psutil.virtual_memory().percent:.0f}% > {BATAS_RAM_PERSEN}%, anggota dihentikan")


def bersihkan_memori(device: str) -> None:
    """Jalankan garbage collector dan kosongkan cache CUDA."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def prediksi(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """Hitung logit untuk seluruh isi ``loader`` dengan mixed precision di GPU."""
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            out.append(model(x.to(device, non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def prediksi_donut_df(model: nn.Module, df: pd.DataFrame, anggota: dict, ctx: Konteks, nama: str) -> np.ndarray:
    """Skor Donut untuk semua baris ``df`` (dari TIFF asli), dengan log perkiraan sisa waktu."""
    proc = (model.module if isinstance(model, nn.DataParallel) else model).proc
    dev = ctx.cfg.device
    dl = DataLoader(DonutDataset(df["path"].values, proc), batch_size=anggota["bs"], shuffle=False,
                    num_workers=ctx.cfg.num_workers, pin_memory=dev == "cuda")
    model.eval()
    out, t0 = [], time.time()
    for i, (x, _) in enumerate(dl):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            out.append(model(x.to(dev, non_blocking=True)).float().cpu())
        if i == 9 or (i + 1) % max(1, len(dl) // 4) == 0:
            el = time.time() - t0
            print(f"  [{anggota['name']}] {nama} {i + 1}/{len(dl)} batch | {el / 60:.1f} mnt, perkiraan sisa "
                  f"{el / (i + 1) * (len(dl) - i - 1) / 60:.1f} mnt | {info_memori(dev)}", flush=True)
        if (i + 1) % 20 == 0:
            cek_ram()
    return torch.cat(out).numpy()


def prediksi_donut(anggota: dict, ctx: Konteks) -> float:
    """Jalankan Donut tanpa training pada holdout dan test, simpan logit, kembalikan akurasi holdout."""
    cfg = ctx.cfg
    out_dir = cfg.artefak / anggota["name"]
    inti, _, _ = buat_model(anggota, ctx.classes, cfg.pretrained)
    inti = inti.to(cfg.device)
    model = nn.DataParallel(inti) if torch.cuda.device_count() > 1 else inti
    print(f"[{anggota['name']}] input {anggota['size'][0]}x{anggota['size'][1]} | {info_memori(cfg.device)}", flush=True)
    vl = prediksi_donut_df(model, ctx.train_df.iloc[ctx.va_idx], anggota, ctx, "holdout")
    np.save(out_dir / "val_logits.npy", vl)
    acc = accuracy_score(ctx.y_va, vl.argmax(1))
    print(f"[{anggota['name']}] tanpa training | val acc {acc:.4f}", flush=True)
    np.save(out_dir / "test_logits.npy", prediksi_donut_df(model, ctx.test_df, anggota, ctx, "test"))
    del model, inti
    return acc


def simpan_tta(model: nn.Module, anggota: dict, ctx: Konteks, mean, std, skala=(1.06, 1.12)) -> None:
    """Simpan logit TTA (rata-rata peluang tampilan asli + tampilan diperbesar, tanpa flip) untuk holdout dan test."""
    cfg = ctx.cfg
    out_dir = cfg.artefak / anggota["name"]
    for nama, df in (("val", ctx.train_df.iloc[ctx.va_idx]), ("test", ctx.test_df)):
        prob = softmax_np(np.load(out_dir / f"{nama}_logits.npy"))
        for s in skala:
            dl = buat_loader(anggota, df, False, mean, std, cfg.num_workers, cfg.device, tta=s)
            prob += softmax_np(prediksi(model, dl, cfg.device))
            del dl
        np.save(out_dir / f"{nama}_logits_tta.npy", np.log(prob / (1 + len(skala)) + 1e-9))


def latih_anggota(anggota: dict, ctx: Konteks) -> float:
    """Latih (atau pakai apa adanya) satu anggota, lalu simpan logit holdout/test, TTA, checkpoint, dan ledger.

    Resep: AdamW (weight decay 0,05), warmup lalu cosine decay, AMP, gradient clipping 1,0, label smoothing 0,1,
    layer-wise LR decay untuk DiT, dan DataParallel untuk model transformers bila ada lebih dari satu GPU.
    Untuk checkpoint RVL-CDIP, epoch 0 (checkpoint apa adanya) ikut dibandingkan sehingga training hanya dipakai
    kalau memang lebih baik di holdout.

    Returns:
        Akurasi holdout terbaik.
    """
    cfg = ctx.cfg
    out_dir = cfg.artefak / anggota["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = cfg.checkpoint_dir / anggota["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if cfg.lewati_jika_ada and (out_dir / "test_logits.npy").exists() and (out_dir / "val_logits.npy").exists():
        print(f"[{anggota['name']}] logit sudah ada, dilewati")
        return float(accuracy_score(ctx.y_va, np.load(out_dir / "val_logits.npy").argmax(1)))
    seed_everything(cfg.seed)
    if anggota["backend"] == "donut":
        acc = prediksi_donut(anggota, ctx)
        catat_ledger(anggota, acc, ctx)
        bersihkan_memori(cfg.device)
        return acc

    dev = cfg.device
    inti, mean, std = buat_model(anggota, ctx.classes, cfg.pretrained)
    if anggota.get("ckpt"):
        inti.set_grad_checkpointing(True)
    inti = inti.to(dev).to(memory_format=torch.channels_last)
    # DataParallel hanya untuk DiT: pada CNN timm di Kaggle, DP membuat training lambat dan RAM terus naik.
    pakai_dp = torch.cuda.device_count() > 1 and anggota.get("dp", anggota["backend"] == "hf")
    model = nn.DataParallel(inti) if pakai_dp else inti
    print(f"[{anggota['name']}] {'DataParallel' if pakai_dp else '1 perangkat'} | {info_memori(dev)}", flush=True)
    accum = anggota.get("accum", 1)
    dl_tr = buat_loader(anggota, ctx.train_df.iloc[ctx.tr_idx], True, mean, std, cfg.num_workers, dev)
    dl_va = buat_loader(anggota, ctx.train_df.iloc[ctx.va_idx], False, mean, std, cfg.num_workers, dev)

    opt = torch.optim.AdamW(grup_parameter(inti, anggota), lr=anggota["lr"], weight_decay=0.05)
    total = anggota["epochs"] * (len(dl_tr) // accum)
    warm = min(len(dl_tr), total // 5 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = (torch.amp.GradScaler("cuda", enabled=dev == "cuda") if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=dev == "cuda"))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, riwayat = -1.0, []
    if anggota.get("head_rvl") or anggota["epochs"] == 0:
        vl = prediksi(model, dl_va, dev)
        best = accuracy_score(ctx.y_va, vl.argmax(1))
        torch.save(inti.state_dict(), ckpt_dir / "best.pt")
        np.save(out_dir / "val_logits.npy", vl)
        riwayat.append(dict(epoch=0, loss=float("nan"), val_acc=best, menit=0.0))
        print(f"[{anggota['name']}] epoch 0 (checkpoint apa adanya): val acc {best:.4f}", flush=True)
    for ep in range(anggota["epochs"]):
        model.train()
        t0, rl = time.time(), 0.0
        opt.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(tqdm(dl_tr, desc=f"{anggota['name']} ep{ep + 1}", leave=False)):
            x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                loss = crit(model(x), y) / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            rl += loss.item() * accum
            if (step + 1) % 20 == 0:
                cek_ram()
            if (step + 1) % max(1, len(dl_tr) // 4) == 0:
                print(f"  [{anggota['name']}] ep{ep + 1} {step + 1}/{len(dl_tr)} | {(time.time() - t0) / 60:.1f} mnt"
                      f" | {info_memori(dev)}", flush=True)
        vl = prediksi(model, dl_va, dev)
        acc = accuracy_score(ctx.y_va, vl.argmax(1))
        riwayat.append(dict(epoch=ep + 1, loss=rl / len(dl_tr), val_acc=acc, menit=(time.time() - t0) / 60))
        print(f"[{anggota['name']}] epoch {ep + 1}: loss {rl / len(dl_tr):.4f} | val acc {acc:.4f} | "
              f"{riwayat[-1]['menit']:.1f} mnt", flush=True)
        if acc > best:
            best = acc
            torch.save(inti.state_dict(), ckpt_dir / "best.pt")
            np.save(out_dir / "val_logits.npy", vl)
    pd.DataFrame(riwayat).to_csv(out_dir / "riwayat.csv", index=False)
    inti.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=dev))
    dl_te = buat_loader(anggota, ctx.test_df, False, mean, std, cfg.num_workers, dev)
    np.save(out_dir / "test_logits.npy", prediksi(model, dl_te, dev))
    if cfg.pakai_tta:
        simpan_tta(model, anggota, ctx, mean, std)
    catat_ledger(anggota, best, ctx)
    del model, inti, opt, dl_tr, dl_va, dl_te
    bersihkan_memori(dev)
    return best


def catat_ledger(anggota: dict, acc: float, ctx: Konteks) -> None:
    """Tambah satu baris ke ``artefak/ledger.csv`` (experiment ledger): konfigurasi, akurasi, akurasi per kelas."""
    f = ctx.cfg.artefak / "ledger.csv"
    vl = np.load(ctx.cfg.artefak / anggota["name"] / "val_logits.npy").argmax(1)
    y = ctx.y_va
    per_kelas = {f"acc_{ctx.classes[c]}": round(float((vl[y == c] == c).mean()), 4) for c in range(ctx.num_classes)}
    baris = dict(waktu=time.strftime("%Y-%m-%d %H:%M"), nama=anggota["name"], backbone=anggota["hub"],
                 view=anggota["view"], size="x".join(map(str, anggota["size"])), epochs=anggota["epochs"],
                 lr=anggota["lr"], augmentasi="tanpa training" if anggota["backend"] == "donut"
                 else "affine+jitter+blur+jpeg+noise, tanpa flip", val_acc=round(float(acc), 4),
                 n_val=len(y), smoke=ctx.cfg.smoke, **per_kelas)
    pd.concat([pd.read_csv(f) if f.exists() else pd.DataFrame(), pd.DataFrame([baris])]).to_csv(f, index=False)


def tulis_submisi_sementara(ctx: Konteks) -> int:
    """Tulis ``submission.csv`` dari rata-rata peluang semua anggota yang sudah selesai.

    Menjamin selalu ada file submission walaupun anggota berikutnya gagal. Ditimpa oleh kombinasi terbaik di akhir.

    Returns:
        Jumlah anggota yang dipakai.
    """
    p = []
    for d in sorted(ctx.cfg.artefak.iterdir()):
        f = d / "test_logits.npy"
        if d.is_dir() and f.exists():
            z = np.load(f)
            if len(z) == len(ctx.test_df):
                p.append(softmax_np(z))
    if p:
        pd.DataFrame({"id": ctx.test_df["id"].values, "label": sum(p).argmax(1).astype(int)}).to_csv(
            ctx.cfg.submission_path, index=False)
        print(f"submission.csv sementara ditulis dari {len(p)} anggota", flush=True)
    return len(p)


def latih_semua(daftar: list[dict], ctx: Konteks) -> list[str]:
    """Latih semua anggota berurutan dengan batas waktu ``cfg.batas_jam``; error satu anggota tidak menghentikan run.

    Returns:
        Nama anggota yang gagal atau dilewati.
    """
    cfg = ctx.cfg
    gagal = []
    for anggota in daftar:
        bersihkan_memori(cfg.device)
        sudah_ada = (cfg.artefak / anggota["name"] / "test_logits.npy").exists()
        if cfg.jam_berjalan() > cfg.batas_jam and not sudah_ada:
            print(f"[{anggota['name']}] dilewati: melewati batas {cfg.batas_jam} jam")
            gagal.append(anggota["name"])
            continue
        try:
            latih_anggota(anggota, ctx)
        except Exception as e:  # noqa: BLE001 - satu anggota gagal tidak menghentikan pipeline
            traceback.print_exc()
            print(f"[{anggota['name']}] GAGAL ({type(e).__name__}), dilewati.")
            gagal.append(anggota["name"])
            bersihkan_memori(cfg.device)
        tulis_submisi_sementara(ctx)
        print(f"waktu berjalan: {cfg.jam_berjalan() * 60:.0f} menit | {info_memori(cfg.device)}", flush=True)
    assert tulis_submisi_sementara(ctx) > 0, "tidak ada anggota yang berhasil, cek log"
    return gagal
