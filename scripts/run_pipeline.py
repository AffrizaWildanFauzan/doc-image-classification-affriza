"""Jalankan seluruh pipeline dari baris perintah (tanpa grafik): data -> training -> ensemble -> submission.csv.

Contoh:
    python scripts/run_pipeline.py                     # profil default rvlcdip_ft
    python scripts/run_pipeline.py --profil dokumen --batas-jam 1.5
    DOC_SMOKE=1 DOC_PRETRAINED=0 python scripts/run_pipeline.py   # uji asap CPU tanpa internet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Konfigurasi, daftar_anggota, seed_everything  # noqa: E402
from src.data import siapkan_konteks  # noqa: E402
from src.ensemble import muat_pool, peluang, pilih_bobot_final, tabel_anggota, tulis_submisi  # noqa: E402
from src.train import latih_semua  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Argumen CLI; nilai yang tidak diberikan diambil dari variabel lingkungan ``DOC_*``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profil", choices=["rvlcdip_ft", "dokumen", "lengkap"])
    p.add_argument("--data-root", type=Path, help="folder berisi train.csv, test.csv, train/, test/")
    p.add_argument("--batas-jam", type=float, help="anggota baru tidak dimulai setelah batas waktu ini")
    p.add_argument("--ft-epochs", type=int, help="epoch fine-tune DiT (0 = checkpoint apa adanya)")
    p.add_argument("--tanpa-donut", action="store_true", help="matikan anggota Donut")
    return p.parse_args()


def main() -> None:
    """Titik masuk CLI."""
    args = parse_args()
    cfg = Konfigurasi.dari_env()
    if args.profil:
        cfg.profil = args.profil
    if args.data_root:
        cfg.data_root = args.data_root
    if args.batas_jam is not None:
        cfg.batas_jam = args.batas_jam
    if args.ft_epochs is not None:
        cfg.ft_epochs = args.ft_epochs
    if args.tanpa_donut:
        cfg.pakai_donut = False
    seed_everything(cfg.seed)
    ctx, _ = siapkan_konteks(cfg)
    anggota = daftar_anggota(cfg)
    print("anggota:", [a["name"] for a in anggota])
    gagal = latih_semua(anggota, ctx)
    pool = muat_pool(cfg.artefak, len(ctx.va_idx), len(ctx.test_df), pakai_tta=cfg.pakai_tta)
    print(tabel_anggota(pool, ctx.y_va).round(4).to_string())
    bobot, info = pilih_bobot_final(pool, ctx.y_va, cfg.seed)
    tulis_submisi(ctx.test_df["id"].values, peluang(pool, bobot, "test"), cfg.submission_path)
    hasil = dict(waktu=time.strftime("%Y-%m-%d %H:%M"), profil=cfg.profil, bobot=bobot, gagal=gagal,
                 val_acc=float(info["acc_final"]), n_val=int(len(ctx.y_va)), menit=round(cfg.jam_berjalan() * 60, 1))
    (cfg.reports_dir / "konfigurasi_final.json").write_text(json.dumps(hasil, indent=2))
    print(json.dumps(hasil, indent=2))
    print("submission:", cfg.submission_path)


if __name__ == "__main__":
    main()
