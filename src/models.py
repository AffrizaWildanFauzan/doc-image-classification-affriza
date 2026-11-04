"""Arsitektur model: backbone timm, DiT/BEiT (transformers), dan Donut, semuanya dari Hugging Face Hub."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from .config import DONUT_TOKEN, RVL_ID2LABEL


class HFWrapper(nn.Module):
    """Pembungkus ``BeitForImageClassification`` (DiT) agar ``forward`` langsung mengembalikan logit.

    Untuk checkpoint yang sudah di-finetune RVL-CDIP (``head_rvl=True``) kepala 16 kelas dipertahankan,
    lalu barisnya diurutkan ulang dari urutan RVL-CDIP ke urutan label lomba.
    """

    def __init__(self, hub: str, classes: dict, head_rvl: bool = False, pretrained: bool = True):
        """``hub`` id model di Hugging Face Hub, ``classes`` peta label -> nama kelas lomba."""
        super().__init__()
        from transformers import BeitConfig, BeitForImageClassification
        self.classes = classes
        if pretrained and head_rvl:
            self.m = BeitForImageClassification.from_pretrained(hub)
        elif pretrained:
            self.m = BeitForImageClassification.from_pretrained(hub, num_labels=len(classes),
                                                                ignore_mismatched_sizes=True)
        else:   # uji asap tanpa internet: arsitektur dit-base, bobot acak
            self.m = BeitForImageClassification(BeitConfig(
                num_labels=len(classes), use_mean_pooling=True, use_absolute_position_embeddings=True,
                layer_scale_init_value=0.1, id2label=dict(enumerate(RVL_ID2LABEL)) if head_rvl else None))
        if head_rvl:
            self.urutkan_kepala()

    def urutkan_kepala(self) -> None:
        """Permutasi baris bobot classifier supaya output ke-c sama dengan kelas lomba ke-c (dicocokkan lewat nama)."""
        rvl = {v.lower().replace(" ", "_"): int(k) for k, v in self.m.config.id2label.items()}
        n = len(self.classes)
        hilang = [self.classes[c] for c in range(n) if self.classes[c] not in rvl]
        assert not hilang, f"kelas lomba tidak ada di checkpoint: {hilang}"
        perm = torch.tensor([rvl[self.classes[c]] for c in range(n)])
        cls = self.m.classifier
        with torch.no_grad():
            cls.weight.copy_(cls.weight[perm].clone())
            cls.bias.copy_(cls.bias[perm].clone())
        print("pemetaan label RVL-CDIP -> lomba:", {self.classes[c]: int(perm[c]) for c in range(n)})

    def set_grad_checkpointing(self, aktif: bool = True) -> None:
        """Aktifkan gradient checkpointing (hemat memori GPU, sekitar 20 sampai 30% lebih lambat)."""
        if aktif:
            self.m.gradient_checkpointing_enable()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Kembalikan logit berukuran ``(batch, 16)``."""
        return self.m(pixel_values=x).logits


class DonutKelas(nn.Module):
    """Donut (encoder Swin + decoder teks) sebagai pengklasifikasi tanpa OCR.

    Satu langkah decoder dengan prompt ``<s_rvlcdip><s_class>``; logit token berikutnya yang dibatasi ke 16 token
    kelas menjadi skor kelas. Setara dengan ``generate()`` tetapi memberi peluang lengkap untuk ensemble.
    """

    def __init__(self, hub: str, size: tuple[int, int], classes: dict, pretrained: bool = True):
        """``size`` adalah (tinggi, lebar) input; resolusi asli checkpoint 2560 x 1920."""
        super().__init__()
        from transformers import DonutImageProcessor, VisionEncoderDecoderModel
        tok = dict(DONUT_TOKEN)
        if pretrained:
            self.m = VisionEncoderDecoderModel.from_pretrained(hub)
            self.proc = DonutImageProcessor.from_pretrained(hub)
            try:
                from transformers import AutoTokenizer
                t = AutoTokenizer.from_pretrained(hub)
                tok = {k: t.convert_tokens_to_ids(k) for k in DONUT_TOKEN}
            except Exception as e:  # noqa: BLE001 - id bawaan dipakai sebagai cadangan
                print("tokenizer Donut tidak dimuat, pakai id token bawaan:", type(e).__name__)
        else:   # uji asap tanpa internet: Donut mungil berbobot acak dengan id token yang sama
            from transformers import DonutSwinConfig, DonutSwinModel, MBartConfig, MBartForCausalLM
            enc = DonutSwinModel(DonutSwinConfig(image_size=list(size), embed_dim=16, depths=[1, 1, 1, 1],
                                                 num_heads=[1, 2, 2, 4], window_size=5))
            dec = MBartForCausalLM(MBartConfig(vocab_size=57544, d_model=32, decoder_layers=1,
                                               decoder_attention_heads=2, decoder_ffn_dim=64,
                                               max_position_embeddings=16, is_decoder=True, add_cross_attention=True))
            self.m = VisionEncoderDecoderModel(encoder=enc, decoder=dec)
            self.proc = DonutImageProcessor(do_align_long_axis=False, do_thumbnail=True, do_pad=True,
                                            image_mean=[0.5] * 3, image_std=[0.5] * 3)
        # halaman diperkecil tanpa mengubah rasio lalu di-pad di tengah sampai (tinggi, lebar)
        self.proc.size = {"height": size[0], "width": size[1]}
        n = len(classes)
        self.register_buffer("prompt", torch.tensor([tok["<s_rvlcdip>"], tok["<s_class>"]]), persistent=False)
        self.register_buffer("kelas", torch.tensor([tok[f"<{classes[c]}/>"] for c in range(n)]), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Kembalikan skor 16 kelas berukuran ``(batch, 16)``."""
        dec = self.prompt.unsqueeze(0).expand(x.size(0), -1)
        return self.m(pixel_values=x, decoder_input_ids=dec).logits[:, -1][:, self.kelas]


def buat_model(anggota: dict, classes: dict, pretrained: bool = True):
    """Bangun model untuk satu anggota ensemble.

    Returns:
        ``(model, mean, std)``; ``mean``/``std`` dipakai untuk normalisasi input.
    """
    if anggota["backend"] == "timm":
        kw = dict(pretrained=pretrained, num_classes=len(classes))
        if not anggota["hub"].startswith("timm/resnet"):
            kw["drop_path_rate"] = 0.1
        if any(k in anggota["hub"] for k in ("swin", "eva", "vit", "deit", "beit")):
            kw["img_size"] = anggota["size"]
        nama = "hf_hub:" + anggota["hub"] if pretrained else anggota["hub"].split("/", 1)[1]
        model = timm.create_model(nama, **kw)
        dc = timm.data.resolve_model_data_config(model)
        return model, dc["mean"], dc["std"]
    if anggota["backend"] == "donut":
        return DonutKelas(anggota["hub"], anggota["size"], classes, pretrained), (0.5,) * 3, (0.5,) * 3
    model = HFWrapper(anggota["hub"], classes, head_rvl=anggota.get("head_rvl", False), pretrained=pretrained)
    return model, (0.5,) * 3, (0.5,) * 3


def grup_parameter(model: nn.Module, anggota: dict):
    """Kelompok parameter untuk AdamW dengan layer-wise LR decay (resep fine-tuning BEiT/DiT).

    Lapisan awal mendapat LR ``lr * layer_decay**k`` yang lebih kecil agar representasi hasil pretraining tidak cepat
    rusak. Tanpa ``layer_decay`` semua parameter memakai LR yang sama.
    """
    ld = anggota.get("layer_decay")
    if not ld:
        return model.parameters()
    n_layer = model.m.config.num_hidden_layers
    grup = {}
    for n, p in model.named_parameters():
        if ".encoder.layer." in n:
            i = int(n.split(".encoder.layer.")[1].split(".")[0]) + 1
        elif ".embeddings." in n:
            i = 0
        else:
            i = n_layer + 1   # pooler dan classifier
        grup.setdefault(ld ** (n_layer + 1 - i), []).append(p)
    return [dict(params=ps, lr=anggota["lr"] * sk) for sk, ps in grup.items()]
