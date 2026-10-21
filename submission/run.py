"""UPAR 2027 Track 1 submission: ConvNeXt attribute model + metric-aware decoder.

The ingestion program imports this file and calls load_model() once, then
predict_batch(samples) per batch. Everything needed ships in assets/:

    assets/model_*.npz     one or more checkpoints (fp16 numpy); logits are averaged
    assets/decoder.json    per-attribute logit thresholds + group rules

decoder.json picks one of two decoders, both tuned on leave-one-domain-out
predictions (see fit_decoder.py):
  threshold: p = sigmoid(logit - t_a), so "present if p > 0.5" is logit > t_a.
  set:       per image, choose the attribute set maximising an estimate of
             mA + instance precision/recall; chosen -> 0.5 + 0.49 q, else 0.49 q.
Members of a one-hot group may be lifted above 0.5 when the group would
otherwise be empty.

Runs in pytorch/pytorch:2.4.1 (torch, torchvision, numpy, Pillow only).
"""
from __future__ import annotations

import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # clip_vit.py ships next to this file
ASSETS = os.path.join(HERE, "assets")
IMAGE_SIZE = (256, 128)  # (H, W), same as training
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 128

BACKBONES = {
    "resnet50": (models.resnet50, 2048),
    "convnext_tiny": (models.convnext_tiny, 768),
    "convnext_small": (models.convnext_small, 768),
    "convnext_base": (models.convnext_base, 1024),
}


class AttributeNet(nn.Module):
    """Must match upar_common.AttributeNet so state dicts load unchanged."""

    def __init__(self, backbone, num_attrs=40):
        super().__init__()
        ctor, dim = BACKBONES[backbone]
        net = ctor(weights=None)
        if backbone.startswith("resnet"):
            self.features = nn.Sequential(*list(net.children())[:-2])
        else:
            self.features = net.features
        self.norm = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(0.0)
        self.head = nn.Linear(dim, num_attrs)

    def forward(self, x):
        f = self.features(x)
        f = f.mean((2, 3)) + f.amax((2, 3))
        return self.head(self.drop(self.norm(f)))


_MODELS: list[nn.Module] = []
_DEVICE = torch.device("cpu")
_THR: np.ndarray | None = None
_GROUPS: list[list[int]] = []
_DEC: dict = {}
_NAMES: list[str] = []
_POOL: ThreadPoolExecutor | None = None
_MEAN_T = _STD_T = None


def load_model() -> None:
    global _DEVICE, _THR, _GROUPS, _NAMES, _POOL, _MEAN_T, _STD_T, _DEC
    if _MODELS:
        return
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    paths = sorted(glob.glob(os.path.join(ASSETS, "model_*.npz"))
                   + glob.glob(os.path.join(ASSETS, "model_*.pth")))
    for path in paths:
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=False) as npz:
                backbone = str(npz["__backbone__"])
                raw = {k: torch.from_numpy(npz[k].copy()) for k in npz.files if k != "__backbone__"}
        else:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            backbone, raw = ckpt["backbone"], ckpt["state_dict"]
        if backbone.startswith("siglip"):
            from siglip_vit import SigLIPAttributeNet
            model = SigLIPAttributeNet(backbone, IMAGE_SIZE)
        elif backbone.startswith("clip_"):
            from clip_vit import CLIPAttributeNet
            model = CLIPAttributeNet(backbone, IMAGE_SIZE)
        else:
            model = AttributeNet(backbone)
        state = {k: v.float() if v.is_floating_point() else v for k, v in raw.items()}
        model.load_state_dict(state)
        model.eval().to(_DEVICE).to(memory_format=torch.channels_last)
        _MODELS.append(model)
    if not _MODELS:
        raise RuntimeError(f"no assets/model_*.npz or model_*.pth found in {ASSETS}")
    with open(os.path.join(ASSETS, "decoder.json"), "r", encoding="utf-8") as fh:
        dec = json.load(fh)
    _DEC = dec
    _NAMES = dec["attribute_names"]
    _THR = np.asarray(dec["logit_thresholds"], dtype=np.float32)
    _GROUPS = list(dec.get("force_groups", {}).values())
    _POOL = ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4))
    _MEAN_T = torch.tensor(MEAN, device=_DEVICE).view(1, 3, 1, 1)
    _STD_T = torch.tensor(STD, device=_DEVICE).view(1, 3, 1, 1)
    # warm-up so cudnn autotuning is not billed to the first real batch
    with torch.inference_mode():
        _forward(torch.zeros(2, *IMAGE_SIZE, 3, dtype=torch.uint8))


def _load_image(path: str) -> np.ndarray:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
            return np.asarray(im, dtype=np.uint8)
    except Exception:  # unreadable image: neutral grey, decoder still returns 40 values
        return np.full((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), 128, dtype=np.uint8)


@torch.inference_mode()
def _forward(u8: torch.Tensor) -> np.ndarray:
    x = u8.to(_DEVICE, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    x = ((x - _MEAN_T) / _STD_T).contiguous(memory_format=torch.channels_last)
    use_amp = _DEVICE.type == "cuda"
    total = None
    tta = _DEC.get("tta", True)  # flip TTA; decoder.json says whether it was fitted with it
    with torch.autocast(device_type=_DEVICE.type, dtype=torch.float16, enabled=use_amp):
        for model in _MODELS:
            logit = model(x).float()
            if tta:
                logit = (logit + model(x.flip(3)).float()) / 2
            total = logit if total is None else total + logit
    return (total / len(_MODELS)).cpu().numpy()


def _decode_set(logits: np.ndarray) -> np.ndarray:
    """Per-image set decoder; mirrors fit_decoder.decode_set."""
    prm = _DEC["set_params"]
    T, w, rho = prm["T"], prm["w"], prm["rho"]
    max_k, qmin = prm.get("max_k", 20), prm.get("qmin", 0.0)
    q = 1.0 / (1.0 + np.exp(-np.clip(logits / T, -30, 30)))
    pi = np.asarray(_DEC["priors"], dtype=np.float64)[None]
    g = q / pi - (1 - q) / (1 - pi)
    sq = np.maximum(q.sum(1, keepdims=True), 1e-6)
    best_j = np.full(len(q), -1e30)
    chosen = np.zeros(q.shape, dtype=bool)
    for k in range(1, max_k + 1):
        score = w * g / 80 + (1 - w) * (rho / k + (1 - rho) / sq) * q
        if qmin:
            score = np.where(q >= qmin, score, -1e6)
        order = np.argsort(-score, axis=1, kind="stable")[:, :k]
        j = np.take_along_axis(score, order, 1).sum(1)
        better = j > best_j
        best_j = np.where(better, j, best_j)
        if better.any():
            sel = np.zeros((int(better.sum()), q.shape[1]), dtype=bool)
            np.put_along_axis(sel, order[better], True, 1)
            chosen[better] = sel
    for cols in _GROUPS:
        empty = ~chosen[:, cols].any(1)
        if empty.any():
            rows = np.where(empty)[0]
            chosen[rows, np.asarray(cols)[q[rows][:, cols].argmax(1)]] = True
    return np.where(chosen, 0.51 + 0.48 * q, 0.49 * q)


def _decode(logits: np.ndarray) -> np.ndarray:
    prob = _decode_set(logits.astype(np.float64)) if _DEC.get("mode") == "set" \
        else _decode_threshold(logits)
    # Keep every value at least 0.01 from the 0.5 cut, so any rounding when
    # predictions.csv is written can never flip a decision.
    present = prob > 0.5
    return np.where(present, np.maximum(prob, 0.51), np.minimum(prob, 0.49)).clip(0.0, 1.0)


def _decode_threshold(logits: np.ndarray) -> np.ndarray:
    margin = logits - _THR
    prob = 1.0 / (1.0 + np.exp(-np.clip(margin, -30, 30)))
    for cols in _GROUPS:
        sub = margin[:, cols]
        empty = (sub <= 0).all(1)
        if empty.any():
            rows = np.where(empty)[0]
            best = np.asarray(cols)[sub[rows].argmax(1)]
            prob[rows, best] = np.maximum(prob[rows, best], 0.51)
    return np.clip(prob, 0.0, 1.0)


def _reorder(prob: np.ndarray, names: list[str]) -> np.ndarray:
    if names == _NAMES:
        return prob
    pos = {n: i for i, n in enumerate(_NAMES)}
    return np.stack([prob[:, pos[n]] if n in pos else np.zeros(len(prob)) for n in names], 1)


def predict_batch(samples: list[dict[str, Any]]) -> list[list[float]]:
    if not _MODELS:
        load_model()
    arrays = list(_POOL.map(_load_image, [s["image_path"] for s in samples]))
    u8 = torch.from_numpy(np.stack(arrays))
    if _DEVICE.type == "cuda":
        u8 = u8.pin_memory()
    with torch.inference_mode():
        logits = _forward(u8)
    prob = _reorder(_decode(logits), samples[0]["attribute_names"])
    return prob.astype(np.float64).tolist()


def predict_image(sample: dict[str, Any]) -> list[float]:
    return predict_batch([sample])[0]
