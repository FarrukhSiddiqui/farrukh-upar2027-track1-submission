"""Shared pieces for UPAR 2027 Track 1: attributes, data cache, model, metrics.

Imported by cache_images.py, train_lodo.py and fit_decoder.py. The submission's
run.py carries its own copy of the model builder so it has no dependency on
this file.
"""
import csv
import io
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
CACHE_DIR = ROOT / "cache"
RUNS_DIR = ROOT / "runs"

IMAGE_SIZE = (256, 128)  # (H, W)
DOMAINS = ("Market1501", "PA100k", "PETA")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Groups whose members are (almost) always exactly one-hot in the annotations.
# The decoder may force at least one positive per group.
GROUPS = {
    "age": [0, 1, 2],
    "upper_color": list(range(8, 20)),
    "lower_color": list(range(21, 33)),
    "lower_type": [33, 34],
}


def read_gt(split):
    """Return (image paths relative to data/, labels [N,40] uint8, names)."""
    csv_path = DATA_ROOT / "annotations" / "task1" / split / "gt.csv"
    paths, labels = [], []
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        names = [h.strip() for h in header[1:]]
        for row in reader:
            if len(row) != 41:
                continue
            paths.append(row[0])
            labels.append([int(v) for v in row[1:]])
    return paths, np.asarray(labels, dtype=np.uint8), names


def domain_of(path):
    return path.split("/", 1)[0]


# --------------------------------------------------------------------------- atomic writes
# A power cut mid-write must never leave a half-written file under the real
# name: write to <name>.tmp, fsync, rename over the target, fsync the folder.

def atomic_write_bytes(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def atomic_torch_save(obj, path):
    buf = io.BytesIO()
    torch.save(obj, buf)
    atomic_write_bytes(path, buf.getvalue())


def atomic_np_save(path, arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    atomic_write_bytes(path, buf.getvalue())


def load_cache():
    """Images [N,H,W,3] uint8 memmap, labels [N,40], domain ids [N], names."""
    meta = np.load(CACHE_DIR / "meta.npz", allow_pickle=False)
    n = len(meta["labels"])
    images = np.memmap(CACHE_DIR / "images_u8.bin", dtype=np.uint8, mode="r",
                       shape=(n, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    return images, meta["labels"], meta["domains"], list(meta["names"]), list(meta["paths"])


# --------------------------------------------------------------------------- model

BACKBONES = {
    # name: (constructor, weights, feature dim)
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 768),
    "convnext_small": (models.convnext_small, models.ConvNeXt_Small_Weights.IMAGENET1K_V1, 768),
    "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights.IMAGENET1K_V1, 1024),
}


class AttributeNet(nn.Module):
    """Backbone feature map -> global avg+max pooling -> BN -> linear(40)."""

    def __init__(self, backbone="convnext_tiny", pretrained=True, num_attrs=40, drop=0.2):
        super().__init__()
        ctor, weights, dim = BACKBONES[backbone]
        net = ctor(weights=weights if pretrained else None)
        if backbone.startswith("resnet"):
            self.features = nn.Sequential(*list(net.children())[:-2])
        else:
            self.features = net.features
            # ConvNeXt applies LayerNorm in the classifier; keep it on the pooled vector.
        self.backbone = backbone
        self.norm = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(dim, num_attrs)

    def forward(self, x):
        f = self.features(x)
        f = f.mean((2, 3)) + f.amax((2, 3))
        return self.head(self.drop(self.norm(f)))


# --------------------------------------------------------------------------- metrics

EPS = 1e-20  # same epsilon as the reference UPAR / Rethinking-PAR evaluation


def par_metrics(gt, pred):
    """Reference UPAR metrics on binary arrays [N,40]. Returns a dict."""
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    gt_pos = gt.sum(0)
    gt_neg = (~gt).sum(0)
    tp = (gt & pred).sum(0)
    tn = (~gt & ~pred).sum(0)
    fp = (~gt & pred).sum(0)
    fn = (gt & ~pred).sum(0)
    pos_rec = tp / (gt_pos + EPS)
    neg_rec = tn / (gt_neg + EPS)
    label_ma = (pos_rec + neg_rec) / 2
    label_prec = tp / (tp + fp + EPS)
    label_f1 = 2 * label_prec * pos_rec / (label_prec + pos_rec + EPS)

    inter = (gt & pred).sum(1)
    union = (gt | pred).sum(1)
    inst_acc = (inter / (union + EPS)).mean()
    inst_prec = (inter / (pred.sum(1) + EPS)).mean()
    inst_rec = (inter / (gt.sum(1) + EPS)).mean()
    inst_f1 = 2 * inst_prec * inst_rec / (inst_prec + inst_rec + EPS)

    ma = label_ma.mean()
    return {
        "mA": float(ma),
        "Label_F1": float(label_f1.mean()),
        "Instance_Acc": float(inst_acc),
        "Instance_Prec": float(inst_prec),
        "Instance_Rec": float(inst_rec),
        "Instance_F1": float(inst_f1),
        "Challenge_Avg": float(2 * ma * inst_f1 / (ma + inst_f1 + EPS)),
        "label_mA": label_ma,
    }


def per_domain_metrics(gt, pred, domains):
    """Leaderboard view: each domain scored alone, then averaged."""
    per = {}
    for d in np.unique(domains):
        m = domains == d
        per[int(d)] = par_metrics(gt[m], pred[m])
    keys = [k for k in next(iter(per.values())) if k != "label_mA"]
    mean = {k: float(np.mean([p[k] for p in per.values()])) for k in keys}
    return mean, per
