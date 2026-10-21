"""Zero-shot CLIP logits (text-initialised head, no fine-tuning) for every cached image.

Writes runs/zeroshot_<backbone>/all_logits.npy aligned with the image cache, for
the zero-shot fusion / stacking experiments in stack_experiments.py.

    python zeroshot_logits.py --backbone clip_vit_l14
"""
import argparse

import numpy as np
import torch

from train_lodo import BatchStream, predict_logits
from upar_common import RUNS_DIR, atomic_np_save, build_model, load_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip_vit_l14")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels, _, _, _ = load_cache()
    model = build_model(args.backbone, pretrained=True).to(device).to(memory_format=torch.channels_last)
    stream = BatchStream(images, labels, np.arange(len(labels)), 256, shuffle=False, drop_last=False)
    logits = predict_logits(model, stream, device, tta=False)
    out = RUNS_DIR / f"zeroshot_{args.backbone}"
    out.mkdir(parents=True, exist_ok=True)
    atomic_np_save(out / "all_logits.npy", logits.astype(np.float32))
    print(f"wrote {out / 'all_logits.npy'} {logits.shape}")


if __name__ == "__main__":
    main()
