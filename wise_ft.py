"""WiSE-FT: interpolate fine-tuned CLIP weights back toward the zero-shot model.

    theta(alpha) = alpha * theta_finetuned + (1 - alpha) * theta_zeroshot

Both ends share the text-initialised head, so the whole network interpolates.
For every alpha this writes a complete run directory, runs/<run>_wise<alpha>/,
with re-predicted held-out logits per LODO fold and the merged final model, so
fit_decoder.py and make_submission.py work on it unchanged:

    python wise_ft.py --run clip_b16_v1 --alphas 0.5 0.6 0.7 0.8 0.9
    python fit_decoder.py --run clip_b16_v1_wise0.7
"""
import argparse
import json

import numpy as np
import torch

from train_lodo import BatchStream, predict_logits
from upar_common import (DOMAINS, RUNS_DIR, atomic_np_save, atomic_torch_save,
                         atomic_write_bytes, build_model, load_cache)


def merge(ft, zs, alpha):
    return {k: (alpha * v.float() + (1 - alpha) * zs[k].float()).to(v.dtype)
            if v.is_floating_point() else v for k, v in ft.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels, _, _, _ = load_cache()
    src = RUNS_DIR / args.run
    backbone = torch.load(src / "none" / "model.pth", map_location="cpu",
                          weights_only=False)["backbone"]
    zs = {k: v.cpu() for k, v in build_model(backbone, pretrained=True).state_dict().items()}
    model = build_model(backbone, pretrained=False).to(device).to(memory_format=torch.channels_last)

    for alpha in args.alphas:
        dst = RUNS_DIR / f"{args.run}_wise{alpha:g}"
        for fold in list(DOMAINS) + ["none"]:
            ck = torch.load(src / fold / "model.pth", map_location="cpu", weights_only=False)
            merged = merge(ck["state_dict"], zs, alpha)
            out = dst / fold
            out.mkdir(parents=True, exist_ok=True)
            atomic_torch_save(dict(ck, state_dict=merged, wise_alpha=alpha), out / "model.pth")
            if fold != "none":
                idx = np.load(src / fold / "oof_index.npy")
                model.load_state_dict(merged)
                stream = BatchStream(images, labels, idx, 256, shuffle=False, drop_last=False)
                logits = predict_logits(model, stream, device, tta=True)
                atomic_np_save(out / "oof_logits.npy", logits.astype(np.float32))
                atomic_np_save(out / "oof_index.npy", idx)
            atomic_write_bytes(out / "done.json", json.dumps({"wise_alpha": alpha}).encode())
        print(f"alpha={alpha:g}: wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
