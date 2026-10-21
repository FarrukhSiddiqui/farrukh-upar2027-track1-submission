"""Uniform model soup: average the weights of runs fine-tuned from the same init.

For every LODO fold and the final model, writes runs/<name>/<fold>/model.pth as
the parameter-wise mean of the same fold across --runs, plus the fold's held-out
index, so wise_ft.py (which re-predicts held-out logits) and make_submission.py
work on the soup unchanged:

    python soup.py --runs clip_l14_v1 clip_l14_v2 --name clip_l14_soup
    python wise_ft.py --run clip_l14_soup --alphas 1.0 0.9 --no-tta --suffix _nt
"""
import argparse
import json
import shutil

import torch

from upar_common import DOMAINS, RUNS_DIR, atomic_torch_save, atomic_write_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    for fold in list(DOMAINS) + ["none"]:
        cks = [torch.load(RUNS_DIR / r / fold / "model.pth", map_location="cpu", weights_only=False)
               for r in args.runs]
        assert len({c["backbone"] for c in cks}) == 1, "soup needs one backbone"
        sd0 = cks[0]["state_dict"]
        avg = {k: (sum(c["state_dict"][k].float() for c in cks) / len(cks)).to(v.dtype)
               if v.is_floating_point() else v for k, v in sd0.items()}
        out = RUNS_DIR / args.name / fold
        out.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(dict(cks[0], state_dict=avg, soup_of=args.runs), out / "model.pth")
        if fold != "none":
            shutil.copy(RUNS_DIR / args.runs[0] / fold / "oof_index.npy", out / "oof_index.npy")
        atomic_write_bytes(out / "done.json", json.dumps({"soup_of": args.runs}).encode())
        print(f"{fold}: averaged {len(cks)} models", flush=True)


if __name__ == "__main__":
    main()
