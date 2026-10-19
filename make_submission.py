"""Package submission/ into a Codabench zip.

    python make_submission.py --run convnext_tiny_v1                 # final model (holdout none)
    python make_submission.py --run convnext_tiny_v1 --models none PETA PA100k Market1501

Copies the chosen checkpoints (as fp16 to halve the size) and the run's
decoder.json into submission/assets/, then zips the folder contents so run.py
sits at the archive root.
"""
import argparse
import shutil
import zipfile
from pathlib import Path

import torch

from upar_common import ROOT, RUNS_DIR

SUB = ROOT / "submission"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--decoder-run", help="run whose decoder.json to ship (default: --run)")
    ap.add_argument("--models", nargs="+", default=["none"],
                    help="fold dirs under runs/<run>/ to ship; logits are averaged")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    assets = SUB / "assets"
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True)
    for i, fold in enumerate(args.models):
        ckpt = torch.load(RUNS_DIR / args.run / fold / "model.pth", map_location="cpu",
                          weights_only=False)
        state = {k: v.half() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()}
        torch.save({"state_dict": state, "backbone": ckpt["backbone"]},
                   assets / f"model_{i}_{fold}.pth")
    decoder = RUNS_DIR / (args.decoder_run or args.run) / "decoder.json"
    shutil.copy(decoder, assets / "decoder.json")

    out = Path(args.out or ROOT / f"submission_{args.run}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(SUB.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts and f.name != ".DS_Store":
                zf.write(f, f.relative_to(SUB))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    for name in zipfile.ZipFile(out).namelist():
        print("  ", name)


if __name__ == "__main__":
    main()
