"""Run a UPAR Track 1 submission on a folder of images and write predictions.csv.

Uses the submission's own run.py exactly as the Codabench ingestion program
does (load_model(), then predict_batch() in batches of run.BATCH_SIZE), so
the output matches what was scored on the leaderboard.

    python predict.py --submission <unzipped submission dir> \
        --images <root dir of the images> [--list images.csv] --out predictions.csv

--list is a file with one image path per line, relative to --images (the
Codabench input_data/images.csv layout). Without it, every .jpg/.jpeg/.png
under --images is used. The CSV has the gt.csv layout: image path, then the
40 attribute probabilities (present when > 0.5).
"""
import argparse
import csv
import importlib.util
import os
import sys
import time
from pathlib import Path

ATTRIBUTES = [
    "Age-Young", "Age-Adult", "Age-Old", "Gender-Female", "Hair-Length-Short",
    "Hair-Length-Long", "Hair-Length-Bald", "UpperBody-Length-Short", "UpperBody-Color-Black",
    "UpperBody-Color-Blue", "UpperBody-Color-Brown", "UpperBody-Color-Green",
    "UpperBody-Color-Grey", "UpperBody-Color-Orange", "UpperBody-Color-Pink",
    "UpperBody-Color-Purple", "UpperBody-Color-Red", "UpperBody-Color-White",
    "UpperBody-Color-Yellow", "UpperBody-Color-Other", "LowerBody-Length-Short",
    "LowerBody-Color-Black", "LowerBody-Color-Blue", "LowerBody-Color-Brown",
    "LowerBody-Color-Green", "LowerBody-Color-Grey", "LowerBody-Color-Orange",
    "LowerBody-Color-Pink", "LowerBody-Color-Purple", "LowerBody-Color-Red",
    "LowerBody-Color-White", "LowerBody-Color-Yellow", "LowerBody-Color-Other",
    "LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress", "Accessory-Backpack",
    "Accessory-Bag", "Accessory-Glasses-Normal", "Accessory-Glasses-Sun", "Accessory-Hat",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True, help="unzipped submission (run.py + assets/)")
    ap.add_argument("--images", required=True, help="root folder of the images")
    ap.add_argument("--list", help="optional file with one image path per line, relative to --images")
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    root = Path(args.images).resolve()
    if args.list:
        rel = [l.strip() for l in open(args.list) if l.strip()]
    else:
        rel = sorted(str(p.relative_to(root)) for p in root.rglob("*")
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    samples = [{"index": i, "image": r, "image_path": str(root / r), "attribute_names": ATTRIBUTES}
               for i, r in enumerate(rel)]
    out = Path(args.out).resolve()

    sub = Path(args.submission).resolve()
    os.chdir(sub)  # the ingestion program runs inside the submission folder
    sys.path.insert(0, str(sub))
    spec = importlib.util.spec_from_file_location("run", sub / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)

    t0 = time.time()
    run.load_model()
    rows = []
    bs = getattr(run, "BATCH_SIZE", 32)
    for i in range(0, len(samples), bs):
        rows += run.predict_batch(samples[i:i + bs])
        if (i // bs) % 50 == 0:
            print(f"{min(i + bs, len(samples))}/{len(samples)} images, {time.time() - t0:.1f}s", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# image"] + ATTRIBUTES)
        for s, r in zip(samples, rows):
            w.writerow([s["image"]] + [f"{float(v):.6f}" for v in r])
    print(f"wrote {out}: {len(rows)} images in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
