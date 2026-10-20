"""Emulate the Codabench ingestion program against an unzipped submission.

Builds an input_data/ folder in the documented layout (images.csv, images/,
task.json) from validation images, then, with the submission folder as the
working directory, imports run.py, calls load_model() and predict_batch()
(or predict_image()) and checks every output row against the Submission
page's rules: 40 finite values in [0, 1] per image, in attribute order.

    python emulate_ingestion.py --submission <unzipped dir> --work <scratch dir> \\
        [--per-domain 400] [--out probs.npy]

Stdlib + numpy + Pillow only, so it runs in a clean torch 2.4.1 env.
"""
import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DOMAINS = ("Market1501", "PA100k", "PETA")


def build_input(work, per_domain):
    inp = work / "input_data"
    if inp.exists():
        shutil.rmtree(inp)
    (inp / "images").mkdir(parents=True)
    with open(ROOT / "data/annotations/task1/val/gt.csv", newline="") as f:
        rows = list(csv.reader(f))
    names = [h.strip() for h in rows[0][1:]]
    rows = rows[1:]
    rng = np.random.default_rng(0)
    chosen = []
    for d in DOMAINS:
        dom = [r for r in rows if r[0].startswith(d + "/")]
        chosen += [dom[i] for i in rng.choice(len(dom), min(per_domain, len(dom)), replace=False)]
    rel_paths, labels = [], []
    for i, r in enumerate(chosen):
        rel = f"{r[0].split('/')[0]}/images/{i}.jpg"
        (inp / "images" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "data" / r[0], inp / "images" / rel)
        rel_paths.append(rel)
        labels.append([int(v) for v in r[1:]])
    # awkward inputs: grayscale, tiny crop, PNG bytes under a .jpg name
    src = Image.open(ROOT / "data" / chosen[0][0])
    extras = [("Extra/images/gray.jpg", src.convert("L"), "JPEG"),
              ("Extra/images/tiny.jpg", src.resize((6, 13)), "JPEG"),
              ("Extra/images/png_named.jpg", src, "PNG")]
    for rel, im, fmt in extras:
        (inp / "images" / rel).parent.mkdir(parents=True, exist_ok=True)
        im.save(inp / "images" / rel, fmt)
        rel_paths.append(rel)
    (inp / "images.csv").write_text("\n".join(rel_paths) + "\n")
    (inp / "task.json").write_text(json.dumps({"attribute_names": names,
                                               "num_images": len(rel_paths)}))
    return inp, names, rel_paths, np.asarray(labels)


def check_row(row, names, where):
    if isinstance(row, dict):
        assert set(row) == set(names), f"{where}: dict keys != attribute names"
        row = [row[n] for n in names]
    vals = [float(v) for v in (row.tolist() if hasattr(row, "tolist") else row)]
    assert len(vals) == 40, f"{where}: {len(vals)} values, expected 40"
    for v in vals:
        assert math.isfinite(v), f"{where}: non-finite value {v}"
        assert 0.0 <= v <= 1.0, f"{where}: value {v} outside [0, 1]"
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--per-domain", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sub = Path(args.submission).resolve()
    inp, names, rel_paths, labels = build_input(Path(args.work).resolve(), args.per_domain)
    samples = [{"index": i, "image": rel, "image_path": str(inp / "images" / rel),
                "attribute_names": names} for i, rel in enumerate(rel_paths)]

    os.chdir(sub)  # "The working directory is your submission folder"
    sys.path.insert(0, str(sub))
    spec = importlib.util.spec_from_file_location("run", sub / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)

    import torch
    print(f"python {sys.version.split()[0]}, torch {torch.__version__}, "
          f"cuda available: {torch.cuda.is_available()}")
    t0 = time.time()
    if hasattr(run, "load_model"):
        run.load_model()
    t_load = time.time() - t0

    rows = []
    t0 = time.time()
    if hasattr(run, "predict_batch"):
        bs = getattr(run, "BATCH_SIZE", 32)
        for i in range(0, len(samples), bs):
            out = run.predict_batch(samples[i:i + bs])
            assert len(out) == len(samples[i:i + bs]), "predict_batch returned wrong row count"
            rows += [check_row(r, names, f"image {i + j}") for j, r in enumerate(out)]
    else:
        rows = [check_row(run.predict_image(s), names, f"image {s['index']}") for s in samples]
    t_pred = time.time() - t0
    probs = np.asarray(rows)
    print(f"OK: {len(rows)} rows, all 40 values finite and in [0, 1] "
          f"(includes grayscale, 6x13 px and PNG-as-.jpg images)")
    print(f"load_model {t_load:.1f}s, predict {len(rows)} images in {t_pred:.1f}s "
          f"({len(rows) / t_pred:.0f} img/s on this device)")

    n = len(labels)
    pred, gt = probs[:n] > 0.5, labels.astype(bool)
    print(f"positives per image: predicted {pred.sum(1).mean():.2f}, ground truth {gt.sum(1).mean():.2f}")
    if args.out:
        np.save(args.out, probs)


if __name__ == "__main__":
    main()
