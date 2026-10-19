"""Run submission/run.py the way the ingestion program does and score it.

    python test_submission.py                       # all val images, every domain
    python test_submission.py --domain PETA --limit 2000

Images are read from the original JPEGs (not the training cache), through
run.py's own loader, in batches of run.BATCH_SIZE. Prints the leaderboard
metrics and the throughput. Scores of a model trained on these images are
in-sample; use a LODO fold checkpoint for an honest number.
"""
import argparse
import importlib.util
import time

import numpy as np

from upar_common import DATA_ROOT, DOMAINS, ROOT, domain_of, per_domain_metrics, read_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--domain", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    paths, labels, names = read_gt(args.split)
    keep = np.array([args.domain is None or domain_of(p) == args.domain for p in paths])
    paths = [p for p, k in zip(paths, keep) if k]
    labels = labels[keep]
    if args.limit:
        sel = np.random.default_rng(0).choice(len(paths), min(args.limit, len(paths)), replace=False)
        paths, labels = [paths[i] for i in sel], labels[sel]
    domains = np.array([DOMAINS.index(domain_of(p)) for p in paths])

    spec = importlib.util.spec_from_file_location("run", ROOT / "submission" / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)

    t0 = time.time()
    run.load_model()
    t_load = time.time() - t0
    samples = [{"index": i, "image": p, "image_path": str(DATA_ROOT / p), "attribute_names": names}
               for i, p in enumerate(paths)]
    bs = getattr(run, "BATCH_SIZE", 32)
    t0 = time.time()
    preds = []
    for i in range(0, len(samples), bs):
        preds += run.predict_batch(samples[i:i + bs])
    t_pred = time.time() - t0
    prob = np.asarray(preds)
    assert prob.shape == (len(samples), 40), prob.shape
    assert np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1

    mean, per = per_domain_metrics(labels, prob > 0.5, domains)
    for d, m in per.items():
        print(f"{DOMAINS[d]:<11} " + "  ".join(f"{k}={v:.4f}" for k, v in m.items() if k != "label_mA"))
    print("MEAN        " + "  ".join(f"{k}={v:.4f}" for k, v in mean.items()))
    print(f"load {t_load:.1f}s, predict {len(samples)} images in {t_pred:.1f}s "
          f"({len(samples) / t_pred:.0f} img/s)")


if __name__ == "__main__":
    main()
