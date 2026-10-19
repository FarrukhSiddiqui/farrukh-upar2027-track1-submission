"""Metric-aware decoder: tune per-attribute thresholds on LODO held-out logits.

The leaderboard ranks by Challenge_Avg = harmonic mean of label mA and instance
F1, computed per test dataset and averaged. A fixed 0.5 cut is far from optimal
for that: mA wants low thresholds on rare attributes, instance F1 wants fewer
false positives. This script fits, on out-of-domain predictions only:

  * one logit threshold per attribute (shared by all domains), and
  * for each one-hot-ish group (age, upper/lower colour, lower type) whether
    to force at least one positive per image (the member with the largest
    logit - threshold margin).

It searches by coordinate ascent on the mean per-domain Challenge_Avg. It
reports an honest estimate by fitting on two held-out domains and scoring the
third (nested LODO), then fits on all three for the shipped decoder.

The decoder ships as probabilities: p = sigmoid(logit - t). An attribute
counts as present when p > 0.5, i.e. logit > t, exactly the tuned rule.

    python fit_decoder.py --run convnext_tiny_v1
"""
import argparse
import json

import numpy as np
import torch

from upar_common import DOMAINS, GROUPS, RUNS_DIR, atomic_write_bytes, load_cache


EPS = 1e-20


def load_oof(run):
    _, labels, domains, names, _ = load_cache()
    logits, idx = [], []
    for d in DOMAINS:
        fold = RUNS_DIR / run / d
        logits.append(np.load(fold / "oof_logits.npy"))
        idx.append(np.load(fold / "oof_index.npy"))
    idx = np.concatenate(idx)
    return np.concatenate(logits), labels[idx], domains[idx], names


class Scorer:
    """Torch re-implementation of upar_common.par_metrics, batched over candidates."""

    def __init__(self, logits, gt, domains, device):
        self.logits = torch.as_tensor(logits, device=device)
        self.gt = torch.as_tensor(gt, device=device).bool()
        self.dom = torch.as_tensor(domains, device=device)
        self.doms = sorted(set(domains.tolist()))
        self.masks = [self.dom == d for d in self.doms]

    def decode(self, thr, force):
        """thr [C,40] -> pred [C,N,40] bool."""
        margin = self.logits.unsqueeze(0) - thr.unsqueeze(1)
        pred = margin > 0
        for g, on in force.items():
            if not on:
                continue
            cols = GROUPS[g]
            sub = margin[..., cols]
            none = ~pred[..., cols].any(-1)
            arg = sub.argmax(-1)
            fix = torch.zeros_like(pred[..., cols])
            fix.scatter_(-1, arg.unsqueeze(-1), True)
            pred[..., cols] |= fix & none.unsqueeze(-1)
        return pred

    def score(self, pred):
        """pred [C,N,40] -> (Challenge_Avg [C], dict of per-metric [C]) averaged over domains."""
        out = {k: [] for k in ("mA", "Instance_F1", "Challenge_Avg")}
        for m in self.masks:
            p, g = pred[:, m], self.gt[m].unsqueeze(0)
            tp = (p & g).sum(1).float()
            tn = (~p & ~g).sum(1).float()
            pos = g.sum(1).float()
            neg = (~g).sum(1).float()
            ma = ((tp / (pos + EPS) + tn / (neg + EPS)) / 2).mean(-1)
            inter = (p & g).sum(2).float()
            prec = (inter / (p.sum(2).float() + EPS)).mean(-1)
            rec = (inter / (g.sum(2).float() + EPS)).mean(-1)
            f1 = 2 * prec * rec / (prec + rec + EPS)
            out["mA"].append(ma)
            out["Instance_F1"].append(f1)
            out["Challenge_Avg"].append(2 * ma * f1 / (ma + f1 + EPS))
        out = {k: torch.stack(v).mean(0) for k, v in out.items()}
        return out["Challenge_Avg"], out

    def evaluate(self, thr, force):
        avg, parts = self.score(self.decode(thr.unsqueeze(0), force))
        return {k: float(v[0]) for k, v in parts.items()}


def fit(scorer, grid, passes, device, chunk=16):
    thr = torch.zeros(40, device=device)
    force = {g: False for g in GROUPS}
    best = scorer.evaluate(thr, force)["Challenge_Avg"]
    for _ in range(passes):
        improved = False
        # group rules: try toggling each
        for g in GROUPS:
            trial = dict(force, **{g: not force[g]})
            s = scorer.evaluate(thr, trial)["Challenge_Avg"]
            if s > best + 1e-6:
                force, best, improved = trial, s, True
        # thresholds: coordinate ascent, candidates evaluated in batches
        for j in range(40):
            cands = torch.as_tensor(grid, device=device, dtype=thr.dtype)
            scores = []
            for c in cands.split(chunk):
                t = thr.unsqueeze(0).repeat(len(c), 1)
                t[:, j] = c
                scores.append(scorer.score(scorer.decode(t, force))[0])
            scores = torch.cat(scores)
            k = int(scores.argmax())
            if float(scores[k]) > best + 1e-6:
                thr[j], best, improved = cands[k], float(scores[k]), True
        if not improved:
            break
    return thr, force, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name under runs/")
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--grid-min", type=float, default=-6.0)
    ap.add_argument("--grid-max", type=float, default=6.0)
    ap.add_argument("--grid-step", type=float, default=0.1)
    ap.add_argument("--shrink", type=float, default=1.0,
                    help="scale the fitted thresholds toward 0 (1 = no shrink)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits, gt, domains, names = load_oof(args.run)
    grid = np.round(np.arange(args.grid_min, args.grid_max + 1e-9, args.grid_step), 4)
    zero = torch.zeros(40, device=device)
    no_force = {g: False for g in GROUPS}

    # Nested LODO: fit on two domains' OOF logits, score the third.
    print("Nested LODO (decoder fitted without the scored domain):")
    nested = []
    for d in sorted(set(domains.tolist())):
        tr, te = domains != d, domains == d
        s_tr = Scorer(logits[tr], gt[tr], domains[tr], device)
        s_te = Scorer(logits[te], gt[te], domains[te], device)
        thr, force, _ = fit(s_tr, grid, args.passes, device)
        base = s_te.evaluate(zero, no_force)
        tuned = s_te.evaluate(thr * args.shrink, force)
        nested.append((base, tuned))
        print(f"  {DOMAINS[d]:<11} @0.5: avg={base['Challenge_Avg']:.4f} "
              f"(mA {base['mA']:.4f}, iF1 {base['Instance_F1']:.4f})  ->  tuned: "
              f"avg={tuned['Challenge_Avg']:.4f} (mA {tuned['mA']:.4f}, "
              f"iF1 {tuned['Instance_F1']:.4f})")
    b = np.mean([x[0]["Challenge_Avg"] for x in nested])
    t = np.mean([x[1]["Challenge_Avg"] for x in nested])
    print(f"  mean        @0.5: {b:.4f}  ->  tuned: {t:.4f}  ({t - b:+.4f})")

    # Final decoder on all three held-out domains.
    scorer = Scorer(logits, gt, domains, device)
    thr, force, best = fit(scorer, grid, args.passes, device)
    thr = thr * args.shrink
    final = scorer.evaluate(thr, force)
    print(f"\nAll-domain fit (optimistic, in-sample): {json.dumps({k: round(v, 4) for k, v in final.items()})}")
    print("force groups:", {g: v for g, v in force.items()})
    print("thresholds (logit):")
    for n, v in zip(names, thr.tolist()):
        print(f"  {n:<32} {v:+.2f}")

    out = {
        "attribute_names": names,
        "logit_thresholds": [round(v, 4) for v in thr.tolist()],
        "force_groups": {g: GROUPS[g] for g, on in force.items() if on},
        "nested_lodo": {"at_0.5": round(b, 4), "tuned": round(t, 4)},
        "run": args.run,
    }
    path = RUNS_DIR / args.run / "decoder.json"
    atomic_write_bytes(path, json.dumps(out, indent=2).encode())
    print(f"saved {path}")


if __name__ == "__main__":
    main()
