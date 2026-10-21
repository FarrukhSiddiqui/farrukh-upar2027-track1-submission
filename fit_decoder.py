"""Metric-aware decoder: tune per-attribute thresholds on LODO held-out logits.

The leaderboard ranks by Challenge_Avg = harmonic mean of label mA and instance
F1, computed per test dataset and averaged. A fixed 0.5 cut is far from optimal
for that: mA wants low thresholds on rare attributes, instance F1 wants fewer
false positives. This script fits, on out-of-domain predictions only:

  * one logit threshold per attribute (shared by all domains), and
  * for each one-hot-ish group (age, upper/lower colour, lower type) whether
    to force at least one positive per image (the member with the largest
    logit - threshold margin).

It searches by coordinate ascent on the mean per-domain Challenge_Avg.

A second, per-image decoder ("set" mode) picks each image's attribute set S
by maximising a first-order estimate of the metric itself:

    J(S) = w * sum_{a in S} g_a / 80  +  (1 - w) * [rho * sum_S q / |S| + (1 - rho) * sum_S q / sum_all q]
    q = sigmoid(logit / T),   g_a = q_a / pi_a - (1 - q_a) / (1 - pi_a)

The first term is the expected mA gain (balanced accuracy rewards a positive
when q_a > prior pi_a), the second the expected instance precision and
recall. For each k the best S of size k is the top-k of the per-attribute
score, so it is exact over k = 1..K. w, rho and T are grid-searched.

Both modes report an honest estimate by fitting on two held-out domains and
scoring the third (nested LODO). The mode with the better nested score is
shipped, fitted on all three domains.

Thresholds ship as p = sigmoid(logit - t); set mode ships 0.51 + 0.48 q for
chosen attributes and 0.49 q otherwise (run.py keeps every value >= 0.01 away
from 0.5, so rounding in predictions.csv cannot flip a decision). Either way "present iff p > 0.5"
reproduces the tuned decision.

    python fit_decoder.py --run convnext_tiny_v1
"""
import argparse
import json

import numpy as np
import torch

from upar_common import DOMAINS, GROUPS, RUNS_DIR, atomic_write_bytes, load_cache


EPS = 1e-20


def load_oof(run, folds=DOMAINS):
    _, labels, domains, names, _ = load_cache()
    logits, idx = [], []
    for d in folds:
        fold = RUNS_DIR / run / d
        logits.append(np.load(fold / "oof_logits.npy"))
        idx.append(np.load(fold / "oof_index.npy"))
    idx = np.concatenate(idx)
    return np.concatenate(logits), labels[idx], domains[idx], names, labels, domains


def domain_priors(labels, domains, use):
    """Positive rate per attribute with every domain in `use` weighted equally."""
    return np.mean([labels[domains == d].mean(0) for d in use], 0).clip(1e-4, 1 - 1e-4)


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


def decode_set(logits, priors, T, w, rho, max_k=20, force=None, qmin=0.0):
    """Per-image set decoder (see module docstring). logits [N,40] -> bool [N,40].

    qmin: an attribute is only eligible if q >= qmin (the top one always is).
    """
    q = torch.sigmoid(logits / T)
    pi = priors.view(1, -1)
    g = q / pi - (1 - q) / (1 - pi)
    sq = q.sum(1, keepdim=True).clamp_min(1e-6)
    best_j = torch.full((len(q),), -1e30, device=q.device)
    best_k = torch.ones(len(q), dtype=torch.long, device=q.device)
    order_by_k = []
    for k in range(1, max_k + 1):
        score = w * g / 80 + (1 - w) * (rho / k + (1 - rho) / sq) * q
        if qmin:
            score = torch.where(q >= qmin, score, torch.full_like(score, -1e6))
        top, order = score.sort(1, descending=True)
        j = top[:, :k].sum(1)
        better = j > best_j
        best_j = torch.where(better, j, best_j)
        best_k = torch.where(better, torch.full_like(best_k, k), best_k)
        order_by_k.append(order[:, :k].to(torch.int8))  # small: 40 attrs
    pred = torch.zeros_like(q, dtype=torch.bool)
    for k in range(1, max_k + 1):
        rows = best_k == k
        if rows.any():
            cols = order_by_k[k - 1][rows].long()
            sub = torch.zeros(int(rows.sum()), q.shape[1], dtype=torch.bool, device=q.device)
            sub.scatter_(1, cols, True)
            pred[rows] = sub
    for g_name, on in (force or {}).items():
        if on:
            cols = GROUPS[g_name]
            none = ~pred[:, cols].any(1)
            arg = torch.as_tensor(cols, device=q.device)[q[:, cols].argmax(1)]
            pred[none.nonzero().squeeze(1), arg[none]] = True
    return pred


SET_GRID = {"T": [0.4, 0.55, 0.7, 0.85, 1.0, 1.2], "w": [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8],
            "rho": [0.3, 0.45, 0.6, 0.75]}


def fit_set(scorer, priors):
    best, params = -1.0, None
    for T in SET_GRID["T"]:
        for w in SET_GRID["w"]:
            for rho in SET_GRID["rho"]:
                pred = decode_set(scorer.logits, priors, T, w, rho)
                s = float(scorer.score(pred.unsqueeze(0))[0][0])
                if s > best:
                    best, params = s, {"T": T, "w": w, "rho": rho}
    force = {g: False for g in GROUPS}
    for g in GROUPS:  # then group rules, kept only if they help
        trial = dict(force, **{g: True})
        pred = decode_set(scorer.logits, priors, force=trial, **params)
        s = float(scorer.score(pred.unsqueeze(0))[0][0])
        if s > best + 1e-6:
            force, best = trial, s
    return params, force, best


def eval_set(scorer, priors, params, force):
    pred = decode_set(scorer.logits, priors, force=force, **params)
    return {k: float(v[0]) for k, v in scorer.score(pred.unsqueeze(0))[1].items()}


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
    ap.add_argument("--mode", choices=["auto", "threshold", "set"], default="auto",
                    help="auto = ship whichever has the better nested-LODO score")
    ap.add_argument("--folds", nargs="+", default=list(DOMAINS),
                    help="held-out folds to use; with one fold, nested LODO becomes a "
                         "split-half check (fit on half the images, score the other half)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits, gt, domains, names, all_labels, all_domains = load_oof(args.run, args.folds)
    fold_ids = [DOMAINS.index(f) for f in args.folds]
    grid = np.round(np.arange(args.grid_min, args.grid_max + 1e-9, args.grid_step), 4)
    zero = torch.zeros(40, device=device)
    no_force = {g: False for g in GROUPS}

    def fmt(m):
        return f"avg={m['Challenge_Avg']:.4f} (mA {m['mA']:.4f}, iF1 {m['Instance_F1']:.4f})"

    # Nested LODO: fit on two domains' OOF logits, score the third. With a
    # single fold available, fit on one random half of it and score the other.
    if len(fold_ids) > 1:
        print("Nested LODO (decoder fitted without the scored domain):")
        splits = [(DOMAINS[d], domains != d, domains == d, [x for x in fold_ids if x != d])
                  for d in fold_ids]
    else:
        print("Split-half check on one fold (optimistic: same domain on both sides):")
        half = np.random.default_rng(0).random(len(domains)) < 0.5
        splits = [("half A->B", half, ~half, fold_ids), ("half B->A", ~half, half, fold_ids)]
    nested = []
    for name, tr, te, prior_doms in splits:
        s_tr = Scorer(logits[tr], gt[tr], domains[tr], device)
        s_te = Scorer(logits[te], gt[te], domains[te], device)
        priors = torch.as_tensor(domain_priors(all_labels, all_domains, prior_doms),
                                 device=device, dtype=torch.float32)
        thr, force, _ = fit(s_tr, grid, args.passes, device)
        params, sforce, _ = fit_set(s_tr, priors)
        base = s_te.evaluate(zero, no_force)
        tuned = s_te.evaluate(thr * args.shrink, force)
        setm = eval_set(s_te, priors, params, sforce)
        nested.append((base, tuned, setm))
        print(f"  {name:<11} @0.5: {fmt(base)}\n"
              f"  {'':<11} thresholds: {fmt(tuned)}\n"
              f"  {'':<11} set decoder: {fmt(setm)}  {params}")
    b, t, st = (float(np.mean([x[i]["Challenge_Avg"] for x in nested])) for i in range(3))
    print(f"  mean @0.5 {b:.4f} | thresholds {t:.4f} ({t - b:+.4f}) | "
          f"set decoder {st:.4f} ({st - b:+.4f})")
    mode = args.mode if args.mode != "auto" else ("set" if st > t else "threshold")
    print(f"shipping mode: {mode}")

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
        "mode": "threshold",
        "attribute_names": names,
        "logit_thresholds": [round(v, 4) for v in thr.tolist()],
        "force_groups": {g: GROUPS[g] for g, on in force.items() if on},
        "nested_lodo": {"at_0.5": round(b, 4), "threshold": round(t, 4), "set": round(st, 4)},
        "folds": args.folds,
        "run": args.run,
        # whether the held-out logits (and so run.py) use flip TTA
        "tta": json.loads((RUNS_DIR / args.run / "oof_meta.json").read_text())["tta"]
        if (RUNS_DIR / args.run / "oof_meta.json").exists() else True,
    }
    if mode == "set":
        priors = torch.as_tensor(domain_priors(all_labels, all_domains, fold_ids),
                                 device=device, dtype=torch.float32)
        params, sforce, _ = fit_set(scorer, priors)
        final = eval_set(scorer, priors, params, sforce)
        print(f"set decoder, all-domain fit (in-sample): {params} {fmt(final)}")
        out.update({
            "mode": "set",
            "set_params": dict(params, max_k=20),
            "priors": [round(v, 6) for v in priors.tolist()],
            "force_groups": {g: GROUPS[g] for g, on in sforce.items() if on},
        })
    path = RUNS_DIR / args.run / "decoder.json"
    atomic_write_bytes(path, json.dumps(out, indent=2).encode())
    print(f"saved {path}")


if __name__ == "__main__":
    main()
