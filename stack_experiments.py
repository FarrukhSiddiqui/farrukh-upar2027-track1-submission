"""Nested-LODO test of second-stage ideas on held-out logits (CPU only).

Variants, all scored with the per-image set decoder:
  base      ViT-L WiSE 0.9 logits as they are (the shipped model)
  stackL    per-attribute logistic regression on all 40 ViT-L logits
            (learns label correlations, e.g. skirt -> female)
  stackLB   same on ViT-L + ViT-B logits (80 features)
  stackLZ   same on ViT-L + zero-shot ViT-B/16 logits
  fuseZ b   ViT-L logit + b * zero-shot logit (ViT-B/16 or ViT-L/14)

Protocol for each scored domain C (training domains A, B):
  * the second stage is fitted on A+B and applied to C;
  * the decoder is tuned on A+B logits produced by cross-fitting
    (stage fitted on B applied to A, and vice versa), so it never sees
    in-sample second-stage outputs.
The decoder grid is reduced for CPU time and identical for every variant.

    python stack_experiments.py
"""
import sys

import numpy as np
import torch

import fit_decoder as fd
from upar_common import DOMAINS, RUNS_DIR

torch.set_num_threads(6)
DEV = torch.device("cpu")
GRID = {"T": [0.7, 0.85, 1.0], "w": [0.35, 0.5, 0.65, 0.8], "rho": [0.3, 0.45]}


def load(run):
    lg, idx = [], []
    for d in DOMAINS:
        lg.append(np.load(RUNS_DIR / run / d / "oof_logits.npy"))
        idx.append(np.load(RUNS_DIR / run / d / "oof_index.npy"))
    return np.concatenate(lg), np.concatenate(idx)


def fit_stacker(X, Y, dom, lam=1e-3, iters=200):
    """Per-attribute L2 logistic regression; every domain weighs equally."""
    X = torch.as_tensor(X); Y = torch.as_tensor(Y, dtype=torch.float32)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-3)
    Z = (X - mu) / sd
    w_dom = torch.as_tensor(dom)
    wts = torch.zeros(len(Z))
    for d in w_dom.unique():
        wts[w_dom == d] = 1.0 / int((w_dom == d).sum())
    wts = wts / wts.sum()
    W = torch.zeros(Z.shape[1], 40, requires_grad=True)
    b = torch.zeros(40, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            Z @ W + b, Y, reduction="none").mean(1) * wts).sum() + lam * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    W, b = W.detach(), b.detach()
    return lambda Xn: ((torch.as_tensor(Xn) - mu) / sd) @ W + b


def fit_decoder(logits, gt, dom, priors):
    s = fd.Scorer(logits, gt, dom, DEV)
    best, prm = -1.0, None
    for T in GRID["T"]:
        for w in GRID["w"]:
            for rho in GRID["rho"]:
                sc = float(s.score(fd.decode_set(s.logits, priors, T, w, rho).unsqueeze(0))[0][0])
                if sc > best:
                    best, prm = sc, (T, w, rho)
    return prm


def evaluate(name, feats, base, gt, dom, labels, domains, stage):
    """stage: None (use base) or 'stack' or ('fuse', beta) -- features from feats."""
    scores = []
    for c in range(3):
        tr, te = dom != c, dom == c
        a, b_ = [x for x in range(3) if x != c]
        pri = torch.as_tensor(fd.domain_priors(labels, domains, [a, b_]), dtype=torch.float32)

        def apply(fit_mask, app_mask):
            if stage is None:
                return torch.as_tensor(base[app_mask])
            if isinstance(stage, tuple):  # fusion, nothing learned beyond beta
                return torch.as_tensor(base[app_mask] + stage[1] * feats[app_mask])
            f = fit_stacker(feats[fit_mask], gt[fit_mask], dom[fit_mask])
            return f(feats[app_mask])

        # decoder tuned on cross-fitted A/B logits
        ma, mb = dom == a, dom == b_
        lg_tr = torch.empty(int(tr.sum()), 40)
        sub = np.where(tr)[0]
        lg_tr[np.isin(sub, np.where(ma)[0])] = apply(mb, ma).float()
        lg_tr[np.isin(sub, np.where(mb)[0])] = apply(ma, mb).float()
        prm = fit_decoder(lg_tr, gt[tr], dom[tr], pri)
        lg_te = apply(tr, te).float()
        st = fd.Scorer(lg_te, gt[te], dom[te], DEV)
        _, m = st.score(fd.decode_set(st.logits, pri, *prm).unsqueeze(0))
        scores.append(float(m["Challenge_Avg"][0]))
    print(f"{name:<26} " + "  ".join(f"{DOMAINS[i]} {s:.4f}" for i, s in enumerate(scores))
          + f"   mean {np.mean(scores):.4f}", flush=True)
    return float(np.mean(scores))


def main():
    _, _, _, _, labels, domains = fd.load_oof("clip_l14_v1_wise0.9_nt")  # full, cache-aligned
    L, idx = load("clip_l14_v1_wise0.9_nt")
    B, idx_b = load("clip_b16_v1_wise0.8_nt")
    assert (idx == idx_b).all()
    zB = np.load(RUNS_DIR / "zeroshot_clip_vit_b16" / "all_logits.npy")[idx]
    zL = np.load(RUNS_DIR / "zeroshot_clip_vit_l14" / "all_logits.npy")[idx]
    gt, dom = labels[idx], domains[idx]

    evaluate("base (shipped ViT-L)", None, L, gt, dom, labels, domains, None)
    evaluate("stackL (label corr.)", L, L, gt, dom, labels, domains, "stack")
    evaluate("stackLB (L + B logits)", np.concatenate([L, B], 1), L, gt, dom, labels, domains, "stack")
    evaluate("stackLZ (L + zero-shot B)", np.concatenate([L, zB], 1), L, gt, dom, labels, domains, "stack")
    for beta in [0.1, 0.2, 0.3]:
        evaluate(f"fuse zero-shot B, b={beta}", zB, L, gt, dom, labels, domains, ("fuse", beta))
    for beta in [0.1, 0.2]:
        evaluate(f"fuse zero-shot L, b={beta}", zL, L, gt, dom, labels, domains, ("fuse", beta))


if __name__ == "__main__":
    sys.exit(main())
