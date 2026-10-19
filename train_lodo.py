"""Leave-one-domain-out (LODO) training for UPAR 2027 Track 1.

The hidden test set is a domain the model never saw, so validation holds out
one whole source dataset (train+val images) and trains on the other two:

    python train_lodo.py --holdout PETA        # fold: train Market+PA100k, eval PETA
    python train_lodo.py --holdout all-folds   # the three folds back to back
    python train_lodo.py --holdout none        # final model on all three datasets

Each fold writes runs/<name>/<holdout>/{model.pth, oof_logits.npy, log.jsonl}.
fit_decoder.py turns the held-out logits of all folds into thresholds.
"""
import argparse
import json
import math
import queue
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from upar_common import (DOMAINS, IMAGE_SIZE, MEAN, RUNS_DIR, AttributeNet,
                         load_cache, per_domain_metrics)


# --------------------------------------------------------------------------- data

class BatchStream:
    """Draws index batches, gathers pixels from the memmap in a background thread."""

    def __init__(self, images, labels, indices, batch_size, weights=None, shuffle=True,
                 drop_last=True, prefetch=6):
        self.images, self.labels = images, labels
        self.indices = np.asarray(indices)
        self.batch_size, self.shuffle, self.drop_last = batch_size, shuffle, drop_last
        self.prefetch = prefetch
        self.p = None if weights is None else weights / weights.sum()

    def __len__(self):
        n = len(self.indices) / self.batch_size
        return int(n) if self.drop_last else math.ceil(n)

    def _order(self):
        if not self.shuffle:
            return self.indices
        if self.p is None:
            return np.random.permutation(self.indices)
        return np.random.choice(self.indices, size=len(self.indices), replace=True, p=self.p)

    def __iter__(self):
        order = self._order()
        q = queue.Queue(self.prefetch)

        def worker():
            for b in range(len(self)):
                idx = order[b * self.batch_size:(b + 1) * self.batch_size]
                srt = np.sort(idx)  # sorted reads are friendlier to the memmap
                q.put((torch.from_numpy(np.ascontiguousarray(self.images[srt])).pin_memory(),
                       torch.from_numpy(self.labels[srt].astype(np.float32)), srt))
            q.put(None)

        threading.Thread(target=worker, daemon=True).start()
        while (item := q.get()) is not None:
            yield item


def to_input(u8, device):
    """uint8 NHWC on CPU -> normalized float NCHW on device (channels_last)."""
    x = u8.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


def gpu_augment(x, strength=1.0):
    """Augment a normalized batch on the GPU. No hue changes, since colours are labels."""
    n, _, h, w = x.shape
    dev = x.device
    # horizontal flip
    flip = torch.rand(n, device=dev) < 0.5
    x = torch.where(flip.view(n, 1, 1, 1), x.flip(3), x)
    # random scale / translate / small aspect change via affine grid
    s = 1 + (torch.rand(n, device=dev) * 0.25 - 0.15) * strength
    ar = 1 + (torch.rand(n, device=dev) * 0.2 - 0.1) * strength
    tx = (torch.rand(n, device=dev) * 2 - 1) * 0.08 * strength
    ty = (torch.rand(n, device=dev) * 2 - 1) * 0.08 * strength
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0] = s * ar
    theta[:, 1, 1] = s / ar
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    # low-resolution simulation: down-sample then up-sample a third of the batch
    lowres = torch.rand(n, device=dev) < 0.33 * strength
    if lowres.any():
        f = float(np.random.uniform(0.25, 0.6))
        small = F.interpolate(x[lowres], scale_factor=f, mode="bilinear", antialias=True)
        x[lowres] = F.interpolate(small, size=(h, w), mode="bilinear")
    # brightness / contrast / saturation (in normalized space, approximately)
    b = (torch.rand(n, 1, 1, 1, device=dev) * 0.4 - 0.2) * strength
    c = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 0.4 - 0.2) * strength
    gray = x.mean(1, keepdim=True)
    sat = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 0.4 - 0.2) * strength
    x = gray + (x - gray) * sat
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * c + m + b
    # random erasing
    erase = torch.rand(n, device=dev) < 0.4 * strength
    if erase.any():
        eh = (torch.rand(n, device=dev) * 0.3 + 0.1) * h
        ew = (torch.rand(n, device=dev) * 0.5 + 0.2) * w
        y0 = torch.rand(n, device=dev) * (h - eh)
        x0 = torch.rand(n, device=dev) * (w - ew)
        yy = torch.arange(h, device=dev).view(1, h, 1)
        xx = torch.arange(w, device=dev).view(1, 1, w)
        box = ((yy >= y0.view(n, 1, 1)) & (yy < (y0 + eh).view(n, 1, 1))
               & (xx >= x0.view(n, 1, 1)) & (xx < (x0 + ew).view(n, 1, 1)) & erase.view(n, 1, 1))
        x = torch.where(box.unsqueeze(1), torch.randn_like(x) * 0.5, x)
    return x.contiguous(memory_format=torch.channels_last)


# --------------------------------------------------------------------------- train

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].lerp_(v, 1 - self.decay)
            else:
                self.shadow[k].copy_(v)


@torch.no_grad()
def predict_logits(model, stream, device, tta=True):
    model.eval()
    out = []
    for u8, _, _ in stream:
        x = to_input(u8, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logit = model(x).float()
            if tta:
                logit = (logit + model(x.flip(3)).float()) / 2
        out.append(logit.cpu())
    return torch.cat(out).numpy()


def run_fold(args, holdout, images, labels, domains):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if holdout == "none":
        train_idx = np.arange(len(labels))
        eval_idx = np.array([], dtype=np.int64)
    else:
        h = DOMAINS.index(holdout)
        train_idx = np.where(domains != h)[0]
        eval_idx = np.where(domains == h)[0]
    if args.limit:
        rng = np.random.default_rng(0)
        train_idx = np.sort(rng.choice(train_idx, min(args.limit, len(train_idx)), replace=False))
        if len(eval_idx):
            eval_idx = np.sort(rng.choice(eval_idx, min(args.limit // 2, len(eval_idx)), replace=False))

    # Domain balancing: sample weight per image ∝ domain_size^(-alpha).
    counts = np.bincount(domains[train_idx], minlength=len(DOMAINS)).astype(np.float64)
    w_dom = np.where(counts > 0, np.maximum(counts, 1) ** -args.domain_alpha, 0)
    weights = w_dom[domains[train_idx]]
    share = {DOMAINS[d]: float(w_dom[d] * counts[d] / (w_dom * counts).sum())
             for d in range(len(DOMAINS)) if counts[d]}

    out_dir = RUNS_DIR / args.name / holdout
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== fold holdout={holdout}: train {len(train_idx)}, eval {len(eval_idx)}, "
          f"sampling share {share}")

    train_stream = BatchStream(images, labels, train_idx, args.batch_size, weights=weights)
    eval_stream = BatchStream(images, labels, eval_idx, 256, shuffle=False, drop_last=False)

    model = AttributeNet(args.backbone, pretrained=True, drop=args.dropout).to(device)
    model = model.to(memory_format=torch.channels_last)
    head_params = list(model.head.parameters()) + list(model.norm.parameters())
    head_ids = {id(p) for p in head_params}
    back_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([
        {"params": back_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=args.weight_decay)
    total = args.epochs * len(train_stream)
    warm = min(len(train_stream), total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, total - warm)))))
    ema = EMA(model, args.ema)
    eval_model = AttributeNet(args.backbone, pretrained=False, drop=args.dropout).to(device)
    eval_model = eval_model.to(memory_format=torch.channels_last)

    log = (out_dir / "log.jsonl").open("w")
    best, step = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss = time.time(), 0.0
        bar = tqdm(train_stream, desc=f"[{holdout}] epoch {epoch}/{args.epochs}", leave=False)
        for i, (u8, y, _) in enumerate(bar):
            x = gpu_augment(to_input(u8, device), args.aug)
            y = y.to(device, non_blocking=True)
            if args.label_smooth:
                y = y * (1 - args.label_smooth) + 0.5 * args.label_smooth
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            ema.update(model)
            step += 1
            run_loss += loss.item() if i % 20 == 0 else 0
            if i % 20 == 0:
                bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
        rec = {"epoch": epoch, "train_loss_sampled": run_loss / max(1, len(train_stream) // 20),
               "time_s": round(time.time() - t0, 1)}

        if len(eval_idx):
            eval_model.load_state_dict(ema.shadow)
            logits = predict_logits(eval_model, eval_stream, device, tta=False)
            mean, _ = per_domain_metrics(labels[eval_idx], logits > 0, domains[eval_idx])
            rec.update({f"ood_{k}": round(v, 4) for k, v in mean.items()})
            if mean["Challenge_Avg"] > best:
                best = mean["Challenge_Avg"]
        print(json.dumps(rec))
        log.write(json.dumps(rec) + "\n")
        log.flush()

    # Keep the final EMA weights (no early stopping on the held-out domain:
    # the final all-data model has no held-out set to stop on either).
    state = {k: v.cpu() for k, v in ema.shadow.items()}
    torch.save({"state_dict": state, "backbone": args.backbone, "image_size": list(IMAGE_SIZE),
                "holdout": holdout, "args": vars(args)}, out_dir / "model.pth")
    if len(eval_idx):
        eval_model.load_state_dict(ema.shadow)
        logits = predict_logits(eval_model, eval_stream, device, tta=True)
        np.save(out_dir / "oof_logits.npy", logits.astype(np.float32))
        np.save(out_dir / "oof_index.npy", eval_idx)
        mean, _ = per_domain_metrics(labels[eval_idx], logits > 0, domains[eval_idx])
        print(f"[{holdout}] final EMA+flipTTA @0.5: " +
              ", ".join(f"{k}={v:.4f}" for k, v in mean.items()))
    print(f"saved {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="all-folds",
                    help="Market1501 | PA100k | PETA | all-folds | none")
    ap.add_argument("--name", default="convnext_tiny_v1")
    ap.add_argument("--backbone", default="convnext_tiny")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--head-lr-mult", type=float, default=10.0)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--label-smooth", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--aug", type=float, default=1.0, help="augmentation strength")
    ap.add_argument("--domain-alpha", type=float, default=0.5,
                    help="0 = natural mix, 1 = equal share per domain")
    ap.add_argument("--limit", type=int, default=0, help="subsample for smoke tests")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images, labels, domains, _, _ = load_cache()
    folds = list(DOMAINS) if args.holdout == "all-folds" else [args.holdout]
    for holdout in folds:
        run_fold(args, holdout, images, labels, domains)


if __name__ == "__main__":
    main()
