"""Build weighted logit ensembles of two runs' held-out predictions for fit_decoder.

Writes runs/<name>/ with each fold's weighted-average held-out logits and an
ens_meta.json recording the members and weights, then (with --fit) runs
fit_decoder.py on it and adds "model_weights" to its decoder.json, in the
order make_submission.py --models must list the members.

    python ensemble_eval.py --runs clip_l14_v1_wise0.9_nt siglip2_l16_v1_wise0.9_nt \
        --weights 0.65 0.35 --name ens_cl65 --fit
"""
import argparse
import json
import subprocess
import sys

import numpy as np

from upar_common import DOMAINS, RUNS_DIR, atomic_np_save, atomic_write_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--fit", action="store_true", help="run fit_decoder.py afterwards")
    args = ap.parse_args()
    assert len(args.runs) == len(args.weights)
    w = np.asarray(args.weights) / np.sum(args.weights)

    dst = RUNS_DIR / args.name
    metas = [json.loads((RUNS_DIR / r / "oof_meta.json").read_text())
             if (RUNS_DIR / r / "oof_meta.json").exists() else {"tta": True} for r in args.runs]
    assert len({m["tta"] for m in metas}) == 1, "members must share the TTA setting"
    for fold in DOMAINS:
        idx = [np.load(RUNS_DIR / r / fold / "oof_index.npy") for r in args.runs]
        assert all((i == idx[0]).all() for i in idx), f"{fold}: held-out indices differ"
        lg = sum(wi * np.load(RUNS_DIR / r / fold / "oof_logits.npy") for wi, r in zip(w, args.runs))
        (dst / fold).mkdir(parents=True, exist_ok=True)
        atomic_np_save(dst / fold / "oof_logits.npy", lg.astype(np.float32))
        atomic_np_save(dst / fold / "oof_index.npy", idx[0])
    atomic_write_bytes(dst / "oof_meta.json", json.dumps(metas[0]).encode())
    atomic_write_bytes(dst / "ens_meta.json",
                       json.dumps({"runs": args.runs, "weights": w.tolist()}).encode())
    if args.fit:
        out = subprocess.run([sys.executable, "fit_decoder.py", "--run", args.name],
                             capture_output=True, text=True, check=True).stdout
        print("\n".join(l for l in out.splitlines() if "mean @" in l))
        dec_path = dst / "decoder.json"
        dec = json.loads(dec_path.read_text())
        dec["model_weights"] = w.tolist()
        dec["ensemble_runs"] = args.runs
        atomic_write_bytes(dec_path, json.dumps(dec, indent=2).encode())


if __name__ == "__main__":
    main()
