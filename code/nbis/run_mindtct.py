#!/usr/bin/env python3
"""Extract NBIS minutiae (.xyt) for every test image via mindtct.

Run inside WSL where the image paths (.) and the NBIS binaries exist.
Resumable: skips images whose .xyt already exists.

Usage (WSL):
  python3 run_mindtct.py \
    --scores ./outputs/results/sifq_v4_test.csv \
    --mindtct ~/nbisinstall/bin/mindtct \
    --out-dir ~/nbis_xyt
"""
from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def stem_id(path: str) -> str:
    return Path(str(path)).stem


def one(args):
    mindtct, img, oroot = args
    xyt = oroot + ".xyt"
    if os.path.exists(xyt) and os.path.getsize(xyt) > 0:
        return (oroot, "skip", 0)
    if not os.path.exists(img):
        return (oroot, "missing_img", 0)
    try:
        # -m1: maintain direction/representation suited for bozorth3 matching
        r = subprocess.run([mindtct, "-m1", img, oroot],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120)
        if r.returncode != 0:
            return (oroot, f"err_rc{r.returncode}", 0)
        n = 0
        if os.path.exists(xyt):
            with open(xyt) as f:
                n = sum(1 for _ in f)
        return (oroot, "ok", n)
    except subprocess.TimeoutExpired:
        return (oroot, "timeout", 0)
    except Exception as e:  # noqa
        return (oroot, f"exc_{type(e).__name__}", 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--mindtct", default=os.path.expanduser("~/nbisinstall/bin/mindtct"))
    ap.add_argument("--out-dir", default=os.path.expanduser("~/nbis_xyt"))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    out = Path(os.path.expanduser(args.out_dir))
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.scores)
    paths = df["path"].astype(str).tolist()
    # stems should be unique (subject_sensor_roll_NN); warn if not
    stems = [stem_id(p) for p in paths]
    if len(set(stems)) != len(stems):
        raise SystemExit(f"Non-unique image stems ({len(set(stems))} unique of {len(stems)}) — id collision")

    jobs = [(os.path.expanduser(args.mindtct), p, str(out / s)) for p, s in zip(paths, stems)]
    counts = {"ok": 0, "skip": 0}
    fails = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            oroot, status, n = fu.result()
            counts[status] = counts.get(status, 0) + 1
            if status not in ("ok", "skip"):
                fails.append((oroot, status))
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}  ok={counts.get('ok',0)} skip={counts.get('skip',0)} fail={len(fails)}", flush=True)

    print("DONE", counts)
    if fails:
        print("FAILS (first 10):", fails[:10])
    # write index
    idx = pd.DataFrame({"path": paths, "stem": stems,
                        "xyt": [str(out / s) + ".xyt" for s in stems]})
    idx["xyt_exists"] = idx["xyt"].map(lambda x: os.path.exists(x) and os.path.getsize(x) > 0)
    idx.to_csv(out / "xyt_index.csv", index=False)
    print(f"index -> {out/'xyt_index.csv'}  ({int(idx['xyt_exists'].sum())}/{len(idx)} have minutiae)")


if __name__ == "__main__":
    main()
