#!/usr/bin/env python3
"""Generate genuine + impostor pairs and score them with NBIS bozorth3.

Finger identity = identity + "_" + roll (a finger = subject's one finger, captured
once per sensor). Genuine = same finger, different capture (necessarily cross-sensor
here). Impostor = different finger (sampled).

Output: pairs_scores.csv  with columns
  probe_stem, gallery_stem, probe_path, label(1=genuine/0=impostor), score, q_probe

Run in WSL:
  python3 run_bozorth.py \
    --scores ./results/sifq_v4_test.csv \
    --xyt-index ~/nbis_xyt/xyt_index.csv \
    --bozorth3 ~/nbisinstall/bin/bozorth3 \
    --out ./outputs/nbis/pairs_scores.csv \
    --impostors-per-probe 30 --seed 0
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--xyt-index", required=True)
    ap.add_argument("--bozorth3", default=os.path.expanduser("~/nbisinstall/bin/bozorth3"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--impostors-per-probe", type=int, default=30)
    ap.add_argument("--max-genuine-per-finger", type=int, default=0, help="0 = all C(n,2)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sc = pd.read_csv(args.scores)
    sc["fid"] = sc["identity"].astype(str) + "_" + sc["roll"].astype(str)
    sc["stem"] = sc["path"].map(lambda p: Path(str(p)).stem)

    idx = pd.read_csv(args.xyt_index)
    idx = idx[idx["xyt_exists"]].copy()
    have = dict(zip(idx["stem"], idx["xyt"]))
    sc = sc[sc["stem"].isin(have)].copy()
    print(f"images with minutiae: {len(sc)} (of scores file)")

    q = dict(zip(sc["stem"], sc["q_hat"]))
    path_of = dict(zip(sc["stem"], sc["path"]))
    stems = sc["stem"].tolist()
    all_stems = np.array(stems)
    by_fid = sc.groupby("fid")["stem"].apply(list).to_dict()
    codes = pd.Categorical(sc["fid"]).codes.astype(np.int64)  # per-stem finger code

    pairs = []  # (probe_stem, gallery_stem, label)
    # genuine: within-finger unordered pairs
    for fid, ss in by_fid.items():
        cmb = list(combinations(ss, 2))
        if args.max_genuine_per_finger and len(cmb) > args.max_genuine_per_finger:
            sel = rng.choice(len(cmb), size=args.max_genuine_per_finger, replace=False)
            cmb = [cmb[i] for i in sel]
        for a, b in cmb:
            pairs.append((a, b, 1))
    n_gen = len(pairs)
    print(f"genuine pairs={n_gen}; sampling impostors...", flush=True)

    # impostor: per probe, sample K stems from other fingers (vectorized on finger codes)
    K = args.impostors_per_probe
    for i in range(len(stems)):
        cand_idx = np.flatnonzero(codes != codes[i])
        k = min(K, len(cand_idx))
        sel = rng.choice(cand_idx, size=k, replace=False)
        probe = stems[i]
        for j in sel:
            pairs.append((probe, stems[j], 0))
    n_imp = len(pairs) - n_gen
    print(f"pairs: genuine={n_gen}  impostor={n_imp}  total={len(pairs)}", flush=True)

    # write mates list (probe_xyt gallery_xyt per line) + run bozorth3 in chunks
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores = np.empty(len(pairs), dtype=np.int64)

    CHUNK = 20000
    boz = os.path.expanduser(args.bozorth3)
    for start in range(0, len(pairs), CHUNK):
        chunk = pairs[start:start + CHUNK]
        # bozorth3 -M mates list: ONE filename per line, consecutive lines paired
        # (line 2k = probe, line 2k+1 = gallery).
        with tempfile.NamedTemporaryFile("w", suffix=".lis", delete=False) as mf:
            for a, b, _ in chunk:
                mf.write(f"{have[a]}\n{have[b]}\n")
            mfile = mf.name
        # outfmt=s -> one score per line, same order as mates file
        r = subprocess.run([boz, "-m1", "-l", "-A", "outfmt=s", "-A", "maxfiles=300000", "-M", mfile],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        os.unlink(mfile)
        vals = [int(x) for x in r.stdout.split()]
        if len(vals) != len(chunk):
            raise SystemExit(f"bozorth3 returned {len(vals)} scores for {len(chunk)} pairs "
                             f"(chunk @ {start}). stderr: {r.stderr[:300]}")
        scores[start:start + len(chunk)] = vals
        print(f"  scored {start+len(chunk)}/{len(pairs)}", flush=True)

    df = pd.DataFrame({
        "probe_stem": [p[0] for p in pairs],
        "gallery_stem": [p[1] for p in pairs],
        "probe_path": [path_of[p[0]] for p in pairs],
        "label": [p[2] for p in pairs],
        "score": scores,
        "q_probe": [q[p[0]] for p in pairs],
    })
    df.to_csv(out_path, index=False)
    g = df[df.label == 1]["score"].to_numpy()
    i = df[df.label == 0]["score"].to_numpy()
    print(f"GENUINE  score: med={np.median(g):.0f} mean={g.mean():.1f}")
    print(f"IMPOSTOR score: med={np.median(i):.0f} mean={i.mean():.1f}")
    # quick separation AUC
    a = np.concatenate([g, i]); lab = np.concatenate([np.ones(len(g)), np.zeros(len(i))])
    order = np.argsort(a, kind="mergesort"); ranks = np.empty(len(a)); ranks[order] = np.arange(1, len(a) + 1)
    auc = (ranks[lab == 1].sum() - len(g) * (len(g) + 1) / 2) / (len(g) * len(i))
    print(f"Genuine-vs-impostor ROC-AUC = {auc:.4f}  (Bozorth3 real matcher)")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
