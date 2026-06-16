#!/usr/bin/env python3
"""Error-vs-Reject Characteristic (ERC) using Bozorth3 match scores + SIFQ quality.

Standard (Tabassi/Grother) ERC:
  - impostor threshold fixed once at a target FMR (high score => match).
  - genuine comparisons ordered by pair quality = min(q_probe, q_gallery).
  - progressively reject the lowest-quality genuine pairs; FNMR = fraction of the
    REMAINING genuine pairs scoring below threshold.
  A useful quality makes FNMR drop as rejection rises => low AUC_ERC.
Compared against a random-rejection baseline (quality carries no information).

Run in WSL (venv python with numpy/pandas):
  python3 erc_bozorth.py \
    --pairs ./outputs/nbis/pairs_scores.csv \
    --scores ./results/sifq_v4_test.csv \
    --quality-col q_hat --out-dir ./outputs/nbis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    a = np.concatenate([pos, neg])
    lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a)); ranks[order] = np.arange(1, len(a) + 1)
    return float((ranks[lab == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def erc_curve(pair_q: np.ndarray, gen_score: np.ndarray, threshold: float,
              grid: np.ndarray) -> tuple[np.ndarray, float]:
    """Reject lowest pair-quality fraction; FNMR among survivors. Returns (fnmr[], auc)."""
    order = np.argsort(pair_q, kind="mergesort")[::-1]  # high quality first
    gs = gen_score[order]
    n = len(gs)
    fnmr = []
    for rej in grid:
        keep = max(1, int(round((1.0 - rej) * n)))
        fnmr.append(float(np.mean(gs[:keep] < threshold)))
    fnmr = np.array(fnmr)
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return fnmr, float(trap(fnmr, grid))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--quality-col", default="q_hat")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fmr-sweep", default="0.1,0.01,0.001,0.0001")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.pairs)
    sc = pd.read_csv(args.scores)
    sc["stem"] = sc["path"].map(lambda p: Path(str(p)).stem)
    qmap = dict(zip(sc["stem"], sc[args.quality_col]))

    pairs["q_probe"] = pairs["probe_stem"].map(qmap)
    pairs["q_gal"] = pairs["gallery_stem"].map(qmap)
    pairs = pairs.dropna(subset=["q_probe", "q_gal"]).copy()

    gen = pairs[pairs.label == 1]
    imp = pairs[pairs.label == 0]
    g_score = gen["score"].to_numpy(float)
    i_score = imp["score"].to_numpy(float)
    pair_q = np.minimum(gen["q_probe"].to_numpy(float), gen["q_gal"].to_numpy(float))

    auc_sep = roc_auc(g_score, i_score)
    print(f"[matcher] Bozorth3 genuine-vs-impostor ROC-AUC = {auc_sep:.4f}  "
          f"(genuine med={np.median(g_score):.0f}, impostor med={np.median(i_score):.0f})")
    print(f"[data] genuine pairs={len(g_score)}  impostor={len(i_score)}\n")

    grid = np.linspace(0.0, 0.5, 11)
    fmrs = [float(x) for x in args.fmr_sweep.split(",") if x.strip()]
    rng = np.random.default_rng(0)
    report = {"matcher_roc_auc": auc_sep, "n_genuine": int(len(g_score)),
              "n_impostor": int(len(i_score)), "fmr_sweep": []}

    print("[ERC] FMR     thr   AUC_ERC(Q)  AUC_ERC(random)  FNMR@rej0 -> FNMR@rej0.5")
    for fmr in fmrs:
        thr = float(np.quantile(i_score, 1.0 - fmr))
        fnmr_q, auc_q = erc_curve(pair_q, g_score, thr, grid)
        # random baseline: average AUC over shuffles of quality
        aucs_r = []
        for _ in range(10):
            fr, ar = erc_curve(rng.permutation(pair_q), g_score, thr, grid)
            aucs_r.append(ar)
        auc_r = float(np.mean(aucs_r))
        print(f"      {fmr:<7g} {thr:>4.0f}   {auc_q:.4f}      {auc_r:.4f}         "
              f"{fnmr_q[0]:.3f} -> {fnmr_q[-1]:.3f}")
        report["fmr_sweep"].append({
            "target_fmr": fmr, "threshold": thr,
            "auc_erc_q": auc_q, "auc_erc_random": auc_r,
            "fnmr_at_rej0": float(fnmr_q[0]), "fnmr_at_rej_max": float(fnmr_q[-1]),
            "fnmr_curve": fnmr_q.tolist(),
        })

    (out / "erc_bozorth_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nInterpretation: AUC_ERC(Q) < AUC_ERC(random) AND FNMR dropping with rejection "
          f"=> SIFQ Q has real utility on a working matcher.")
    print(f"saved -> {out/'erc_bozorth_report.json'}")


if __name__ == "__main__":
    main()
