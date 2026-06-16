#!/usr/bin/env python3
"""
run_all_eval.py — chạy TOÀN BỘ eval cho 1 checkpoint trên TEST (held-out) và in tóm tắt:
  1. SCORE test  -> sifq_<tag>_test.csv (q_hat + 6 concept + emb)
  2. TRACK 2     -> KS invariance giữa các sensor (so NFIQ2 0.629)
  3. TRACK 1     -> Bozorth ERC AUC theo FMR (so NFIQ2)
  4. TRACK 4     -> concept-vs-Q trên ảnh thật (corr + std => concept sống/chết)
Tùy chọn --compare <scores_csv_cũ> để so v3 vs v2.

Chạy TỪ REPO ROOT (chỗ chứa data/ và SFIQ-2/):
  python SFIQ-2/code/evaluate/run_all_eval.py \
      --run SFIQ-2/experiments/run_flare_v3 --tag v3 \
      --compare SFIQ-2/outputs/sifq_flare_v2_test.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # repo root (parents: evaluate, code, SFIQ-2, ROOT)
CODE = ROOT / "SFIQ-2" / "code"
# union of v2 (current) + legacy concept names so compare works across old/new scored CSVs
CC = ["ridge_valley_clarity", "noise_level", "contrast_uniformity", "usable_area",
      "ridge_frequency", "orientation_coherence",
      "continuity", "minutiae_reliability"]


def sh(cmd):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def concept_report(scores_csv):
    d = pd.read_csv(scores_csv)
    rep = {
        "q_std": float(d["q_hat"].std()),
        "q_range": (float(d["q_hat"].min()), float(d["q_hat"].max())),
        "concepts": {c: {"corr_Q": float(d[c].corr(d["q_hat"])), "std": float(d[c].std())}
                     for c in CC if c in d.columns},
    }
    rep["alive"] = sum(1 for c in CC if c in d.columns and d[c].std() > 0.1)
    return rep


def ks_mean(track2_dir):
    f = glob.glob(str(track2_dir) + "/*ks_pairs.csv")
    if not f:
        return None
    ks = pd.read_csv(f[0])
    col = [c for c in ks.columns if "ks" in c.lower()][0]
    return float(ks[col].mean())


def erc_auc(erc_dir):
    p = Path(erc_dir) / "erc_bozorth_report.json"
    if not p.exists():
        return None
    r = json.load(open(p))
    return {float(s["target_fmr"]): float(s["auc_erc_q"]) for s in r["fmr_sweep"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="experiment dir (has checkpoints/ + sensor_map.json)")
    ap.add_argument("--tag", required=True, help="short label, e.g. v3")
    ap.add_argument("--compare", default="", help="previous scores CSV to compare against (e.g. v2)")
    ap.add_argument("--py", default=sys.executable, help="python interpreter")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    py = args.py
    run_dir = Path(args.run)
    tag = args.tag
    ckpt = run_dir / "checkpoints" / "stage4_finetune_final.pt"
    smap = run_dir / "sensor_map.json"
    scores = ROOT / "SFIQ-2" / "outputs" / f"sifq_{tag}_test.csv"
    t2_dir = ROOT / "SFIQ-2" / "outputs" / f"track2_{tag}"
    erc_dir = ROOT / "SFIQ-2" / "outputs" / "nbis" / tag

    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    print("=== 1) SCORE test ===", flush=True)
    sh([py, CODE / "score_manifest_sifq.py", "--ckpt", ckpt, "--manifest", ROOT / "data/manifest_all.csv",
        "--split", "test", "--sensor-map", smap, "--output-csv", scores, "--device", args.device, "--num-workers", "8"])

    print("=== 2) TRACK 2 (invariance KS) ===", flush=True)
    sh([py, CODE / "evaluate/track/track2.py", "--input-csv", scores, "--output-dir", t2_dir, "--quality-column", "q_hat"])

    print("=== 3) TRACK 1 (Bozorth ERC) ===", flush=True)
    sh([py, CODE / "nbis/erc_bozorth.py", "--pairs", ROOT / "SFIQ-2/outputs/nbis/pairs_scores.csv",
        "--scores", scores, "--quality-col", "q_hat", "--out-dir", erc_dir])

    # ---- summary ----
    cr = concept_report(scores)
    ks = ks_mean(t2_dir)
    auc = erc_auc(erc_dir) or {}
    nfiq2 = erc_auc(ROOT / "SFIQ-2/outputs/nbis/nfiq2") or {}

    print("\n" + "=" * 60)
    print(f"SUMMARY [{tag}]  (TEST held-out)")
    print("=" * 60)
    print(f"Track2 invariance: KS_mean = {ks:.3f}   (NFIQ2 0.629; THẤP=tốt)" if ks is not None else "Track2: (no KS)")
    print(f"Q: std={cr['q_std']:.2f}  range={cr['q_range'][0]:.0f}-{cr['q_range'][1]:.0f}  | concept SỐNG: {cr['alive']}/6 (std>0.1)")
    print("Track4 concept-vs-Q (ảnh thật):")
    for c in CC:
        v = cr["concepts"].get(c)
        if v:
            dead = " <-- CHẾT" if v["std"] < 0.1 else ""
            print(f"  {c:24} corr={v['corr_Q']:+.3f}  std={v['std']:.3f}{dead}")
    print("Track1 Bozorth AUC_ERC (THẤP=tốt):")
    for fmr in [0.1, 0.01, 0.001, 0.0001]:
        a = auc.get(fmr)
        n = nfiq2.get(fmr)
        line = f"  FMR {fmr:<7}: {tag} {a:.3f}" if a is not None else f"  FMR {fmr}: (n/a)"
        if a is not None and n is not None:
            verdict = "WIN " if a < n else ("lose" if a > n else "tie ")
            line += f" | NFIQ2 {n:.3f}  [{verdict}]"
        print(line)

    if args.compare and Path(args.compare).exists():
        c2 = concept_report(args.compare)
        print("\n--- so với", Path(args.compare).name, "---")
        print(f"  Q std        : {tag} {cr['q_std']:.2f}  vs  {c2['q_std']:.2f}")
        print(f"  concept sống : {tag} {cr['alive']}/6  vs  {c2['alive']}/6")
        for c in CC:
            a = cr["concepts"].get(c, {}).get("std", 0.0)
            b = c2["concepts"].get(c, {}).get("std", 0.0)
            flag = " (sống lại)" if b < 0.1 <= a else (" (chết đi)" if a < 0.1 <= b else "")
            print(f"  {c:24} std {a:.3f} vs {b:.3f}{flag}")

    print("\nOutputs:", scores, "|", t2_dir, "|", erc_dir)


if __name__ == "__main__":
    main()
