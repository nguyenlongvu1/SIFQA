#!/usr/bin/env python3
"""Audit concept set bằng DỮ LIỆU THẬT (test held-out):
  1. aliveness (std) + corr-với-Q
  2. concept-concept correlation matrix  -> REDUNDANCY (concept nào trùng nhau)
  3. eta^2 theo sensor -> concept BÁM SENSOR (cao=xấu, không invariant) hay theo ảnh (thấp=tốt)
Chạy: python concept_audit.py --scores SFIQ-2/outputs/sifq_v3_test.csv [--scores2 v2.csv]
"""
import argparse
import numpy as np
import pandas as pd

# union of v2 (current) + legacy concept names so this audits both old and new scored CSVs
CC = ["ridge_valley_clarity", "noise_level", "contrast_uniformity", "usable_area",
      "ridge_frequency", "orientation_coherence",
      "continuity", "minutiae_reliability"]


def eta2_sensor(d, c):
    grand = d[c].mean()
    ss_tot = ((d[c] - grand) ** 2).sum()
    if ss_tot < 1e-12:
        return float("nan")
    ss_between = sum(len(g) * (g[c].mean() - grand) ** 2 for _, g in d.groupby("sensor"))
    return ss_between / ss_tot


def audit(path):
    d = pd.read_csv(path)
    cc = [c for c in CC if c in d.columns]
    print(f"\n===== {path}  (n={len(d)}) =====")
    print(f"{'concept':24} {'std':>6} {'corrQ':>7} {'eta2_sensor':>11}  verdict")
    for c in cc:
        std = d[c].std()
        cq = d[c].corr(d["q_hat"])
        e2 = eta2_sensor(d, c)
        dead = "CHẾT" if std < 0.1 else ("yếu" if std < 0.13 else "sống")
        sens = " BÁM-SENSOR" if (e2 == e2 and e2 > 0.30) else ""
        print(f"{c:24} {std:6.3f} {cq:+7.3f} {e2:11.3f}  {dead}{sens}")
    print("\nConcept-concept |corr| (redundancy >0.6 = TRÙNG):")
    M = d[cc].corr().abs()
    print("      " + " ".join(f"{c[:6]:>6}" for c in cc))
    for i, c in enumerate(cc):
        row = " ".join(f"{M.iloc[i, j]:6.2f}" for j in range(len(cc)))
        print(f"{c[:6]:>6} {row}")
    # các cặp trùng nhất
    pairs = []
    for i in range(len(cc)):
        for j in range(i + 1, len(cc)):
            pairs.append((M.iloc[i, j], cc[i], cc[j]))
    pairs.sort(reverse=True)
    print("Top cặp trùng:")
    for v, a, b in pairs[:4]:
        print(f"  |r|={v:.2f}  {a} <-> {b}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--scores2", default="")
    args = ap.parse_args()
    audit(args.scores)
    if args.scores2:
        audit(args.scores2)


if __name__ == "__main__":
    main()
