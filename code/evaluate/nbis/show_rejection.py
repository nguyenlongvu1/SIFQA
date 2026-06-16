#!/usr/bin/env python3
"""In bảng rejection-rate vs FNMR (ERC) cho SIFQ vs NFIQ2 ở các FMR."""
import argparse
import json


def curve(rep, fmr):
    for s in rep["fmr_sweep"]:
        if abs(s["target_fmr"] - fmr) < 1e-9:
            return s["fnmr_curve"], s["auc_erc_q"]
    return None, None


def near(points, r):
    # fnmr_curve = list FNMR tại reject = linspace(0, 0.5, len(points))
    n = len(points)
    idx = int(round(r / 0.5 * (n - 1)))
    idx = max(0, min(n - 1, idx))
    return points[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sifq", default="SFIQ-2/outputs/nbis/v3/erc_bozorth_report.json")
    ap.add_argument("--nfiq2", default="SFIQ-2/outputs/nbis/nfiq2/erc_bozorth_report.json")
    ap.add_argument("--fmrs", default="0.0001,0.001,0.01,0.1")
    args = ap.parse_args()

    sifq = json.load(open(args.sifq))
    nf = json.load(open(args.nfiq2))
    fmrs = [float(x) for x in args.fmrs.split(",")]
    rejects = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

    for fmr in fmrs:
        cv, av = curve(sifq, fmr)
        cn, an = curve(nf, fmr)
        if cv is None or cn is None:
            continue
        win = "SIFQ WIN" if av < an else ("NFIQ2" if an < av else "tie")
        print(f"=== FMR={fmr:g}   AUC_ERC: SIFQ {av:.3f} | NFIQ2 {an:.3f}  ({win}; thấp=tốt) ===")
        print(f"{'reject':>7} | {'FNMR SIFQ':>10} | {'FNMR NFIQ2':>11} | winner")
        f0v = near(cv, 0.0)
        f0n = near(cn, 0.0)
        for r in rejects:
            fv = near(cv, r)
            fn = near(cn, r)
            w = "SIFQ" if fv < fn - 1e-9 else ("NFIQ2" if fn < fv - 1e-9 else "tie")
            print(f"{r*100:6.0f}% | {fv:10.3f} | {fn:11.3f} | {w}")
        # tóm tắt mức giảm FNMR khi loại 50%
        print(f"  -> SIFQ giảm FNMR {f0v:.3f}->{near(cv,0.5):.3f} ({(1-near(cv,0.5)/f0v)*100:.0f}% relative) | "
              f"NFIQ2 {f0n:.3f}->{near(cn,0.5):.3f} ({(1-near(cn,0.5)/f0n)*100:.0f}%)")
        print()


if __name__ == "__main__":
    main()
