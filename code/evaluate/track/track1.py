#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import compute_erc_curve, load_matcher_frame, load_quality_frame, plot_erc, print_curve_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Track 1: ERC for SIFQ quality vs DeepPrint matcher.")
    ap.add_argument("--quality-csv", default="SFIQ-2/outputs/sifq_sd302_embeddings.csv")
    ap.add_argument("--matcher-csv", default="DeepPrint/outputs/deepprint_embeddings.csv")
    ap.add_argument("--output-dir", default="SFIQ-2/code/evaluate/result/track1")
    ap.add_argument("--quality-column", default="q_hat")
    ap.add_argument("--target-fmr", type=float, default=1e-2,
                    help="Primary operating FMR (also reported in the sweep).")
    ap.add_argument("--fmr-sweep", default="0.1,0.01,0.001,0.0001",
                    help="Comma-separated FMRs to report AUC_ERC at (transparency vs "
                         "cherry-picking a single FMR). Set '' to disable.")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quality_df = load_quality_frame(Path(args.quality_csv))
    matcher_df = load_matcher_frame(Path(args.matcher_csv))

    # Primary curve at --target-fmr (used for the plot + the saved curve).
    curve = compute_erc_curve(quality_df, str(args.quality_column), matcher_df, target_fmr=float(args.target_fmr))
    curve.to_csv(output_dir / "track1_erc_deepprint_matcher.csv", index=False)
    plot_erc({str(args.quality_column): curve}, output_dir / "track1_erc.png", "Track 1: ERC on DeepPrint matcher")
    print_curve_report(f"Track 1 / {args.quality_column} vs DeepPrint matcher (FMR={args.target_fmr})", curve)

    # FMR sweep — report AUC_ERC at several operating points so the headline number is not a
    # single cherry-picked FMR. Lower AUC_ERC = better (FNMR drops faster with rejection).
    fmr_sweep = [float(x) for x in str(args.fmr_sweep).split(",") if x.strip()]
    sweep = []
    print("\n[Track 1] AUC_ERC sweep over FMR (lower = better):")
    for fmr in fmr_sweep:
        c = compute_erc_curve(quality_df, str(args.quality_column), matcher_df, target_fmr=fmr)
        auc = float(c["auc_erc"].iloc[0])
        fnmr0 = float(c.loc[c["rejection_ratio"] == c["rejection_ratio"].min(), "fnmr"].iloc[0])
        fnmr_last = float(c.loc[c["rejection_ratio"] == c["rejection_ratio"].max(), "fnmr"].iloc[0])
        drop = fnmr0 - fnmr_last  # >0 means FNMR drops as we reject low-Q (good)
        sweep.append({"target_fmr": fmr, "auc_erc": auc,
                      "fnmr_at_reject0": fnmr0, "fnmr_at_reject_max": fnmr_last,
                      "fnmr_drop": drop})
        print(f"  FMR={fmr:<8g} AUC_ERC={auc:.4f}  FNMR {fnmr0:.3f}->{fnmr_last:.3f}  (drop {drop:+.3f})")

    report = {
        "quality_column": str(args.quality_column),
        "target_fmr": float(args.target_fmr),
        "auc_erc": float(curve["auc_erc"].iloc[0]),
        "fmr_sweep": sweep,
        "curve": curve.to_dict(orient="records"),
    }
    (output_dir / "track1_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved report to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
