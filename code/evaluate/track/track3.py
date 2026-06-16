#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import compute_erc_curve, load_matcher_frame, load_quality_frame, plot_erc, print_curve_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Track 3: cross-matcher transfer for SIFQ quality.")
    ap.add_argument("--quality-csv", default="SFIQ-2/outputs/sifq_sd302_embeddings.csv")
    ap.add_argument("--deepprint-matcher-csv", default="DeepPrint/outputs/task2_deepprint/DeepPrint_embeddings.csv")
    ap.add_argument("--sifq-matcher-csv", default="SFIQ-2/outputs/sifq_sd302_embeddings.csv")
    ap.add_argument("--output-dir", default="SFIQ-2/code/evaluate/result/track3")
    ap.add_argument("--quality-column", default="q_hat")
    ap.add_argument("--target-fmr", type=float, default=1e-2)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quality_df = load_quality_frame(Path(args.quality_csv))
    deepprint_df = load_matcher_frame(Path(args.deepprint_matcher_csv))
    sifq_df = load_matcher_frame(Path(args.sifq_matcher_csv))

    deepprint_curve = compute_erc_curve(quality_df, str(args.quality_column), deepprint_df, target_fmr=float(args.target_fmr))
    sifq_curve = compute_erc_curve(quality_df, str(args.quality_column), sifq_df, target_fmr=float(args.target_fmr))

    deepprint_curve.to_csv(output_dir / "track3_erc_deepprint_matcher.csv", index=False)
    sifq_curve.to_csv(output_dir / "track3_erc_sifq_matcher.csv", index=False)
    plot_erc(
        {"DeepPrint matcher": deepprint_curve, "SIFQ matcher": sifq_curve},
        output_dir / "track3_cross_matcher_transfer.png",
        "Track 3: Cross-matcher transfer across matchers",
    )
    print_curve_report(f"Track 3 / {args.quality_column} vs DeepPrint matcher", deepprint_curve)
    print_curve_report(f"Track 3 / {args.quality_column} vs SIFQ matcher", sifq_curve)
    delta_auc = float(sifq_curve["auc_erc"].iloc[0] - deepprint_curve["auc_erc"].iloc[0])
    print(f"[Track 3] AUC delta (SIFQ matcher - DeepPrint matcher) = {delta_auc:.6f}")

    report = {
        "quality_column": str(args.quality_column),
        "target_fmr": float(args.target_fmr),
        "embedding_matcher_auc_erc": float(deepprint_curve["auc_erc"].iloc[0]),
        "sifq_matcher_auc_erc": float(sifq_curve["auc_erc"].iloc[0]),
        "delta_auc": delta_auc,
        "deepprint_curve": deepprint_curve.to_dict(orient="records"),
        "sifq_curve": sifq_curve.to_dict(orient="records"),
    }
    (output_dir / "track3_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved report to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
