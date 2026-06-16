#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import compute_track4_grounding, load_model, resolve_device


def main() -> None:
    ap = argparse.ArgumentParser(description="Track 4: concept grounding for SIFQ.")
    ap.add_argument("--ckpt", default="SFIQ-2/weights/stage4_finetune_final.pt")
    ap.add_argument("--image-root", default="data/SD302a/images/challengers")
    ap.add_argument("--output-dir", default="SFIQ-2/code/evaluate/result/track4")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-track4-images", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = resolve_device(str(args.device))  # "auto" -> cuda/cpu (load_model passes this to .to())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tf = load_model(Path(args.ckpt), device)
    track4_summary = compute_track4_grounding(
        model=model,
        tf=tf,
        image_root=Path(args.image_root),
        output_dir=output_dir,
        max_images=int(args.max_track4_images),
        seed=int(args.seed),
    )
    print("[Track 4] concept grounding (rho=monotonicity, range=magnitude) + Q deduction:")
    n_grounded = 0
    for row in track4_summary.to_dict(orient="records"):
        g = bool(row.get("grounded", False))
        n_grounded += int(g)
        print(
            f"  {'OK ' if g else '   '}{row['degradation']:<18} "
            f"target_rho={row['mean_target_rho']:+.3f}  range={row.get('mean_target_range', float('nan')):.3f}  "
            f"non_target_rho={row['mean_non_target_rho']:+.3f}  "
            f"| Q_rho={row.get('q_rho_vs_level', float('nan')):+.3f}  Q_drop={row.get('q_drop_clean_to_max', float('nan')):+.1f}"
        )
    print(f"[Track 4] grounded {n_grounded}/{len(track4_summary)} degradations "
          f"(cần |rho|>0.5 đúng chiều VÀ range>0.2)")
    print("[Track 4] Q deduction: Q_rho nên ÂM MẠNH (Q tụt khi degrade tăng); "
          "Q_drop = số điểm Q mất từ sạch→hỏng nặng (thang 0..100).")

    report = {"summary": track4_summary.to_dict(orient="records")}
    (output_dir / "track4_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved report to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
