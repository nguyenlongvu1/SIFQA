#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import os
import random
import re
from pathlib import Path

import cv2
import numpy as np


def preprocess_cl(bgr: np.ndarray, size: int = 512) -> np.ndarray:
    """Contactless camera image (BGR) → contact-like grayscale square (size×size)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sig = max(gray.shape) / 16.0
    bg = cv2.GaussianBlur(gray, (0, 0), sig)
    hp = gray - bg                                   # remove low-freq illumination/colour gradient
    # ROI = largest connected high ridge-energy region (finger usually fills frame)
    energy = cv2.GaussianBlur(np.abs(hp), (0, 0), sig / 2)
    th = (energy > 0.25 * float(energy.max() + 1e-6)).astype(np.uint8)
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(th, 8)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h = (int(stats[k, i]) for i in (0, 1, 2, 3))
    else:
        x, y, w, h = 0, 0, gray.shape[1], gray.shape[0]
    cx, cy = x + w // 2, y + h // 2
    s = max(32, int(max(w, h) * 0.95))
    x0, y0 = max(0, cx - s // 2), max(0, cy - s // 2)
    x1, y1 = min(gray.shape[1], cx + s // 2), min(gray.shape[0], cy + s // 2)
    # high-pass → flat/background auto-collapse to mid-grey; CLAHE boosts ridges
    norm = (hp - hp.min()) / (float(np.ptp(hp)) + 1e-6) * 255.0
    enh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(norm.astype(np.uint8))
    enh = 255 - enh                                  # ridges dark (match contact polarity)
    crop = enh[y0:y1, x0:x1]
    if crop.size == 0:
        crop = enh
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def process_all_contactless(data_root: Path, size: int, skip_existing: bool) -> int:
    src_root = data_root / "PolyU" / "contactless_2d_fingerprint_images"
    n_done = 0
    for bmp in sorted(src_root.rglob("*.bmp")):
        rel = bmp.relative_to(src_root)
        dst = data_root / "PolyU" / "contactless_processed" / rel.with_suffix(".png")
        if skip_existing and dst.exists():
            n_done += 1
            continue
        bgr = cv2.imread(str(bmp), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[warn] unreadable: {bmp}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), preprocess_cl(bgr, size=size))
        n_done += 1
        if n_done % 500 == 0:
            print(f"  processed {n_done} contactless images")
    return n_done


def relpath(p: Path, root: Path) -> str:
    return os.path.relpath(p, root).replace("\\", "/")


def build_polyu_rows(data_root: Path) -> list[tuple]:
    """Return rows (path_rel, sensor, finger_int, modality) for CB(.jpg) + CL(processed .png).

    Paths are relative to the SIFQ repo root (parent of data/) so they match the
    'data/...' convention in manifest_all.csv (e.g. 'data/PolyU/...').
    """
    repo_root = data_root.parent
    rows = []
    poly = data_root / "PolyU"
    for sess in ("first_session", "second_session"):
        for f in sorted((poly / "contact-based_fingerprints" / sess).glob("*.jpg")):
            m = re.match(r"(\d+)_(\d+)", f.name)
            if m:
                rows.append((relpath(f, repo_root), "PolyU_CB", int(m.group(1)), "CB"))
    for sess in ("first_session", "second_session"):
        for d in sorted((poly / "contactless_processed" / sess).glob("p*")):
            if not d.is_dir():
                continue
            m = re.match(r"p(\d+)$", d.name)
            if not m:
                continue
            fid = int(m.group(1))
            for f in sorted(d.glob("*.png")):
                rows.append((relpath(f, repo_root), "PolyU_CL", fid, "CL"))
    return rows


def main() -> None:
    here = Path(__file__).resolve()
    default_data = here.parents[2] / "data"
    ap = argparse.ArgumentParser(description="Preprocess PolyU contactless + (re)build PolyU manifests.")
    ap.add_argument("--data-root", default=str(default_data))
    ap.add_argument("--size", type=int, default=512, help="Output square size for processed contactless.")
    ap.add_argument("--skip-existing", action="store_true", help="Skip already-processed images.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-process", action="store_true", help="Only rebuild manifests (skip image processing).")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    cols = ["path", "sensor", "finger_id", "roll", "dataset", "split", "q_gt"]

    if not args.no_process:
        n = process_all_contactless(data_root, int(args.size), bool(args.skip_existing))
        print(f"Contactless processed: {n} images -> {data_root/'PolyU'/'contactless_processed'}")

    # ---- finger-level 70/10/20 split (same seed => reproducible) ----
    rows = build_polyu_rows(data_root)
    if not rows:
        raise SystemExit("No PolyU rows found (did processing run? check contactless_processed/).")
    fingers = sorted({r[2] for r in rows})
    sh = fingers[:]
    random.Random(int(args.seed)).shuffle(sh)
    n = len(sh)
    n_tr, n_va = round(n * 0.7), round(n * 0.1)
    train, val = set(sh[:n_tr]), set(sh[n_tr:n_tr + n_va])

    def split_of(fid: int) -> str:
        return "train" if fid in train else ("val" if fid in val else "test")

    poly_csv = data_root / "manifest_polyu.csv"
    with open(poly_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for path, sensor, fid, _mod in sorted(rows):
            w.writerow([path, sensor, f"polyu_{fid}", 1, "polyu", split_of(fid), ""])
    print(f"Wrote {poly_csv} ({len(rows)} rows, fingers {n}: train {len(train)}/val {len(val)}/test {n-len(train)-len(val)})")

    # ---- merge with the base manifest -> working files ----
    import pandas as pd

    base = pd.read_csv(data_root / "manifest_all.csv")[cols]
    poly = pd.read_csv(poly_csv)[cols]
    overlap = set(base["finger_id"].astype(str)) & set(poly["finger_id"].astype(str))
    if overlap:
        raise SystemExit(f"finger_id collision base/PolyU: {list(overlap)[:5]}")
    merged = pd.concat([base, poly], ignore_index=True)
    merged.to_csv(data_root / "manifest_all_polyu.csv", index=False)
    merged[merged["split"].isin(["train", "fvc_train"])].to_csv(data_root / "manifest_train_polyu.csv", index=False)
    merged[merged["split"] == "val"].to_csv(data_root / "manifest_val_polyu.csv", index=False)

    test = merged[merged["split"] == "test"]
    print(f"manifest_all_polyu.csv: {len(merged)} rows | test sensors: {sorted(test['sensor'].astype(str).unique())}")
    print("Saved manifest_all_polyu.csv, manifest_train_polyu.csv, manifest_val_polyu.csv")


if __name__ == "__main__":
    main()
