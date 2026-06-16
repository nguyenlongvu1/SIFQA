#!/usr/bin/env python3
"""
make_flare_qmat.py
==================
Build the SIFQ teacher targets q_mat(x) from the FLaRE matcher (Yu-Yy/FLARE).

FLaRE is NOT a flat-embedding-cosine matcher: it outputs a DENSE descriptor
(256 cells x 12 dims) + a foreground mask, matched by a masked dense-correlation
(`calculate_score`). So we cannot use the cosine-to-centroid NPZ path. Instead we
use FLaRE's OWN score on genuine pairs:

    q_mat(x) = sigmoid( per-sensor-zscore( mean_{x' same finger, x'!=x} FLaRE_score(x, x') ) )

- mean-genuine = "how well does this capture match the OTHER captures of its finger"
  = the matcher's verdict on the single-image's utility (the L_mat teacher signal).
- per-sensor z-score removes the "this sensor scores higher" OFFSET that lives in the
  label and that GRL cannot reach (deep matchers leak some sensor signature).

Pipeline (all driven from here):
  1. symlink the manifest-split images into <work>/image/query as 000000.png ...
  2. run FLaRE  extract_VotingPose.py  then  extract_FDD.py -p VotingPose
  3. read the per-image .pkl descriptors, compute per-finger mean-genuine
  4. per-sensor z-score -> sigmoid -> q_mat, write CSV (path,q_mat,...)

Output CSV columns: path,q_mat,finger,sensor,mean_genuine,n_mates
Feed it to training with:  train_sifq.py --q-mat-csv <out.csv> --lambda-mat 10 ...

Run from the SIFQ repo root so the manifest's relative paths resolve. The work dir
defaults to ~ (WSL native fs) so symlinks work even though images live on a mounted Windows drive.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def zscore_sigmoid_per_group(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Per-group z-score then sigmoid. NaN stays NaN. Degenerate group -> 0.5."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    for g in np.unique(groups[finite]):
        m = finite & (groups == g)
        v = values[m]
        mu, sigma = float(v.mean()), float(v.std())
        if sigma < 1e-6:
            out[m] = 0.5
        else:
            out[m] = 1.0 / (1.0 + np.exp(-(v - mu) / sigma))
    return out


def normalize_mean_genuine(values: np.ndarray, sensors: np.ndarray,
                           mode: str = "blend", alpha: float = 0.5) -> np.ndarray:
    """Raw FLaRE mean-genuine -> q_mat in [0,1]. NaN stays NaN.

    mode:
      "sensor" — per-sensor z-score (v1/v2/v3). Removes the 'this sensor scores higher'
                 OFFSET, but ALSO ERASES a whole sensor being globally bad: SD302 sensor H
                 (FLaRE raw mean-genuine 0.003, both Bozorth & VeriFinger fail) becomes
                 q_mat 0.5 -> SIFQ never learns H is low -> loses ERC (VeriFinger diagnosis).
      "global" — single z-score over all SD302 sensors -> keeps 'H globally bad' but re-adds
                 cosmetic sensor offset.
      "blend"  — alpha*global + (1-alpha)*per-sensor on SD302 sensors; FVC (sensor 'DB*')
                 stays per-sensor (FVC is the CLEAN degradation base, must NOT be dragged
                 down by the SD302 global). Keeps genuine per-sensor utility (H low) AND most
                 cosmetic invariance. RECOMMENDED.
    """
    values = np.asarray(values, dtype=np.float64)
    sensors = np.asarray(sensors).astype(str)
    finite = np.isfinite(values)
    if mode == "sensor":
        return zscore_sigmoid_per_group(values, sensors)

    is_fvc = np.char.startswith(sensors, "DB")
    # per-sensor z (for FVC under any mode, and the per-sensor half of blend)
    zs = np.full(values.shape, np.nan)
    for g in np.unique(sensors[finite]):
        m = finite & (sensors == g)
        v = values[m]
        sd = float(v.std())
        zs[m] = 0.0 if sd < 1e-6 else (v - float(v.mean())) / sd
    # global z over SD302 sensors only (FVC excluded from the reference)
    ref = finite & ~is_fvc
    if ref.sum() >= 2:
        gmu, gsd = float(values[ref].mean()), float(values[ref].std())
    else:
        gmu, gsd = float(values[finite].mean()), float(values[finite].std())
    zg = (values - gmu) / (gsd if gsd > 1e-6 else 1.0)

    if mode == "global":
        z = np.where(is_fvc, zs, zg)
    elif mode == "blend":
        a = float(alpha)
        z = np.where(is_fvc, zs, a * zg + (1.0 - a) * zs)
    else:
        raise ValueError("mat-norm must be sensor|global|blend")
    out = np.full(values.shape, np.nan)
    out[finite] = 1.0 / (1.0 + np.exp(-z[finite]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FLaRE mean-genuine q_mat teacher targets.")
    ap.add_argument("--manifest", default="data/manifest_all.csv")
    ap.add_argument("--split", default="train", help="Manifest split to score (train|val|test|all).")
    ap.add_argument("--flare-dir", default="external/FLARE", help="Yu-Yy/FLARE repo (has extract_FDD.py).")
    ap.add_argument("--work", default="", help="Work dir for the FLaRE layout (default: ~/flare_qmat_<split>).")
    ap.add_argument("--out-csv", default="SFIQ-2/outputs/flare_qmat_train.csv")
    ap.add_argument("--python", default=sys.executable, help="Python interpreter to run FLaRE scripts.")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--pose", default="VotingPose", choices=["VotingPose", "RegressionPose"])
    ap.add_argument("--skip-extract", action="store_true", help="Reuse existing .pkl (skip pose+FDD).")
    ap.add_argument("--skip-pose", action="store_true",
                    help="Reuse existing symlinks + VotingPose results and run ONLY extract_FDD "
                         "(resume past a dependency error without redoing the ~2min pose pass).")
    ap.add_argument("--mat-norm", default="blend", choices=["sensor", "global", "blend"],
                    help="q_mat normalization. 'blend' (default) fixes the per-sensor erasure of a "
                         "globally-bad sensor (e.g. SD302 H); 'sensor' = old v1/v2/v3 behaviour.")
    ap.add_argument("--blend-alpha", type=float, default=0.5,
                    help="Global weight in blend (0 = pure per-sensor, 1 = pure global).")
    ap.add_argument("--renorm-from", default="",
                    help="RE-NORMALIZE mode: read an existing q_mat CSV (with raw 'mean_genuine' + "
                         "'sensor'), recompute 'q_mat' with --mat-norm/--blend-alpha, write to "
                         "--out-csv, and EXIT — skips the whole FLaRE pipeline (no re-extraction). "
                         "Use to switch v3's per-sensor q_mat to the blend fix without re-running FLaRE.")
    args = ap.parse_args()

    # --- RE-NORMALIZE mode: reuse the already-extracted raw mean_genuine, skip FLaRE ---
    if str(args.renorm_from):
        d = pd.read_csv(args.renorm_from)
        for c in ("mean_genuine", "sensor"):
            if c not in d.columns:
                raise SystemExit(f"--renorm-from CSV missing column '{c}'")
        d["q_mat"] = normalize_mean_genuine(
            d["mean_genuine"].to_numpy(float), d["sensor"].astype(str).to_numpy(),
            mode=str(args.mat_norm), alpha=float(args.blend_alpha),
        )
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(out, index=False)
        g = d.groupby("sensor")["q_mat"].mean().round(3)
        print("[renorm] per-sensor q_mat mean (a globally-bad sensor, e.g. SD302 H, should be LOW):")
        print("  ", g.to_dict())
        print(f"[renorm] wrote {out}  ({len(d)} rows, mat-norm={args.mat_norm}, alpha={args.blend_alpha})")
        return

    repo_root = Path.cwd()
    flare_dir = Path(args.flare_dir).resolve()
    if not (flare_dir / "extract_FDD.py").exists():
        raise SystemExit(f"FLaRE repo not found at {flare_dir} (need extract_FDD.py).")

    df = pd.read_csv(args.manifest)
    for c in ("path", "sensor", "finger_id"):
        if c not in df.columns:
            raise SystemExit(f"Manifest must have column '{c}'.")
    if str(args.split).lower() != "all":
        df = df[df["split"].astype(str) == str(args.split)].copy()
    df = df.reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"No rows for split={args.split}.")

    # Finger identity = finger_id + roll (the matching unit; genuine = same finger).
    if "roll" in df.columns:
        df["finger"] = df["finger_id"].astype(str) + "_" + df["roll"].astype(str)
    else:
        df["finger"] = df["finger_id"].astype(str)

    work = Path(args.work) if args.work else (Path.home() / f"flare_qmat_{args.split}")
    img_q = work / "image" / "query"
    img_g = work / "image" / "gallery"
    feat_q = work / f"FDD_feat_{args.pose}" / "query"

    # idx-named symlinks avoid basename collisions across datasets/sensors.
    df["key"] = [f"{i:06d}" for i in range(len(df))]
    df["abspath"] = df["path"].astype(str).str.replace("\\", "/", regex=False).map(
        lambda p: p if os.path.isabs(p) else str((repo_root / p).resolve())
    )

    if not args.skip_extract:
        if not args.skip_pose:
            for d in (img_q, img_g):
                d.mkdir(parents=True, exist_ok=True)
                for f in d.glob("*"):
                    f.unlink()
            n_link = 0
            for _, r in df.iterrows():
                src = r["abspath"]
                if not os.path.exists(src):
                    continue
                os.symlink(src, img_q / f"{r['key']}.png")
                n_link += 1
            # gallery just needs >=1 file for the loader
            for r in df.head(3).itertuples():
                os.symlink(r.abspath, img_g / f"{r.key}.png")
            print(f"[prep] symlinked {n_link}/{len(df)} images -> {img_q}")
        else:
            print(f"[prep] --skip-pose: reusing existing symlinks + pose under {work}")

        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
        scripts = [("extract_FDD.py", ["-p", args.pose])]
        if not args.skip_pose:
            scripts = [("extract_VotingPose.py", [])] + scripts
        for script, extra in scripts:
            cmd = [args.python, script, "-f", str(work), "-g", str(args.gpu)] + extra
            print(f"[flare] {' '.join(cmd)}")
            subprocess.run(cmd, cwd=str(flare_dir), env=env, check=True)

    # ---- read descriptors ----
    import pickle
    sys.path.insert(0, str(flare_dir))
    from extract_FDD import calculate_score  # noqa: E402

    feats, masks, keep = [], [], []
    for i, r in df.iterrows():
        pkl = feat_q / f"{r['key']}.pkl"
        if not pkl.exists():
            continue
        d = pickle.load(open(pkl, "rb"))
        feats.append(d["feature"]); masks.append(d["mask"]); keep.append(i)
    if not feats:
        raise SystemExit(f"No descriptors found under {feat_q}. Did extraction run?")
    F = np.stack(feats); M = np.stack(masks)
    sub = df.iloc[keep].reset_index(drop=True)
    print(f"[score] descriptors for {len(sub)}/{len(df)} images")

    # ---- per-finger mean-genuine (only within-finger blocks; cheap & memory-safe) ----
    mean_gen = np.full(len(sub), np.nan, dtype=np.float64)
    n_mates = np.zeros(len(sub), dtype=int)
    fingers = sub["finger"].to_numpy()
    for fg in np.unique(fingers):
        idx = np.where(fingers == fg)[0]
        if idx.size < 2:
            continue  # singleton finger -> no genuine mate
        S = calculate_score(F[idx], F[idx], M[idx], M[idx], ndim_feat=12, binary=False)
        np.fill_diagonal(S, np.nan)
        mean_gen[idx] = np.nanmean(S, axis=1)
        n_mates[idx] = idx.size - 1

    # ---- normalize raw mean-genuine -> q_mat (default: blend, fixes globally-bad sensor) ----
    q_mat = normalize_mean_genuine(mean_gen, sub["sensor"].astype(str).to_numpy(),
                                   mode=str(args.mat_norm), alpha=float(args.blend_alpha))

    out = pd.DataFrame({
        "path": sub["path"].values,           # keep manifest-relative path for matching in train
        "q_mat": q_mat,
        "finger": sub["finger"].values,
        "sensor": sub["sensor"].astype(str).values,
        "mean_genuine": mean_gen,
        "n_mates": n_mates,
    })
    out = out[np.isfinite(out["q_mat"])].reset_index(drop=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[done] wrote {len(out)} q_mat targets -> {args.out_csv}")
    print(f"       q_mat mean={out.q_mat.mean():.3f} std={out.q_mat.std():.3f} "
          f"min={out.q_mat.min():.3f} max={out.q_mat.max():.3f}")
    # sensor offset sanity: per-sensor mean of RAW mean-genuine (high spread = bias the
    # z-score just removed); per-sensor mean of q_mat should now be ~0.5 everywhere.
    g = out.groupby("sensor")
    print("[check] per-sensor q_mat mean (should be ~0.5 after norm):")
    print(g["q_mat"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
