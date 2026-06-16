from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from torchvision import transforms

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from sifq.degradations import CONCEPTS, DEG_TO_TARGETS, apply_degradation_pil
from sifq.model import SIFQModel

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def resolve_device(s: str) -> str:
    if s == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return s


def iter_images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def infer_sensor_from_path(path: Path) -> str:
    parts = path.parts
    for i, part in enumerate(parts):
        if part.lower() == "challengers" and i + 1 < len(parts):
            return parts[i + 1]
    return ""


def normalize_sample_key(value: object) -> str:
    # Anchor on "challengers/" first: it is the stable common segment shared by
    # both data/SD302a/... (quality CSV) and archives/SD302a/... (matcher CSV).
    # Anchoring on "archives/" first would make the two CSVs produce different
    # keys (one starting "challengers/", the other "archives/") and never merge.
    text = str(value).replace("\\", "/").lower()
    for marker in ("challengers/", "archives/"):
        idx = text.find(marker)
        if idx >= 0:
            return text[idx:]
    return text


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, eps, None)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = normalize_rows(a.astype(np.float32, copy=False))
    b_n = normalize_rows(b.astype(np.float32, copy=False))
    return a_n @ b_n.T


def load_quality_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "identity" not in df.columns:
        raise SystemExit("Input CSV must contain an identity column")
    if "q_hat" not in df.columns:
        raise SystemExit("Input CSV must contain q_hat")
    if "path" not in df.columns:
        raise SystemExit("Input CSV must contain a path column")
    df["identity"] = df["identity"].astype(str)
    df["path"] = df["path"].astype(str)
    df["sample_key"] = df["path"].map(normalize_sample_key)
    df["q_hat"] = pd.to_numeric(df["q_hat"], errors="coerce")
    df = df[np.isfinite(df["q_hat"].to_numpy(dtype=np.float32))].copy()
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    feat_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    feat_cols = sorted(feat_cols, key=lambda c: int(c[1:]))
    if not feat_cols:
        raise SystemExit("Input CSV must contain embedding columns f0..fN")
    return feat_cols


def load_matcher_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "path" not in df.columns or "identity" not in df.columns:
        raise SystemExit(f"Matcher CSV must contain path and identity columns: {csv_path}")
    feat_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    if not feat_cols:
        raise SystemExit(f"Matcher CSV must contain embedding columns f0..fN: {csv_path}")
    df["path"] = df["path"].astype(str)
    df["identity"] = df["identity"].astype(str)
    df["sample_key"] = df["path"].map(normalize_sample_key)
    for col in feat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[np.isfinite(df[feat_cols].to_numpy(dtype=np.float32)).all(axis=1)].copy()
    return df


def get_concept_columns(df: pd.DataFrame) -> List[str]:
    concept_cols = list(CONCEPTS)
    missing = [c for c in concept_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Input CSV missing concept columns: {missing}")
    return concept_cols


def compute_centroids(df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[List[str], np.ndarray]:
    labels = sorted(df["identity"].astype(str).unique().tolist())
    x = df.loc[:, feature_cols].to_numpy(dtype=np.float32)
    x = normalize_rows(x)
    centroids: List[np.ndarray] = []
    kept_labels: List[str] = []
    for label in labels:
        mask = df["identity"].astype(str).to_numpy() == label
        if not np.any(mask):
            continue
        c = x[mask].mean(axis=0, keepdims=False)
        n = np.linalg.norm(c)
        if n > 1e-12:
            c = c / n
        centroids.append(c.astype(np.float32, copy=False))
        kept_labels.append(label)
    if not centroids:
        raise SystemExit("No centroids could be computed")
    return kept_labels, np.stack(centroids, axis=0)


def compute_erc_curve(
    quality_df: pd.DataFrame,
    quality_col: str,
    matcher_df: pd.DataFrame,
    target_fmr: float = 1e-2,
    rejection_grid: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Standard Error-vs-Reject Characteristic.

    Genuine/impostor scores and the operating threshold are computed ONCE on the full
    matched set (a fixed gallery + fixed threshold); then we progressively reject the
    lowest-quality probes and recompute FNMR on the survivors. A useful quality score
    makes FNMR drop as rejection rises.

    Fixes vs the old harness: (1) genuine is leave-one-out (probe excluded from its own
    centroid), (2) centroids/threshold are not recomputed per rejection level (which made
    the curve insensitive/inverted), (3) default FMR=1e-2 — at 1e-4 the threshold sits at
    the 99.99th impostor percentile (~1.0) on this small cross-sensor test, pinning FNMR≈1.
    """
    if rejection_grid is None:
        rejection_grid = np.linspace(0.0, 0.5, 11)
    matcher_feat_cols = get_feature_columns(matcher_df)
    work = quality_df[["path", "sample_key", "identity", quality_col]].merge(
        matcher_df[["sample_key", "identity", *matcher_feat_cols]],
        on="sample_key",
        how="inner",
        suffixes=("", "_matcher"),
    )
    if work.empty:
        raise SystemExit(f"No overlapping samples between quality data and matcher data for {quality_col}")
    work = work[np.isfinite(work[quality_col].to_numpy(dtype=np.float32))].copy()

    # Matching identity must be per-FINGER, not per-subject (subject has 10 unrelated
    # fingers roll_01..roll_10). Append the finger position parsed from the filename.
    import re as _re

    def _frgp(p: str) -> str:
        stem = Path(str(p)).stem
        m = _re.search(r"roll[_-]?(\d+)", stem, _re.IGNORECASE)
        if not m:
            m = _re.search(r"(\d+)\s*$", stem)
        return m.group(1) if m else "0"

    work["identity"] = work["identity"].astype(str) + "_" + work["path"].map(_frgp)
    # sort by quality DESC so iloc[:keep_n] keeps the highest-quality probes
    work = work.sort_values(quality_col, ascending=False).reset_index(drop=True)

    labels = sorted(work["identity"].astype(str).unique().tolist())
    lab2idx = {lab: i for i, lab in enumerate(labels)}
    y = work["identity"].astype(str).map(lab2idx).to_numpy()
    X = normalize_rows(work.loc[:, matcher_feat_cols].to_numpy(dtype=np.float64))
    N, D = X.shape
    n_lab = len(labels)

    # per-identity sum/count -> fixed gallery centroids (mean over ALL samples)
    sum_vec = np.zeros((n_lab, D), dtype=np.float64)
    cnt = np.zeros(n_lab, dtype=np.int64)
    np.add.at(sum_vec, y, X)
    np.add.at(cnt, y, 1)
    centroids = sum_vec / np.clip(cnt[:, None], 1, None)
    centroids = centroids / np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12, None)

    # genuine = cos(probe, its OWN centroid with the probe left out); needs >=2 samples
    cnt_y = cnt[y]
    loo = (sum_vec[y] - X) / np.clip((cnt_y - 1)[:, None], 1, None)
    loo = loo / np.clip(np.linalg.norm(loo, axis=1, keepdims=True), 1e-12, None)
    genuine = np.sum(X * loo, axis=1)
    genuine[cnt_y < 2] = np.nan

    # impostor = cos(probe, every OTHER finger centroid); threshold fixed once at target FMR
    sim = X @ centroids.T                      # (N, n_lab)
    sim[np.arange(N), y] = -np.inf             # mask own identity
    impostor = sim[np.isfinite(sim)]
    if impostor.size == 0:
        raise SystemExit("No impostor pairs (need >=2 fingers).")
    threshold = float(np.quantile(impostor, 1.0 - float(target_fmr)))
    fmr_actual = float(np.mean(impostor >= threshold))

    rows: List[Dict[str, float]] = []
    for rej in rejection_grid:
        keep_n = max(2, int(math.ceil((1.0 - float(rej)) * N)))
        g = genuine[:keep_n]
        g = g[np.isfinite(g)]
        if g.size == 0:
            continue
        rows.append(
            {
                "rejection_ratio": float(rej),
                "kept_n": int(keep_n),
                "n_genuine": int(g.size),
                "threshold": threshold,
                "fnmr": float(np.mean(g < threshold)),
                "fmr": fmr_actual,
            }
        )

    if not rows:
        raise SystemExit(f"Could not compute ERC for {quality_col}")

    out = pd.DataFrame.from_records(rows).sort_values("rejection_ratio").reset_index(drop=True)
    fnmr_arr = out["fnmr"].to_numpy(dtype=np.float64)
    rej_arr = out["rejection_ratio"].to_numpy(dtype=np.float64)
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    out["auc_erc"] = float(trapezoid(fnmr_arr, rej_arr))
    return out


def plot_erc(curves: Dict[str, pd.DataFrame], out_png: Path, title: str) -> None:
    plt.figure(figsize=(8, 5.5))
    for name, curve in curves.items():
        plt.plot(curve["rejection_ratio"], curve["fnmr"], marker="o", linewidth=2, label=f"{name} (AUC={curve['auc_erc'].iloc[0]:.4f})")
    plt.xlabel("Rejection ratio")
    plt.ylabel("FNMR")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def print_curve_report(name: str, curve: pd.DataFrame) -> None:
    auc = float(curve["auc_erc"].iloc[0])
    best = curve.sort_values("fnmr").iloc[0]
    print(f"[{name}] AUC_ERC={auc:.6f}")
    print(
        f"  best point: rejection={best['rejection_ratio']:.2f}, FNMR={best['fnmr']:.4f}, FMR={best['fmr']:.6f}, threshold={best['threshold']:.4f}"
    )


@torch.no_grad()
def load_model(ckpt_path: Path, device: str) -> Tuple[SIFQModel, transforms.Compose]:
    ck = torch.load(ckpt_path, map_location="cpu")
    sensor2id = ck.get("sensor2id")
    if not isinstance(sensor2id, dict):
        sensor2id = {}
    id2sensor = {int(v): str(k) for k, v in sensor2id.items()} if sensor2id else {}

    sd = ck.get("model", {})
    model = SIFQModel(backbone="mobilenet_v2", n_sensors=max(1, len(id2sensor) or 8)).to(device)
    model.load_state_dict(sd, strict=False)
    model.sensor2id = sensor2id
    model.eval()

    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return model, tf


@torch.no_grad()
def compute_track4_grounding(
    model: SIFQModel,
    tf: transforms.Compose,
    image_root: Path,
    output_dir: Path,
    max_images: int,
    seed: int,
) -> pd.DataFrame:
    image_paths = sorted(iter_images(image_root))
    if not image_paths:
        raise SystemExit(f"No images found under {image_root}")
    if max_images > 0 and len(image_paths) > max_images:
        rng = np.random.default_rng(seed)
        image_paths = [image_paths[i] for i in rng.choice(len(image_paths), size=max_images, replace=False)]

    degrade_types = list(DEG_TO_TARGETS.keys())
    concept_cols = list(CONCEPTS)

    rows: List[Dict[str, object]] = []
    for deg_i, deg in enumerate(degrade_types):
        for img_i, image_path in enumerate(image_paths):
            img = Image.open(image_path).convert("RGB")
            for level in range(0, 4):
                local_seed = seed + deg_i * 10000 + img_i * 101 + level
                random.seed(local_seed)
                np.random.seed(local_seed % (2**32 - 1))
                torch.manual_seed(local_seed)
                degraded = apply_degradation_pil(img, deg, level)
                x = tf(degraded).unsqueeze(0).to(next(model.parameters()).device)
                sensor_name = infer_sensor_from_path(image_path)
                sensor_id = int(getattr(model, "sensor2id", {}).get(sensor_name, 0))
                out = model(x, grl_lambda=0.0, sensor_ids=torch.tensor([sensor_id], device=x.device))
                concept_vals = out.concepts[0].detach().cpu().numpy().astype(np.float32)
                row = {
                    "degradation": deg,
                    "image": str(image_path),
                    "level": int(level),
                    "q_hat": 100.0 * float(out.q[0].detach().cpu().item()),  # 0..100 scale
                }
                for c_idx, c_name in enumerate(concept_cols):
                    row[c_name] = float(concept_vals[c_idx])
                rows.append(row)

    df = pd.DataFrame.from_records(rows)
    df.to_csv(output_dir / "track4_degradation_samples.csv", index=False)

    matrix_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    # Per-(degradation, concept) mean at level 0 (clean) vs max level, to measure
    # response MAGNITUDE. Spearman rho only checks monotonicity, so a near-constant
    # concept with tiny drift can score |rho|≈1 yet be effectively dead. A grounded
    # concept needs both correct-sign |rho| AND a meaningful range.
    max_level = int(df["level"].max())
    for deg in degrade_types:
        subset = df[df["degradation"] == deg].copy()
        targets = {name for name, _direction in DEG_TO_TARGETS.get(deg, [])}
        clean_means = subset[subset["level"] == 0]
        max_means = subset[subset["level"] == max_level]
        target_rhos = []
        target_ranges = []
        non_target_rhos = []
        row = {"degradation": deg}
        for c_name in concept_cols:
            rho, p = spearmanr(subset["level"].to_numpy(dtype=np.float32), subset[c_name].to_numpy(dtype=np.float32))
            rho = float(rho) if np.isfinite(rho) else float("nan")
            # signed range = mean(concept@max_level) - mean(concept@clean)
            rng = float(max_means[c_name].mean() - clean_means[c_name].mean()) if len(clean_means) and len(max_means) else float("nan")
            row[c_name] = rho
            matrix_rows.append({"degradation": deg, "concept": c_name, "spearman_rho": rho,
                                "range_max_minus_clean": rng,
                                "p_value": float(p) if np.isfinite(p) else float("nan")})
            if c_name in targets:
                target_rhos.append(rho)
                target_ranges.append(abs(rng) if np.isfinite(rng) else float("nan"))
            else:
                non_target_rhos.append(rho)

        # Does the final score Q itself deduct as the image degrades? This is the most
        # direct test of the "deduction mechanism": Q should drop monotonically with level.
        q_rho, _ = spearmanr(subset["level"].to_numpy(dtype=np.float32), subset["q_hat"].to_numpy(dtype=np.float32))
        q_rho = float(q_rho) if np.isfinite(q_rho) else float("nan")
        q_drop = float(clean_means["q_hat"].mean() - max_means["q_hat"].mean()) if len(clean_means) and len(max_means) else float("nan")
        summary_rows.append(
            {
                "degradation": deg,
                "target_concepts": ",".join(sorted(targets)),
                "mean_target_rho": float(np.nanmean(target_rhos)) if target_rhos else float("nan"),
                "mean_target_range": float(np.nanmean(target_ranges)) if target_ranges else float("nan"),
                "mean_non_target_rho": float(np.nanmean(non_target_rhos)) if non_target_rhos else float("nan"),
                "q_rho_vs_level": q_rho,          # want strongly NEGATIVE (Q drops as deg rises)
                "q_drop_clean_to_max": q_drop,    # points of Q lost from clean -> max degrade (0..100)
                "grounded": bool(
                    target_rhos and target_ranges
                    and np.nanmean(np.abs(target_rhos)) > 0.5
                    and np.nanmean(target_ranges) > 0.2
                ),
            }
        )

    matrix_df = pd.DataFrame.from_records(matrix_rows)
    summary_df = pd.DataFrame.from_records(summary_rows)
    pivot = matrix_df.pivot(index="degradation", columns="concept", values="spearman_rho").reindex(index=degrade_types, columns=concept_cols)
    pivot.to_csv(output_dir / "track4_crosstalk_matrix.csv")
    summary_df.to_csv(output_dir / "track4_summary.csv", index=False)
    matrix_df.to_csv(output_dir / "track4_matrix_long.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    arr = pivot.to_numpy(dtype=np.float32)
    vmax = float(np.nanmax(np.abs(arr))) if np.isfinite(np.nanmax(np.abs(arr))) else 1.0
    im = ax.imshow(arr, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(concept_cols)))
    ax.set_yticks(np.arange(len(degrade_types)))
    ax.set_xticklabels(concept_cols, rotation=30, ha="right")
    ax.set_yticklabels(degrade_types)
    ax.set_title("Track 4: Concept grounding cross-talk matrix (Spearman rho)")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            color = "white" if np.isfinite(vmax) and abs(v) > 0.5 * vmax else "black"
            ax.text(j, i, f"{v:.2f}" if np.isfinite(v) else "nan", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(output_dir / "track4_crosstalk_matrix.png", dpi=200)
    plt.close(fig)

    return summary_df
