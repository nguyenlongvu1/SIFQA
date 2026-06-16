#!/usr/bin/env python3
"""Track 2: sensor-invariance evaluation for SIFQ embeddings."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROLL_RE = re.compile(r"(?:^|[_-])roll[_-](\d+)", re.IGNORECASE)


def infer_roll(value: object) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value)
    stem = Path(text).stem
    match = ROLL_RE.search(stem)
    if match:
        return int(match.group(1))
    return None


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def ks_2samp_statistic(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    x = np.sort(np.asarray(x, dtype=np.float32))
    y = np.sort(np.asarray(y, dtype=np.float32))
    support = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, support, side="right") / float(x.size)
    cdf_y = np.searchsorted(y, support, side="right") / float(y.size)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def stats_for(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "q05": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "q95": float("nan"),
            "max": float("nan"),
            "iqr": float("nan"),
        }

    q05, q25, q50, q75, q95 = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(q50),
        "min": float(np.min(values)),
        "q05": float(q05),
        "q25": float(q25),
        "q75": float(q75),
        "q95": float(q95),
        "max": float(np.max(values)),
        "iqr": float(q75 - q25),
    }


def prepare_quality_frame(df: pd.DataFrame, quality_col: str) -> pd.DataFrame:
    if "sensor" not in df.columns or "identity" not in df.columns:
        raise SystemExit("CSV must contain at least sensor and identity columns")
    if quality_col not in df.columns:
        raise SystemExit(f"CSV must contain quality column: {quality_col}")

    work = df.copy()
    work["sensor"] = work["sensor"].astype(str)
    work["identity"] = work["identity"].astype(str)
    work[quality_col] = pd.to_numeric(work[quality_col], errors="coerce")
    work = work[np.isfinite(work[quality_col].to_numpy(dtype=np.float32))].copy()

    if "roll" in work.columns:
        roll = pd.to_numeric(work["roll"], errors="coerce")
        work["roll"] = roll
    elif "path" in work.columns:
        work["roll"] = work["path"].map(infer_roll)
    elif "key" in work.columns:
        work["roll"] = work["key"].map(infer_roll)
    else:
        work["roll"] = np.nan

    work = work[pd.notna(work["roll"])].copy()
    work["roll"] = work["roll"].astype(int)

    quality_frame = (
        work.groupby(["sensor", "identity", "roll"], as_index=False)[quality_col]
        .mean()
        .sort_values(["sensor", "identity", "roll"])
        .reset_index(drop=True)
    )
    return quality_frame


def build_paired_quality_rows(qdf: pd.DataFrame, quality_col: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (identity, roll), group in qdf.groupby(["identity", "roll"]):
        by_sensor = {str(sensor): float(vals[quality_col].iloc[0]) for sensor, vals in group.groupby("sensor")}
        sensors = sorted(by_sensor.keys())
        for sensor_a, sensor_b in combinations(sensors, 2):
            q_a = float(by_sensor[sensor_a])
            q_b = float(by_sensor[sensor_b])
            rows.append(
                {
                    "identity": identity,
                    "roll": int(roll),
                    "sensor_a": sensor_a,
                    "sensor_b": sensor_b,
                    "q_a": q_a,
                    "q_b": q_b,
                    "signed_diff": q_a - q_b,
                    "abs_diff": abs(q_a - q_b),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["identity", "roll", "sensor_a", "sensor_b", "q_a", "q_b", "signed_diff", "abs_diff"])
    return pd.DataFrame.from_records(rows)


def sensor_quality_summary(qdf: pd.DataFrame, quality_col: str) -> pd.DataFrame:
    rows = []
    for sensor, group in qdf.groupby("sensor"):
        values = group[quality_col].to_numpy(dtype=np.float32)
        st = stats_for(values)
        st.update({"sensor": str(sensor)})
        rows.append(st)
    return pd.DataFrame.from_records(rows).sort_values("sensor").reset_index(drop=True)


def quality_ks_matrix(qdf: pd.DataFrame, quality_col: str) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
    sensors = sorted(qdf["sensor"].astype(str).unique().tolist())
    matrix = np.zeros((len(sensors), len(sensors)), dtype=np.float32)
    rows: List[Dict[str, object]] = []
    for i, sensor_a in enumerate(sensors):
        qa = qdf.loc[qdf["sensor"] == sensor_a, quality_col].to_numpy(dtype=np.float32)
        for j, sensor_b in enumerate(sensors):
            qb = qdf.loc[qdf["sensor"] == sensor_b, quality_col].to_numpy(dtype=np.float32)
            ks = 0.0 if i == j else ks_2samp_statistic(qa, qb)
            matrix[i, j] = ks
            rows.append(
                {
                    "sensor_a": sensor_a,
                    "sensor_b": sensor_b,
                    "n_a": int(qa.size),
                    "n_b": int(qb.size),
                    "ks_stat": float(ks),
                }
            )
    return sensors, matrix, pd.DataFrame.from_records(rows)


def pair_quality_stats(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (sensor_a, sensor_b), group in pair_df.groupby(["sensor_a", "sensor_b"]):
        signed = group["signed_diff"].to_numpy(dtype=np.float32)
        absdiff = group["abs_diff"].to_numpy(dtype=np.float32)
        pearson_r = safe_pearson(group["q_a"].to_numpy(dtype=np.float32), group["q_b"].to_numpy(dtype=np.float32))
        st_signed = stats_for(signed)
        st_abs = stats_for(absdiff)
        rows.append(
            {
                "sensor_a": str(sensor_a),
                "sensor_b": str(sensor_b),
                "n_pairs": int(len(group)),
                "mean_signed": st_signed["mean"],
                "std_signed": st_signed["std"],
                "mean_abs": st_abs["mean"],
                "median_abs": st_abs["median"],
                "p95_abs": st_abs["q95"],
                "pearson_r": pearson_r,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["sensor_a", "sensor_b", "n_pairs", "mean_signed", "std_signed", "mean_abs", "median_abs", "p95_abs", "pearson_r"])
    return pd.DataFrame.from_records(rows).sort_values(["sensor_a", "sensor_b"]).reset_index(drop=True)


def plot_quality_histograms(qdf: pd.DataFrame, quality_col: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    for sensor in sorted(qdf["sensor"].astype(str).unique().tolist()):
        values = qdf.loc[qdf["sensor"] == sensor, quality_col].to_numpy(dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        # Degenerate (collapsed / near-constant Q) breaks np.histogram's 40-bin edges
        # ("Too many bins for data range"). Draw a point-mass line instead so a
        # COLLAPSED model stays diagnosable (KS=0 across sensors) rather than crashing.
        if np.unique(values).size < 2 or float(np.ptp(values)) < 1e-3:
            plt.axvline(float(values[0]), alpha=0.35, label=str(sensor))
            continue
        plt.hist(values, bins=40, alpha=0.35, density=True, label=str(sensor))
    plt.title(f"{quality_col}: sensor-wise distribution")
    plt.xlabel(quality_col)
    plt.ylabel("Density")
    plt.legend(ncol=8, fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_ks_heatmap(sensors: List[str], matrix: np.ndarray, output_path: Path) -> None:
    vmax = max(1e-6, float(np.nanmax(matrix)))
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, cmap="OrRd", vmin=0.0, vmax=vmax)
    ax.set_xticks(np.arange(len(sensors)))
    ax.set_yticks(np.arange(len(sensors)))
    ax.set_xticklabels(sensors)
    ax.set_yticklabels(sensors)
    ax.set_xlabel("Sensor")
    ax.set_ylabel("Sensor")
    ax.set_title("SIFQ Pairwise KS Statistic")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            color = "white" if value > 0.55 * vmax else "black"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="KS statistic")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_metric_heatmap(
    sensors: List[str],
    pair_stats_df: pd.DataFrame,
    metric_col: str,
    title: str,
    output_path: Path,
    fmt: str = ".1f",
) -> None:
    lookup = {(str(row.sensor_a), str(row.sensor_b)): float(getattr(row, metric_col)) for row in pair_stats_df.itertuples(index=False)}
    matrix = np.zeros((len(sensors), len(sensors)), dtype=np.float32)
    for i, sensor_a in enumerate(sensors):
        for j, sensor_b in enumerate(sensors):
            if i == j:
                matrix[i, j] = 0.0
                continue
            value = lookup.get((sensor_a, sensor_b))
            if value is None:
                value = lookup.get((sensor_b, sensor_a), float("nan"))
            matrix[i, j] = float(value)

    vmax = float(np.nanmax(matrix)) if np.isfinite(np.nanmax(matrix)) else 1.0
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, cmap="OrRd", vmin=0.0, vmax=max(1e-6, vmax))
    ax.set_xticks(np.arange(len(sensors)))
    ax.set_yticks(np.arange(len(sensors)))
    ax.set_xticklabels(sensors)
    ax.set_yticklabels(sensors)
    ax.set_xlabel("Sensor")
    ax.set_ylabel("Sensor")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if np.isfinite(vmax) and value > 0.55 * vmax else "black"
            ax.text(j, i, format(value, fmt), ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean abs diff")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pair_scatter(pair_df: pd.DataFrame, output_dir: Path, max_points: int = 5000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for (sensor_a, sensor_b), group in pair_df.groupby(["sensor_a", "sensor_b"]):
        q_a = group["q_a"].to_numpy(dtype=np.float32)
        q_b = group["q_b"].to_numpy(dtype=np.float32)
        if q_a.size == 0:
            continue
        if q_a.size > max_points:
            idx = rng.choice(np.arange(q_a.size), size=max_points, replace=False)
            q_a = q_a[idx]
            q_b = q_b[idx]
        r = safe_pearson(q_a, q_b)
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(q_a, q_b, s=8, alpha=0.25)
        lo = float(min(np.min(q_a), np.min(q_b)))
        hi = float(max(np.max(q_a), np.max(q_b)))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel(f"Q sensor {sensor_a}")
        ax.set_ylabel(f"Q sensor {sensor_b}")
        ax.set_title(f"Paired quality scatter {sensor_a} vs {sensor_b} (r={r:.3f})")
        fig.tight_layout()
        fig.savefig(output_dir / f"scatter_{sensor_a}_vs_{sensor_b}.png", dpi=200)
        plt.close(fig)


def save_quality_report(df: pd.DataFrame, quality_col: str, output_dir: Path, seed: int = 0) -> Dict[str, object]:
    qdf = prepare_quality_frame(df, quality_col)
    pair_df = build_paired_quality_rows(qdf, quality_col)
    sensor_summary = sensor_quality_summary(qdf, quality_col)
    sensors, ks_matrix, ks_df = quality_ks_matrix(qdf, quality_col)
    pair_stats_df = pair_quality_stats(pair_df)

    plot_quality_histograms(qdf, quality_col, output_dir / f"{quality_col}_sensor_hist.png")
    plot_ks_heatmap(sensors, ks_matrix, output_dir / f"{quality_col}_ks_matrix.png")
    plot_pair_scatter(pair_df, output_dir / f"{quality_col}_pair_scatter", seed=seed)
    if not pair_stats_df.empty and "mean_abs" in pair_stats_df.columns:
        plot_pairwise_metric_heatmap(
            sensors,
            pair_stats_df,
            metric_col="mean_abs",
            title=f"{quality_col} Pairwise Mean Absolute Difference",
            output_path=output_dir / f"{quality_col}_pairwise_mean_abs.png",
            fmt=".1f",
        )
    if not pair_stats_df.empty and "mean_signed" in pair_stats_df.columns:
        plot_pairwise_metric_heatmap(
            sensors,
            pair_stats_df,
            metric_col="mean_signed",
            title=f"{quality_col} Pairwise Mean Signed Difference",
            output_path=output_dir / f"{quality_col}_pairwise_mean_signed.png",
            fmt=".1f",
        )

    sensor_summary.to_csv(output_dir / f"{quality_col}_sensor_summary.csv", index=False)
    ks_df.to_csv(output_dir / f"{quality_col}_ks_pairs.csv", index=False)
    pd.DataFrame(ks_matrix, index=sensors, columns=sensors).to_csv(output_dir / f"{quality_col}_ks_matrix.csv")
    pair_df.to_csv(output_dir / f"{quality_col}_pairs.csv", index=False)
    pair_stats_df.to_csv(output_dir / f"{quality_col}_pair_stats.csv", index=False)

    return {
        "quality_column": quality_col,
        "n_samples": int(len(qdf)),
        "n_pairs": int(len(pair_df)),
        "sensors": sensors,
        "sensor_summary": sensor_summary.to_dict(orient="records"),
        "ks_pairs": ks_df.to_dict(orient="records"),
        "pair_stats": pair_stats_df.to_dict(orient="records"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", default="SFIQ-2/outputs/sifq_sd302_embeddings.csv")
    ap.add_argument("--output-dir", default="SFIQ-2/code/evaluate/result/track2")
    ap.add_argument("--quality-column", default="q_hat")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    report = save_quality_report(df, quality_col=str(args.quality_column), output_dir=output_dir, seed=int(args.seed))
    (output_dir / f"{args.quality_column}_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved Track 2 report to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
