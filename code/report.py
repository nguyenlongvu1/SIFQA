#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("f")]
    if not cols:
        raise SystemExit("No embedding feature columns found (expected f0, f1, ...)")
    return cols


def stats_for(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "q01": float("nan"),
            "q05": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "q95": float("nan"),
            "q99": float("nan"),
            "max": float("nan"),
            "range": float("nan"),
            "iqr": float("nan"),
        }

    q01, q05, q25, q50, q75, q95, q99 = np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(q50),
        "min": float(np.min(values)),
        "q01": float(q01),
        "q05": float(q05),
        "q25": float(q25),
        "q75": float(q75),
        "q95": float(q95),
        "q99": float(q99),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
        "iqr": float(q75 - q25),
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def sample_pair_indices(df: pd.DataFrame, predicate, max_pairs: int, seed: int) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    pairs: List[Tuple[int, int]] = []
    seen = set()
    attempts = 0
    max_attempts = max_pairs * 50 if max_pairs > 0 else 0

    while len(pairs) < max_pairs and attempts < max_attempts:
        i, j = rng.choice(indices, size=2, replace=False)
        if i > j:
            i, j = j, i
        key = (int(i), int(j))
        attempts += 1
        if key in seen:
            continue
        if predicate(df.iloc[i], df.iloc[j]):
            pairs.append(key)
            seen.add(key)

    return pairs


def pair_statistics(embeddings: np.ndarray, df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, Dict[str, object]]:
    genuine_pairs = sample_pair_indices(
        df,
        predicate=lambda a, b: a["identity"] == b["identity"] and a["sensor"] != b["sensor"],
        max_pairs=min(20000, len(df) * 30),
        seed=seed,
    )
    impostor_pairs = sample_pair_indices(
        df,
        predicate=lambda a, b: a["identity"] != b["identity"] and a["sensor"] == b["sensor"],
        max_pairs=min(20000, len(df) * 20),
        seed=seed + 17,
    )

    rows: List[Dict[str, object]] = []
    for i, j in genuine_pairs:
        rows.append(
            {
                "pair_type": "genuine_cross_sensor",
                "i": int(i),
                "j": int(j),
                "sensor_i": df.iloc[i]["sensor"],
                "sensor_j": df.iloc[j]["sensor"],
                "identity_i": df.iloc[i]["identity"],
                "identity_j": df.iloc[j]["identity"],
                "similarity": cosine_similarity(embeddings[i], embeddings[j]),
            }
        )

    for i, j in impostor_pairs:
        rows.append(
            {
                "pair_type": "impostor_in_sensor",
                "i": int(i),
                "j": int(j),
                "sensor_i": df.iloc[i]["sensor"],
                "sensor_j": df.iloc[j]["sensor"],
                "identity_i": df.iloc[i]["identity"],
                "identity_j": df.iloc[j]["identity"],
                "similarity": cosine_similarity(embeddings[i], embeddings[j]),
            }
        )

    pair_df = pd.DataFrame.from_records(rows)
    stats_out: Dict[str, object] = {}
    for pair_type in ["genuine_cross_sensor", "impostor_in_sensor"]:
        values = pair_df.loc[pair_df["pair_type"] == pair_type, "similarity"].to_numpy(dtype=np.float32)
        stats_out[pair_type] = stats_for(values)

    genuine = pair_df.loc[pair_df["pair_type"] == "genuine_cross_sensor", "similarity"].to_numpy(dtype=np.float32)
    impostor = pair_df.loc[pair_df["pair_type"] == "impostor_in_sensor", "similarity"].to_numpy(dtype=np.float32)
    if genuine.size and impostor.size:
        stats_out["comparison"] = {
            "bias_gap_mean": float(np.mean(impostor) - np.mean(genuine)),
            "bias_gap_median": float(np.median(impostor) - np.median(genuine)),
            "separation_margin": float(np.mean(genuine) - np.mean(impostor)),
            "impostor_gt_genuine_rate": float((impostor > np.mean(genuine)).mean()),
        }
    else:
        stats_out["comparison"] = {
            "bias_gap_mean": float("nan"),
            "bias_gap_median": float("nan"),
            "separation_margin": float("nan"),
            "impostor_gt_genuine_rate": float("nan"),
        }

    return pair_df, stats_out


def train_sensor_classifier(df: pd.DataFrame, n_splits: int, seed: int) -> Dict[str, object]:
    # Lười import sklearn để ks-plot / train-log không phụ thuộc nó.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x = df[feature_columns(df)].to_numpy(dtype=np.float32)
    y = df["sensor"].astype(str).to_numpy()
    classes = sorted(np.unique(y).tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.asarray([class_to_idx[v] for v in y], dtype=np.int32)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_accuracies: List[float] = []
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                # multi_class= was removed in sklearn 1.7 (multinomial is the default
                # for multiclass now); passing it raises TypeError on sklearn>=1.7.
                LogisticRegression(max_iter=5000, class_weight="balanced"),
            ),
        ]
    )

    for train_idx, test_idx in skf.split(x, y_idx):
        clf.fit(x[train_idx], y_idx[train_idx])
        pred = clf.predict(x[test_idx])
        fold_accuracies.append(float(accuracy_score(y_idx[test_idx], pred)))
        y_true_all.extend(y_idx[test_idx].tolist())
        y_pred_all.extend(np.asarray(pred, dtype=np.int32).tolist())

    cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(len(classes))))
    return {
        "status": "ok",
        "accuracy_mean": float(np.mean(fold_accuracies)),
        "accuracy_std": float(np.std(fold_accuracies)),
        "accuracy_folds": fold_accuracies,
        "chance_level": float(1.0 / len(classes)),
        "classes": classes,
        "confusion_matrix": cm.tolist(),
    }


def save_confusion_matrix(output_dir: Path, classes: List[str], cm: np.ndarray) -> None:
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(output_dir / "sensorclf_sifq_stage4_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted sensor")
    ax.set_ylabel("True sensor")
    ax.set_title("SFIQ sensor classifier confusion matrix")

    thresh = float(cm.max()) * 0.6 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                color="white" if cm[i, j] >= thresh else "black",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "sensorclf_sifq_stage4_confusion_matrix.png", dpi=200)
    plt.close(fig)


def cmd_sensor(args: argparse.Namespace) -> None:
    embeddings_csv = Path(args.embeddings_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(embeddings_csv)
    if "sensor" not in df.columns or "identity" not in df.columns:
        raise SystemExit("Embeddings CSV must contain sensor and identity columns")

    sensor_clf = train_sensor_classifier(df, n_splits=int(args.cv), seed=int(args.seed))
    cm = np.asarray(sensor_clf["confusion_matrix"], dtype=np.int64)
    classes = list(sensor_clf["classes"])
    save_confusion_matrix(output_dir, classes, cm)

    x = df[feature_columns(df)].to_numpy(dtype=np.float32)
    embeddings = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    pair_df, pair_stats = pair_statistics(embeddings, df, seed=int(args.seed))

    summary = {
        "n_samples": int(len(df)),
        "embedding_dim": int(x.shape[1]),
        "n_sensors": int(df["sensor"].nunique()),
        "sensors": sorted(df["sensor"].astype(str).unique().tolist()),
        "cv": int(args.cv),
        "accuracy_mean": sensor_clf["accuracy_mean"],
        "accuracy_std": sensor_clf["accuracy_std"],
        "accuracy_folds": sensor_clf["accuracy_folds"],
        "chance_level": sensor_clf["chance_level"],
        "confusion_matrix": sensor_clf["confusion_matrix"],
        "pair_stats": pair_stats,
    }
    (output_dir / "sensorclf_sifq_stage4.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pair_df.to_csv(output_dir / "sensorclf_sifq_stage4_pairs.csv", index=False)

    pair_stats_out = {
        "n_samples": int(len(df)),
        "n_finger_ids": int(df["identity"].nunique()),
        "n_sensors": int(df["sensor"].nunique()),
        "genuine_cross_sensor": pair_stats["genuine_cross_sensor"],
        "impostor_in_sensor": pair_stats["impostor_in_sensor"],
        "bias_flag_impostor_gt_genuine": bool(pair_stats["comparison"]["bias_gap_mean"] > 0),
        "comparison": pair_stats["comparison"],
    }
    (output_dir / "bias_sifq_stage4.json").write_text(json.dumps(pair_stats_out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved reports to {output_dir.resolve()}")

def load_ks_table(csv_path: Path) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(csv_path)
    required = {"sensor_a", "sensor_b", "ks"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV must contain columns: {sorted(required)}; missing {sorted(missing)}")

    sensors = sorted(set(df["sensor_a"].astype(str)).union(set(df["sensor_b"].astype(str))))
    idx = {sensor: i for i, sensor in enumerate(sensors)}
    matrix = np.full((len(sensors), len(sensors)), np.nan, dtype=np.float32)

    for row in df.itertuples(index=False):
        a = str(row.sensor_a)
        b = str(row.sensor_b)
        value = float(row.ks)
        i = idx[a]
        j = idx[b]
        matrix[i, j] = value
        matrix[j, i] = value

    np.fill_diagonal(matrix, 0.0)
    return sensors, matrix


def plot_ks_heatmap(sensors: list[str], matrix: np.ndarray, output_path: Path) -> None:
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


def cmd_ks_plot(args: argparse.Namespace) -> None:
    sensors, matrix = load_ks_table(Path(args.input_csv))
    plot_ks_heatmap(sensors, matrix, Path(args.output_png))
    print(f"Saved KS heatmap to {Path(args.output_png).resolve()}")

LOSS_KEYS = [
    ("loss_mean", "total"),
    ("loss_id_mean", "L_id (ArcFace, v1)"),
    ("loss_mat_mean", "L_mat (Q<-teacher)"),
    ("loss_deg_mean", "L_deg (concept)"),
    ("loss_ortho_mean", "L_ortho"),
    ("loss_adv_mean", "L_adv (sensor GRL)"),
    ("loss_qpair_mean", "L_pair (cross-sensor Q)"),
]


def mean_q_std(q_by_sensor: dict) -> float:
    """Trung bình std của Q across sensors — đo mức độ Q collapse (gần 0 = collapse)."""
    stds = [v.get("std", float("nan")) for v in q_by_sensor.values()] if q_by_sensor else []
    stds = [s for s in stds if s == s]  # drop NaN
    return float(np.mean(stds)) if stds else float("nan")


def mean_q_spread(q_by_sensor: dict) -> float:
    """Khoảng q95-q05 trung bình — đo độ trải của Q (0 = mọi ảnh cùng điểm)."""
    spreads = []
    for v in (q_by_sensor or {}).values():
        if "q95" in v and "q05" in v:
            spreads.append(v["q95"] - v["q05"])
    return float(np.mean(spreads)) if spreads else float("nan")


def cmd_train_log(args: argparse.Namespace) -> None:
    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"Không thấy log: {log_path}")
    out_png = Path(args.out) if args.out else log_path.with_name("train_curves.png")

    rows = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("Log rỗng.")

    x = np.arange(len(rows))                       # trục epoch nối liền các stage
    stages = [r.get("stage", "") for r in rows]
    # vị trí đổi stage để vẽ vạch dọc
    boundaries = [i for i in range(1, len(stages)) if stages[i] != stages[i - 1]]
    stage_names = []
    seen = set()
    for i, s in enumerate(stages):
        if s not in seen:
            stage_names.append((i, s))
            seen.add(s)

    sensor_acc = [r.get("sensor_acc_mean", float("nan")) for r in rows]
    q_std = [mean_q_std(r.get("q_by_sensor", {})) for r in rows]
    q_spread = [mean_q_spread(r.get("q_by_sensor", {})) for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)

    # --- (1) các đường loss ---
    ax = axes[0]
    for key, label in LOSS_KEYS:
        vals = np.array([r.get(key, np.nan) for r in rows], dtype=np.float64)
        if np.all(~np.isfinite(vals)) or np.nanmax(np.abs(vals)) < 1e-9:
            continue  # bỏ loss luôn = 0 (tắt)
        ax.plot(x, vals, marker="o", ms=3, label=label)
    ax.set_ylabel("loss (mean/epoch)")
    ax.set_title(f"Loss curves — {log_path.parent.name}")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)

    # --- (2) Q collapse indicators ---
    ax = axes[1]
    ax.plot(x, q_std, marker="o", ms=3, color="crimson", label="mean Q std (per-sensor)")
    ax.plot(x, q_spread, marker="s", ms=3, color="darkorange", label="mean Q spread (q95-q05)")
    ax.axhline(0.05, color="gray", ls="--", lw=1, label="ngưỡng ~collapse")
    ax.set_ylabel("Q variability (thang 0..1)")
    ax.set_title("Q biến thiên — gần 0 = collapse, càng cao càng tốt")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # --- (3) sensor accuracy (invariance) ---
    ax = axes[2]
    ax.plot(x, sensor_acc, marker="o", ms=3, color="teal", label="sensor_acc (đầu đoán sensor)")
    n_sensors = max((len(r.get("q_by_sensor", {})) for r in rows), default=1)
    if n_sensors > 0:
        ax.axhline(1.0 / n_sensors, color="gray", ls="--", lw=1, label=f"chance=1/{n_sensors}")
    ax.set_ylabel("sensor accuracy")
    ax.set_title("Bất biến sensor — càng GẦN chance càng tốt (xoá được dấu sensor)")
    ax.set_xlabel("epoch (nối liền các stage)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # vạch + nhãn ranh giới stage trên cả 3 subplot
    for ax in axes:
        for b in boundaries:
            ax.axvline(b - 0.5, color="black", ls=":", lw=1, alpha=0.5)
    for i, s in stage_names:
        axes[0].text(i, axes[0].get_ylim()[1], s.replace("_", "\n"),
                     fontsize=7, va="top", ha="left", alpha=0.7)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # tóm tắt text ra console
    print(f"Đã lưu biểu đồ: {out_png.resolve()}")
    print(f"{'stage':<24} {'ep':>3} {'L_mat':>9} {'val_mat':>9} {'L_pair':>9} {'L_deg':>9} {'Qstd':>8} {'sens_acc':>9}")
    for i, r in enumerate(rows):
        print(f"{r.get('stage',''):<24} {r.get('epoch',i):>3} "
              f"{r.get('loss_mat_mean',float('nan')):>9.4f} "
              f"{r.get('val_mat',float('nan')):>9.4f} "
              f"{r.get('loss_qpair_mean',float('nan')):>9.4f} "
              f"{r.get('loss_deg_mean',float('nan')):>9.4f} "
              f"{q_std[i]:>8.4f} {sensor_acc[i]:>9.3f}")


def cmd_monitor(args: argparse.Namespace) -> None:
    """Text diagnostics + verdict from a train_log.jsonl (decide keep-training / retune /
    change architecture). Qspread = std across per-sensor mean Q (invariance); Qwithin =
    mean per-sensor Q std (collapse detector); sensAcc -> chance means GRL invariance works."""
    rows = [json.loads(l) for l in open(args.log, encoding="utf-8") if l.strip()]
    if not rows:
        raise SystemExit(f"empty log: {args.log}")

    print(f"{'stage':20} ep | Qspread Qwithin | mat*l  adv   id*l | sensAcc qpair   deg  | grl")
    print("-" * 92)
    for r in rows:
        qs = r.get("q_by_sensor", {}) or {}
        means = [v["mean"] for v in qs.values()] if qs else [0.0]
        stds = [v.get("std", 0.0) for v in qs.values()] if qs else [0.0]
        qspread = float(np.std(means)) * 100.0
        qwithin = float(np.mean(stds)) * 100.0
        matw = float(r.get("loss_mat_mean", 0.0)) * args.lambda_mat
        advw = float(r.get("loss_adv_mean", 0.0)) + float(r.get("loss_adv_concept_mean", 0.0))
        idw = float(r.get("loss_id_mean", 0.0)) * args.lambda_id
        print(f"{r['stage'][:20]:20} {int(r.get('epoch', 0)):>2} | "
              f"{qspread:6.2f} {qwithin:6.2f} | {matw:5.2f} {advw:4.2f} {idw:4.2f} | "
              f"{float(r.get('sensor_acc_mean', 0.0)):6.3f} "
              f"{float(r.get('loss_qpair_mean', 0.0)):5.3f} "
              f"{float(r.get('loss_deg_mean', 0.0)):5.3f} | "
              f"{float(r.get('grl_lambda_last', 0.0)):.2f}")

    last = rows[-1]
    qs = last.get("q_by_sensor", {}) or {}
    stds = [v.get("std", 0.0) for v in qs.values()] if qs else [0.0]
    qwithin = float(np.mean(stds)) * 100.0
    matw = float(last.get("loss_mat_mean", 0.0)) * args.lambda_mat
    advw = float(last.get("loss_adv_mean", 0.0)) + float(last.get("loss_adv_concept_mean", 0.0))
    n_sensors = max(1, len(qs))
    sensacc = float(last.get("sensor_acc_mean", 0.0))

    print("\nverdict (last epoch):")
    if qwithin < 1.0:
        print("  [!] Qwithin ~0 -> Q likely COLLAPSED. Lower --lambda-adv or raise --lambda-mat.")
    else:
        print(f"  [ok] Qwithin={qwithin:.1f} (>1) -> Q still has spread, not collapsed.")
    if advw > 3 * max(1e-6, matw):
        print(f"  [!] adversary({advw:.2f}) >> teacher({matw:.2f}) -> teacher starved. Raise --lambda-mat / lower --lambda-id.")
    else:
        print(f"  [ok] teacher({matw:.2f}) vs adversary({advw:.2f}) comparable.")
    if sensacc > 2.0 / n_sensors:
        print(f"  [!] sensAcc={sensacc:.3f} > chance(~{1/n_sensors:.3f}) -> GRL not invariant yet. Raise --lambda-adv.")
    else:
        print(f"  [ok] sensAcc={sensacc:.3f} ~ chance({1/n_sensors:.3f}) -> sensor-invariant.")


# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description="SFIQ reporting utilities (sensor | ks-plot | train-log | monitor).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_sensor = sub.add_parser("sensor", help="Sensor confusion-matrix + bias pair-stats từ embeddings.")
    p_sensor.add_argument("--embeddings-csv", default="SFIQ-2/outputs/sifq_design_full_scores.csv")
    p_sensor.add_argument("--output-dir", default="SFIQ-2/metrics/generated")
    p_sensor.add_argument("--seed", type=int, default=0)
    p_sensor.add_argument("--cv", type=int, default=5)
    p_sensor.set_defaults(func=cmd_sensor)

    p_ks = sub.add_parser("ks-plot", help="Heatmap KS pairwise kiểu NFIQ2.")
    p_ks.add_argument("--input-csv", default="SFIQ-2/plots/ks_sifq.csv")
    p_ks.add_argument("--output-png", default="SFIQ-2/plots/ks_sifq_nfiq2_style.png")
    p_ks.set_defaults(func=cmd_ks_plot)

    p_tl = sub.add_parser("train-log", help="Đường loss + Q-std + sensor_acc theo epoch.")
    p_tl.add_argument("--log", required=True, help="Đường dẫn train_log.jsonl")
    p_tl.add_argument("--out", default="", help="PNG output (mặc định cạnh file log).")
    p_tl.set_defaults(func=cmd_train_log)

    p_mon = sub.add_parser("monitor", help="Chẩn đoán text + verdict từ train_log.jsonl (giữ/đổi recipe).")
    p_mon.add_argument("--log", required=True, help="Đường dẫn train_log.jsonl")
    p_mon.add_argument("--lambda-mat", type=float, default=10.0)
    p_mon.add_argument("--lambda-id", type=float, default=0.2)
    p_mon.set_defaults(func=cmd_monitor)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

