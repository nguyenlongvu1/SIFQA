#!/usr/bin/env python3
"""
4-stage training loop + logging scaffold (TASK 4.3).

This is designed to be runnable once you provide a CSV manifest (see train/README.md),
but it does not include any project-specific loss (SIFQ details) beyond:
- L_id: optional CE on finger_id
- L_q: optional MSE on q_gt
- L_adv: CE on sensor (via GRL)

It logs per-epoch:
- loss components
- per-sensor Q stats and histograms
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from sifq.model import SIFQModel
from sifq.losses import (
    margin_classification_loss,
    matcher_teacher_loss,
    q_pair_margin_loss,
)
from sifq.degradations import CONCEPT_INDEX, DEG_TO_TARGETS, apply_degradation_pil, sample_degradation_pair

try:
    from tqdm.auto import tqdm  # type: ignore

    _HAS_TQDM = True
except Exception:
    tqdm = None  # type: ignore
    _HAS_TQDM = False


class ManifestDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 224,
        teacher_map: dict[str, np.ndarray] | None = None,
        teacher_dim: int = 0,
        q_mat_map: dict[str, float] | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.teacher_map = teacher_map
        self.teacher_dim = int(teacher_dim)
        self.q_mat_map = q_mat_map or {}
        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return int(self.df.shape[0])

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["path"]
        img = Image.open(path).convert("RGB")
        x = self.tf(img)
        finger_id = int(row["finger_id"]) if "finger_id" in row and not pd.isna(row["finger_id"]) else -1
        sensor = int(row["sensor_id"])
        roll = int(row["roll"]) if "roll" in row and not pd.isna(row["roll"]) else -1
        q_gt = float(row["q_gt"]) if "q_gt" in row and not pd.isna(row["q_gt"]) else float("nan")
        if np.isfinite(q_gt) and q_gt > 1.5:
            q_gt = q_gt / 100.0
        if self.teacher_map is None or self.teacher_dim <= 0:
            teacher_emb = np.zeros((1,), dtype=np.float32)
            teacher_ok = 0
        else:
            te = self.teacher_map.get(path)
            if te is None:
                teacher_emb = np.zeros((self.teacher_dim,), dtype=np.float32)
                teacher_ok = 0
            else:
                teacher_emb = te.astype(np.float32, copy=False)
                teacher_ok = 1
        q_mat = float(self.q_mat_map.get(path, float("nan")))
        return x, finger_id, sensor, roll, q_gt, teacher_emb, teacher_ok, q_mat


class DegradationDataset(Dataset):
    """
    Wrap a manifest dataset to produce (clean, degraded_i, degraded_j) triplets
    for controlled degradation ranking + concept supervision (L_deg).
    """

    def __init__(self, base: ManifestDataset, seed: int = 0):
        self.base = base
        self.rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        row = self.base.df.iloc[idx]
        path = row["path"]
        img = Image.open(path).convert("RGB")

        # sample degradation type + two levels i<j
        rr = random.Random(int(self.rng.integers(0, 2**31 - 1)))
        deg, li, lj = sample_degradation_pair(rr)

        clean = self.base.tf(img)
        xi = self.base.tf(apply_degradation_pil(img, deg, li))
        xj = self.base.tf(apply_degradation_pil(img, deg, lj))
        sensor_id = int(row["sensor_id"])
        return clean, xi, xj, deg, li, lj, sensor_id


class SensorBalancedBatchSampler:
    """
    Yield batches with (approximately) equal samples per sensor.

    This improves GRL pressure and stabilizes adversarial training.
    """

    def __init__(self, sensor_ids: np.ndarray, batch_size: int, seed: int = 0):
        self.sensor_ids = np.asarray(sensor_ids, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.sensors = sorted(set(self.sensor_ids.tolist()))
        if len(self.sensors) < 2:
            raise ValueError("Need >=2 sensors for balanced batching")
        self.per_sensor = max(1, self.batch_size // len(self.sensors))

        self.rng = np.random.default_rng(self.seed)
        self.idxs_by_sensor = {s: np.where(self.sensor_ids == s)[0].tolist() for s in self.sensors}

    def __iter__(self):
        # create shuffled pools per sensor
        pools = {}
        for s, idxs in self.idxs_by_sensor.items():
            idxs = list(idxs)
            self.rng.shuffle(idxs)
            pools[s] = idxs

        # iterate until any pool is exhausted
        while True:
            batch = []
            for s in self.sensors:
                pool = pools[s]
                if len(pool) < self.per_sensor:
                    return
                for _ in range(self.per_sensor):
                    batch.append(pool.pop())
            # fill remainder randomly from all remaining
            rem = self.batch_size - len(batch)
            if rem > 0:
                all_left = []
                for s in self.sensors:
                    all_left.extend(pools[s])
                if len(all_left) < rem:
                    return
                self.rng.shuffle(all_left)
                batch.extend(all_left[:rem])
                # remove used
                used = set(all_left[:rem])
                for s in self.sensors:
                    pools[s] = [i for i in pools[s] if i not in used]
            self.rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        # conservative estimate
        min_count = min(len(v) for v in self.idxs_by_sensor.values())
        return (min_count * len(self.sensors)) // self.batch_size


class PairedKeyBatchSampler:

    def __init__(
        self,
        finger_ids: np.ndarray,
        rolls: np.ndarray,
        sensor_ids: np.ndarray,
        batch_size: int,
        pairs_per_batch: int = 8,
        sensors_per_pair: int = 2,
        seed: int = 0,
    ):
        self.finger_ids = np.asarray(finger_ids, dtype=np.int64)
        self.rolls = np.asarray(rolls, dtype=np.int64)
        self.sensor_ids = np.asarray(sensor_ids, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.pairs_per_batch = int(pairs_per_batch)
        self.sensors_per_pair = int(sensors_per_pair)
        self.seed = int(seed)
        if self.pairs_per_batch <= 0 or self.sensors_per_pair < 2:
            raise ValueError("pairs_per_batch must be >0 and sensors_per_pair must be >=2")

        self.rng = np.random.default_rng(self.seed)

        # Build groups
        groups: dict[tuple[int, int], dict[int, list[int]]] = {}
        for i, (fid, r, sid) in enumerate(zip(self.finger_ids, self.rolls, self.sensor_ids)):
            if fid < 0 or r < 0:
                continue
            key = (int(fid), int(r))
            g = groups.get(key)
            if g is None:
                g = {}
                groups[key] = g
            g.setdefault(int(sid), []).append(int(i))

        # Keep only keys with >=2 distinct sensors
        self.keys = [k for k, bys in groups.items() if len(bys) >= 2]
        if not self.keys:
            raise ValueError("No paired keys with >=2 sensors found; cannot use PairedKeyBatchSampler")
        self.groups = groups

        self.all_idxs = np.arange(len(self.finger_ids), dtype=np.int64)

    def __iter__(self):
        keys = list(self.keys)
        self.rng.shuffle(keys)
        key_ptr = 0
        while True:
            batch: list[int] = []
            # Add paired samples
            for _ in range(self.pairs_per_batch):
                if key_ptr >= len(keys):
                    return
                key = keys[key_ptr]
                key_ptr += 1
                bys = self.groups[key]
                sensors = list(bys.keys())
                if len(sensors) < 2:
                    continue
                self.rng.shuffle(sensors)
                sensors = sensors[: self.sensors_per_pair]
                for sid in sensors:
                    idxs = bys[sid]
                    batch.append(int(self.rng.choice(idxs)))
            # Fill remainder
            rem = self.batch_size - len(batch)
            if rem <= 0:
                batch = batch[: self.batch_size]
            else:
                fill = self.rng.choice(self.all_idxs, size=rem, replace=False).tolist()
                batch.extend([int(x) for x in fill])
            self.rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        # Approx number of batches
        per = max(1, self.pairs_per_batch)
        return len(self.keys) // per


class PairedBalancedBatchSampler:
    """Cross-sensor PAIRS (for L_pair) AND sensor-BALANCE (for the GRL adversary) in the
    same batch — resolves the --pair-batch vs --balanced-batch trade-off.

    Each batch = `pairs_per_batch` cross-sensor pairs whose two sensors are picked to be
    the currently least-used in the batch (so the pairs themselves balance sensors), then
    a sensor-balanced random fill drawn from the LEAST-used sensors over ALL images — which
    naturally includes single-sensor fingers (e.g. FVC) so they are not starved.
    """

    def __init__(self, finger_ids, rolls, sensor_ids, batch_size,
                 pairs_per_batch: int = 8, seed: int = 0):
        self.finger_ids = np.asarray(finger_ids, dtype=np.int64)
        self.rolls = np.asarray(rolls, dtype=np.int64)
        self.sensor_ids = np.asarray(sensor_ids, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.pairs_per_batch = int(pairs_per_batch)
        self.rng = np.random.default_rng(int(seed))

        groups: dict[tuple[int, int], dict[int, list[int]]] = {}
        for i, (fid, r, sid) in enumerate(zip(self.finger_ids, self.rolls, self.sensor_ids)):
            if fid < 0 or r < 0:
                continue
            groups.setdefault((int(fid), int(r)), {}).setdefault(int(sid), []).append(int(i))
        self.groups = groups
        self.pair_keys = [k for k, bys in groups.items() if len(bys) >= 2]
        if not self.pair_keys:
            raise ValueError("No cross-sensor pairs found; use --balanced-batch instead.")
        self.sensors = sorted(set(int(s) for s in self.sensor_ids.tolist()))
        self.idxs_by_sensor = {s: np.where(self.sensor_ids == s)[0] for s in self.sensors}

    def __iter__(self):
        keys = list(self.pair_keys)
        self.rng.shuffle(keys)
        ptr = 0
        while True:
            batch: list[int] = []
            count = {s: 0 for s in self.sensors}
            # 1) cross-sensor pairs, sensors chosen to balance the batch
            for _ in range(self.pairs_per_batch):
                if ptr >= len(keys):
                    return
                bys = self.groups[keys[ptr]]; ptr += 1
                avail = sorted(bys.keys(), key=lambda s: count[s])  # least-used first
                if len(avail) < 2:
                    continue
                for s in avail[:2]:
                    batch.append(int(self.rng.choice(bys[s]))); count[s] += 1
            # 2) sensor-balanced fill from least-used sensors over ALL images (incl. FVC)
            while len(batch) < self.batch_size:
                s = min(self.sensors, key=lambda s: count[s])
                batch.append(int(self.rng.choice(self.idxs_by_sensor[s]))); count[s] += 1
            self.rng.shuffle(batch)
            yield batch[: self.batch_size]

    def __len__(self) -> int:
        return max(1, len(self.pair_keys) // max(1, self.pairs_per_batch))


@dataclass
class StageCfg:
    name: str
    epochs: int
    lr: float
    lambda_adv: float
    enable_id: bool
    enable_q: bool
    enable_deg: bool = True   # allow L_deg independently of enable_q


def per_sensor_q_stats(q: np.ndarray, sensors: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for s in sorted(set(sensors.tolist())):
        vals = q[sensors == s]
        if vals.size == 0:
            continue
        qs = np.quantile(vals, [0.05, 0.5, 0.95]).tolist()
        out[str(s)] = {
            "count": float(vals.size),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "q05": float(qs[0]),
            "q50": float(qs[1]),
            "q95": float(qs[2]),
        }
    return out


def plot_q_hist(q: np.ndarray, sensors: np.ndarray, out_png: str, bins: int = 40, alpha: float = 0.35) -> None:
    plt.figure(figsize=(10, 5))
    for s in sorted(set(sensors.tolist())):
        vals = q[sensors == s]
        if vals.size == 0:
            continue
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        vmin = float(vals.min())
        vmax = float(vals.max())
        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-4:
            # Degenerate distribution; plot as a single point-mass marker.
            plt.axvline(vmin if np.isfinite(vmin) else 0.0, alpha=alpha, label=str(s))
            continue
        plt.hist(vals, bins=bins, alpha=alpha, density=True, label=str(s))
    plt.title("Predicted Q distribution by sensor")
    plt.xlabel("Q_hat")
    plt.ylabel("Density")
    plt.legend(ncol=8, fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def compute_teacher_targets(
    teacher_map: dict,
    df: pd.DataFrame,
    min_samples: int = 5,
    norm_mode: str = "global",
) -> dict:
    """
    Pre-compute teacher quality targets q_mat(x) from cosine-to-own-centroid.

    For every sample x of identity y:
        c_y = mean of L2-normalised teacher embeddings for identity y
        cos = cos(teacher(x), c_y)            # higher => better capture

    Three normalisation modes turn `cos` into q_mat in [0,1]:

    - "identity" (design doc 4.3.1): per-identity z-score then sigmoid:
          q = sigmoid((cos - mu_y) / sigma_y)
      Removes identity-difficulty bias, but the target collapses toward ~0.5 for
      every identity, giving the model almost no cross-sample signal -> Q collapses.

    - "global" (default): single z-score over ALL samples then sigmoid:
          q = sigmoid((cos - mu) / sigma)
      Keeps the full cross-sample spread (genuinely poor captures map low, sharp
      captures map high), which is a far stronger learning signal for Q.

    - "sensor" (recommended with a deep teacher): per-sensor z-score then sigmoid:
          q = sigmoid((cos - mu_s) / sigma_s)   for sensor s
      Keeps full cross-identity spread but equalises each sensor's mean, removing the
      "this sensor scores higher" OFFSET that lives in the label and that GRL cannot
      reach. This is the at-source teacher de-bias for FLaRE/DeepPrint.

    Returns {path: float}; samples without a teacher embedding or in identity groups
    smaller than min_samples are simply absent (treated as NaN downstream).
    """
    if not teacher_map or "finger_id" not in df.columns:
        return {}

    from collections import defaultdict

    groups: dict = defaultdict(list)
    for _, row in df.iterrows():
        path = str(row["path"])
        emb = teacher_map.get(path)
        if emb is None:
            continue
        fid = row.get("finger_id")
        if fid is None or (isinstance(fid, float) and np.isnan(fid)):
            continue
        norm = float(np.linalg.norm(emb))
        if norm < 1e-8:
            continue
        groups[fid].append((path, emb / norm))

    # First pass: cosine-to-own-centroid for every eligible sample.
    cos_records: list[tuple[str, float]] = []
    per_identity: dict = {}
    for fid, items in groups.items():
        if len(items) < min_samples:
            continue
        paths = [p for p, _ in items]
        embs = np.stack([e for _, e in items])            # (N, D)
        prototype = embs.mean(axis=0)
        p_norm = float(np.linalg.norm(prototype))
        if p_norm < 1e-8:
            continue
        prototype /= p_norm
        cos_sims = embs @ prototype                        # (N,) cos of normalised vecs
        per_identity[fid] = (paths, cos_sims)
        for p, c in zip(paths, cos_sims.tolist()):
            cos_records.append((p, float(c)))

    if not cos_records:
        return {}

    q_mat_map: dict = {}
    if str(norm_mode) == "identity":
        for fid, (paths, cos_sims) in per_identity.items():
            mu_y = float(cos_sims.mean())
            sigma_y = float(cos_sims.std())
            if sigma_y < 1e-6:
                for p in paths:
                    q_mat_map[p] = 0.5
                continue
            q = 1.0 / (1.0 + np.exp(-(cos_sims - mu_y) / sigma_y))
            for p, qq in zip(paths, q.tolist()):
                q_mat_map[p] = float(qq)
    elif str(norm_mode) == "sensor":
        # Per-sensor z-score then sigmoid. Removes the per-sensor OFFSET in
        # cos-to-centroid ("sensor X scores systematically higher") — exactly the
        # teacher sensor-bias that GRL cannot fix (GRL cleans the representation,
        # not the precomputed label). Keeps the full cross-identity spread WITHIN
        # each sensor while equalising sensor means, so q_mat (and thus Q) can no
        # longer encode which sensor took the image. Recommended with a deep teacher
        # (FLaRE/DeepPrint) whose embedding still leaks some sensor signature.
        path2sensor = {
            str(p): str(s)
            for p, s in zip(df["path"].astype(str), df["sensor"].astype(str))
        } if "sensor" in df.columns else {}
        from collections import defaultdict as _dd

        by_sensor: dict = _dd(list)
        for p, c in cos_records:
            by_sensor[path2sensor.get(p, "_")].append((p, c))
        for _s, items in by_sensor.items():
            cs = np.array([c for _, c in items], dtype=np.float64)
            mu_s = float(cs.mean())
            sigma_s = float(cs.std())
            if sigma_s < 1e-6:
                for p, _c in items:
                    q_mat_map[p] = 0.5
            else:
                for p, c in items:
                    q_mat_map[p] = float(1.0 / (1.0 + np.exp(-(c - mu_s) / sigma_s)))
    else:
        all_cos = np.array([c for _, c in cos_records], dtype=np.float64)
        mu = float(all_cos.mean())
        sigma = float(all_cos.std())
        if sigma < 1e-6:
            for p, _ in cos_records:
                q_mat_map[p] = 0.5
        else:
            for p, c in cos_records:
                q_mat_map[p] = float(1.0 / (1.0 + np.exp(-(c - mu) / sigma)))

    return q_mat_map


@torch.no_grad()
def eval_val_mat(model, val_dl, device, huber_delta: float) -> float:
    """Mean L_mat (Huber(Q, q_mat)) on the val set — measures whether Q generalises
    to held-out identities. Returns NaN if no val sample carries a q_mat target."""
    if val_dl is None:
        return float("nan")
    model.eval()
    total = 0.0
    count = 0
    for x, _fid, sens, _roll, _qgt, _temb, _tok, q_mat_b in val_dl:
        x = x.to(device)
        sens = sens.to(device)
        out = model(x, grl_lambda=0.0, sensor_ids=sens)
        qm = q_mat_b.to(device, dtype=torch.float32)
        valid = torch.isfinite(qm)
        if valid.any():
            l = torch.nn.functional.huber_loss(out.q[valid], qm[valid], delta=float(huber_delta), reduction="sum")
            total += float(l.detach().cpu().item())
            count += int(valid.sum().item())
    return float(total / count) if count > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split", default="train",
                    help="Comma-separated manifest split(s) to TRAIN on (filters the 'split' "
                         "column). Default 'train' EXCLUDES val/test to prevent leakage. Use "
                         "'train,fvc_train' to also train on FVC; 'all' disables filtering.")
    ap.add_argument("--out-dir", default="experiments/sifq_run")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    ap.add_argument("--tqdm", action="store_true", help="Show tqdm progress bars (requires tqdm).")
    ap.add_argument("--balanced-batch", action="store_true", help="Use sensor-balanced batching.")
    ap.add_argument("--pair-batch", action="store_true", help="Use pair-aware batching by (finger_id, roll).")
    ap.add_argument("--pairs-per-batch", type=int, default=8, help="How many paired keys per batch (pair-batch).")
    ap.add_argument("--sensors-per-pair", type=int, default=2, help="How many sensors per paired key (pair-batch).")
    ap.add_argument("--epochs1", type=int, default=3)
    ap.add_argument("--epochs2", type=int, default=3)
    ap.add_argument("--epochs3", type=int, default=5)
    ap.add_argument("--epochs4", type=int, default=3)
    ap.add_argument("--lambda-adv", type=float, default=0.3,
                    help="Weight for L_adv (GRL sensor adversary on f_intermediate). Design "
                         "§4.3.2 default = 0.3.")
    ap.add_argument("--lambda-adv-concept", type=float, default=0.0,
                    help="Base GRL lambda for the LIGHT direct adversary on the 6 concepts. "
                         "DEFAULT 0 (off, safest): L_pair already gives collapse-free DIRECT "
                         "Q-invariance, so the f_intermediate adversary + L_pair cover #1 with "
                         "zero bottleneck pressure. Enable (try 0.1) ONLY if train_log shows Q "
                         "still sensor-dependent (high q_by_sensor spread / sensor_acc>chance); "
                         "it adds direct pressure on the 6-dim bottleneck and risks narrowing Q.")
    ap.add_argument("--grl-ramp", default="linear", choices=["none", "linear", "sigmoid"])
    ap.add_argument("--lambda-qpair", type=float, default=1.0,
                    help="Weight for L_pair (design 4.3.2): hinge max(0,|Q_s1-Q_s2|-delta) over "
                         "cross-sensor pairs. Needs --pair-batch to have paired samples in a batch.")
    ap.add_argument("--delta", type=float, default=0.1,
                    help="Floor margin for L_pair (design 4.3.2). L_pair is now QUALITY-AWARE: the "
                         "actual margin = max(delta, |q_mat_s1 - q_mat_s2|), so genuinely-worse "
                         "captures (e.g. SD302 sensor H) are NOT forced equal; delta only floors "
                         "the cosmetic case. Was 0.05 (v3, fixed); 0.1 gives cosmetic headroom.")
    ap.add_argument("--lambda-deg", type=float, default=0.0, help="Weight for controlled degradation loss L_deg.")
    ap.add_argument("--deg-clean-quantile", type=float, default=0.5,
                    help="Design §5: degradation base = CLEAN images. Use only the images whose "
                         "teacher q_mat is in the top (1-quantile) as the L_deg base, so blur/noise "
                         "chains start from GOOD captures (degrading an already-bad image is "
                         "uninformative). 0.5 = cleanest half. Set 0 to use ALL images (old behaviour).")
    ap.add_argument("--gamma-deg", type=float, default=0.5, help="Gamma for concept supervision term inside L_deg.")
    ap.add_argument(
        "--gamma-deg-nontarget",
        type=float,
        default=0.25,
        help="Gamma for non-target concept consistency inside L_deg (keeps unrelated concepts close to clean).",
    )
    ap.add_argument(
        "--gamma-deg-mono",
        type=float,
        default=0.5,
        help="Gamma for pairwise monotonic concept ordering inside L_deg.",
    )
    ap.add_argument("--margin-m", type=float, default=0.1, help="Margin m for ranking loss inside L_deg.")
    ap.add_argument(
        "--deg-every",
        type=int,
        default=4,
        help="Compute degradation loss once every K train steps (reduces cost).",
    )
    ap.add_argument("--lambda-ortho", type=float, default=0.02, help="Weight for concept decorrelation (L_ortho).")
    ap.add_argument("--id-loss-mode", default="arcface", choices=["ce", "arcface", "cosface"], help="Identity loss mode.")
    ap.add_argument("--id-scale", type=float, default=30.0, help="Scale for ArcFace/CosFace logits.")
    ap.add_argument("--id-margin", type=float, default=0.35, help="Margin for ArcFace/CosFace.")
    ap.add_argument("--id-easy-margin", action="store_true", help="Use easy-margin ArcFace variant.")
    ap.add_argument("--teacher-npz", default="", help="Optional DeepPrint teacher NPZ with keys emb,path.")
    ap.add_argument("--q-mat-csv", default="",
                    help="Precomputed teacher targets CSV (columns: path,q_mat) — e.g. FLaRE "
                         "mean-genuine, per-sensor z-scored (make_flare_qmat.py). OVERRIDES the "
                         "--teacher-npz cosine-to-centroid path. Use for matchers whose score is "
                         "NOT a flat-embedding cosine (FLaRE dense descriptor, Bozorth3). Run "
                         "training from the repo root so relative paths resolve like the manifest.")
    ap.add_argument("--lambda-q", type=float, default=0.0,
                    help="Weight for L_q: MSE(Q, q_gt) regression on manifest q_gt column. "
                         "Default 0 — disabled to avoid inheriting q_gt bias (e.g. NFIQ2). "
                         "Set > 0 only if q_gt comes from a clean, sensor-invariant source.")
    ap.add_argument("--lambda-mat", type=float, default=10.0,
                    help="Weight for L_mat: Huber(Q, q_mat) teacher signal. Raised from 1.0 because "
                         "Huber on [0,1] is ~0.02 in magnitude vs ArcFace CE ~10 — at equal lambdas "
                         "the teacher contributes <0.2%% of the gradient and Q collapses to a narrow "
                         "band. Watch loss_mat_mean*lambda_mat vs loss_id_mean*lambda_id in train_log "
                         "and tune so they are comparable.")
    ap.add_argument("--lambda-mat-huber-delta", type=float, default=1.0,
                    help="Huber delta for L_mat loss.")
    ap.add_argument("--mat-norm", default="global", choices=["global", "identity", "sensor"],
                    help="Teacher target normalisation: 'global' (single z-score, strong "
                         "cross-sample spread), 'identity' (per-identity z-score, design doc "
                         "4.3.1 — tends to collapse Q), or 'sensor' (per-sensor z-score — "
                         "removes the sensor OFFSET in the teacher label that GRL cannot fix; "
                         "recommended with a deep teacher like FLaRE/DeepPrint).")
    ap.add_argument("--steps-per-epoch", type=int, default=0, help="Optional cap on train batches per epoch (0=all).")
    ap.add_argument("--eval-batches", type=int, default=0, help="Optional cap on eval batches for Q logging (0=all).")
    ap.add_argument(
        "--resume-from",
        default="",
        help="Optional checkpoint path to resume model/optimizer from (expects keys model,opt,sensor2id).",
    )
    ap.add_argument(
        "--run-stages",
        default="1,2,3,4",
        help="Comma-separated stage numbers to run (1..4). Useful to run only stage4.",
    )
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="Gradient accumulation steps. Effective batch = batch-size × grad-accum.")
    ap.add_argument("--patience", type=int, default=0,
                    help="Early stopping patience per stage (epochs). 0=disabled.")
    ap.add_argument("--min-delta", type=float, default=1e-4,
                    help="Min loss improvement to reset patience counter.")
    ap.add_argument("--lambda-id", type=float, default=0.2,
                    help="Weight for L_id (ArcFace). Lowered from 1.0: ArcFace CE (~10) otherwise "
                         "dominates the gradient ~600x over the teacher and degradation signals, "
                         "starving Q-grounding. With a clean teacher (FLaRE) it can take more load, "
                         "so ArcFace only needs to supply representation, not carry utility.")
    ap.add_argument("--val-csv", default="",
                    help="Optional validation manifest (held-out subjects). When given, early "
                         "stopping tracks L_mat on val (Q generalisation to unseen identities) "
                         "instead of train loss.")
    args = ap.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = str(args.device)

    df = pd.read_csv(args.csv)
    if "path" not in df.columns or "sensor" not in df.columns:
        raise SystemExit("CSV must contain at least columns: path,sensor")

    # Filter to the training split(s) so val/test never leak into training. Without this,
    # ArcFace/L_deg/L_adv would still shape the backbone on held-out images even though they
    # carry no q_mat target.
    if "split" in df.columns and str(args.split).lower() != "all":
        keep = {s.strip() for s in str(args.split).split(",") if s.strip()}
        before = len(df)
        df = df[df["split"].astype(str).isin(keep)].reset_index(drop=True)
        print(f"[split] training on {sorted(keep)}: {len(df)}/{before} rows (val/test excluded)")
        if df.empty:
            present = sorted(pd.read_csv(args.csv)["split"].astype(str).unique())
            raise SystemExit(f"No rows for --split {args.split}. Splits present: {present}")

    # Normalise Windows backslashes → forward slashes before resolving (cross-platform manifest)
    df["path"] = df["path"].astype(str).str.replace("\\", "/", regex=False)
    df["path"] = df["path"].map(lambda p: p if os.path.isabs(p) else os.path.abspath(p))
    missing = [p for p in df["path"].tolist() if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Missing {len(missing)} image paths. First: {missing[0]}")

    teacher_map = None
    teacher_dim = 0
    if args.teacher_npz:
        z = np.load(args.teacher_npz, allow_pickle=True)
        if "emb" not in z or "path" not in z:
            raise SystemExit("--teacher-npz must contain keys: emb,path")
        paths = [str(p) for p in z["path"].tolist()]
        embs = z["emb"].astype(np.float32)
        teacher_dim = int(embs.shape[1])
        teacher_map = {}
        for i, p in enumerate(paths):
            teacher_map[p] = embs[i]
            # Also map absolute path for matching manifests that store abs paths
            if not os.path.isabs(p):
                teacher_map[os.path.abspath(p)] = embs[i]

    resume_state = None
    resume_sensor2id = None
    if args.resume_from:
        resume_state = torch.load(args.resume_from, map_location="cpu")
        resume_sensor2id = resume_state.get("sensor2id")

    # Map sensor -> id (keep consistent ids if resuming)
    sensors = sorted(df["sensor"].astype(str).unique().tolist())
    sensor2id = resume_sensor2id or {s: i for i, s in enumerate(sensors)}
    df["sensor_id"] = df["sensor"].astype(str).map(sensor2id)

    # The matching identity is a FINGER, not a subject. In SD302a the manifest
    # finger_id is the subject and `roll` (1..10) is the finger position, so the
    # true identity is (finger_id, roll). Without this, ArcFace pulls a person's
    # 10 different fingers together and the teacher centroids mix 10 unrelated
    # prints — both wreck the quality target. Compose them into a finger-level id.
    if "finger_id" in df.columns and "roll" in df.columns:
        df["finger_id"] = df["finger_id"].astype(str) + "_" + df["roll"].astype(str)

    n_ids: Optional[int] = None
    if "finger_id" in df.columns and df["finger_id"].notna().any():
        # Ensure contiguous ids for CE
        fids = sorted(df["finger_id"].dropna().astype(str).unique().tolist())
        fid2id = {f: i for i, f in enumerate(fids)}
        df["finger_id"] = df["finger_id"].astype(str).map(fid2id)
        n_ids = len(fid2id)

    q_mat_map: dict = {}
    if str(args.q_mat_csv) and float(args.lambda_mat) > 0:
        # Precomputed teacher targets (FLaRE mean-genuine / Bozorth3 / any matcher).
        # Paths are matched to df["path"] via the SAME abspath normalisation used above.
        qdf = pd.read_csv(args.q_mat_csv)
        if "path" not in qdf.columns or "q_mat" not in qdf.columns:
            raise SystemExit("--q-mat-csv must contain columns: path,q_mat")
        for _, rr in qdf.iterrows():
            p = str(rr["path"]).replace("\\", "/")
            p = p if os.path.isabs(p) else os.path.abspath(p)
            v = float(rr["q_mat"])
            if np.isfinite(v):
                q_mat_map[p] = v
        matched = sum(1 for p in df["path"] if p in q_mat_map)
        finite = np.array(list(q_mat_map.values()), dtype=np.float64)
        spread = f"mean={finite.mean():.3f} std={finite.std():.3f}" if finite.size else "n/a"
        print(f"[L_mat] q_mat from CSV {args.q_mat_csv}: {len(q_mat_map)} rows, "
              f"{matched}/{len(df)} matched to manifest | {spread}")
        if matched == 0:
            raise SystemExit(
                "No --q-mat-csv path matched the manifest after abspath normalisation. "
                "Run training from the repo root and ensure the CSV stores the same relative "
                "paths as the manifest (e.g. 'data/SD302a/...')."
            )
    elif teacher_map is not None and float(args.lambda_mat) > 0:
        q_mat_map = compute_teacher_targets(teacher_map, df, norm_mode=str(args.mat_norm))
        n_targets = sum(1 for v in q_mat_map.values() if np.isfinite(v))
        finite = np.array([v for v in q_mat_map.values() if np.isfinite(v)], dtype=np.float64)
        spread = f"mean={finite.mean():.3f} std={finite.std():.3f}" if finite.size else "n/a"
        print(f"[L_mat] q_mat targets ({args.mat_norm}) for {n_targets}/{len(df)} samples | {spread}")

    # Silent-zero footgun: lambda_mat>0 (default 10) but no teacher provided => L_mat is 0
    # for every sample with no error, so Q trains with NO utility anchor.
    if float(args.lambda_mat) > 0 and not q_mat_map:
        print("[warn] lambda_mat>0 but NO teacher targets found (no --q-mat-csv and no usable "
              "--teacher-npz) => L_mat will be 0. Provide a teacher (FLaRE q_mat CSV) or set "
              "--lambda-mat 0 to acknowledge training without a utility anchor.")

    ds = ManifestDataset(
        df,
        image_size=int(args.image_size),
        teacher_map=teacher_map,
        teacher_dim=teacher_dim,
        q_mat_map=q_mat_map,
    )
    if bool(args.pair_batch):
        if "roll" not in df.columns:
            raise SystemExit("--pair-batch requires roll column for (finger_id, roll) pairing.")
        if "finger_id" not in df.columns:
            raise SystemExit("--pair-batch requires finger_id column for (finger_id, roll) pairing.")
        # Combined sampler: cross-sensor pairs (L_pair) AND sensor-balance (GRL) + FVC fill.
        sampler = PairedBalancedBatchSampler(
            df["finger_id"].to_numpy(),
            df["roll"].to_numpy(),
            df["sensor_id"].to_numpy(),
            batch_size=int(args.batch_size),
            pairs_per_batch=int(args.pairs_per_batch),
            seed=0,
        )
        dl = DataLoader(ds, batch_sampler=sampler, num_workers=int(args.num_workers))
    elif bool(args.balanced_batch):
        sampler = SensorBalancedBatchSampler(df["sensor_id"].to_numpy(), batch_size=int(args.batch_size), seed=0)
        dl = DataLoader(ds, batch_sampler=sampler, num_workers=int(args.num_workers))
    else:
        dl = DataLoader(
            ds,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            drop_last=True,
        )

    # Design §4.3.2: L_sens = L_pair + lambda_adv*L_adv. L_pair needs same-finger
    # cross-sensor pairs IN the batch — only --pair-batch supplies them. Without it,
    # q_pair_margin_loss returns ~0 and the design's primary, collapse-free Q-invariance
    # signal is silently absent (invariance then leans entirely on the GRL adversary).
    if float(args.lambda_qpair) > 0 and not bool(args.pair_batch):
        print("[warn] L_pair is ON (lambda_qpair>0) but --pair-batch is OFF => no cross-sensor "
              "pairs per batch => L_pair ~ 0. Add --pair-batch to enable design L_sens fully.")

    # L_deg (concept grounding) is the core of the concept-bottleneck novelty but defaults
    # OFF (lambda_deg=0). Silently training without it leaves the 6 concepts ungrounded.
    if float(args.lambda_deg) <= 0:
        print("[warn] lambda_deg=0 => L_deg OFF => the 6 concepts are NOT grounded to "
              "degradations (Track-4 grounding / interpretability will be meaningless). "
              "Pass --lambda-deg 1.0 to enable concept grounding.")

    deg_dl = None
    if float(args.lambda_deg) > 0:
        # Design §5: the degradation base should be CLEAN images (FVC in the design). Degrading
        # an already-poor capture gives an uninformative "bad>worse>worst" chain and dilutes
        # L_deg. Restrict the base to the images the teacher rates highest (q_mat >= quantile)
        # so blur/noise start from good captures. Falls back to ALL images if no q_mat / quantile 0.
        deg_base = ds
        if q_mat_map and 0.0 < float(args.deg_clean_quantile) < 1.0:
            qv = np.array([q_mat_map.get(p, np.nan) for p in df["path"].tolist()], dtype=np.float64)
            finite = np.isfinite(qv)
            if int(finite.sum()) > 0:
                thr = float(np.nanquantile(qv[finite], float(args.deg_clean_quantile)))
                clean_df = df[finite & (qv >= thr)].reset_index(drop=True)
                if len(clean_df) >= int(args.batch_size):
                    deg_base = ManifestDataset(
                        clean_df, image_size=int(args.image_size),
                        teacher_map=teacher_map, teacher_dim=teacher_dim, q_mat_map=q_mat_map,
                    )
                    print(f"[L_deg] degradation base = cleanest {1 - float(args.deg_clean_quantile):.0%} "
                          f"by q_mat: {len(clean_df)}/{len(df)} images (q_mat>={thr:.3f})")
        deg_ds = DegradationDataset(deg_base, seed=0)
        deg_dl = DataLoader(
            deg_ds,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            drop_last=True,
        )

    # Optional validation loader on held-out subjects: drives early stopping by L_mat
    # generalisation (Q vs teacher q_mat on identities never seen in training).
    val_dl = None
    if str(args.val_csv) and teacher_map is not None and float(args.lambda_mat) > 0:
        vdf = pd.read_csv(args.val_csv)
        if "path" not in vdf.columns or "sensor" not in vdf.columns:
            raise SystemExit("--val-csv must contain at least columns: path,sensor")
        vdf["path"] = vdf["path"].astype(str).str.replace("\\", "/", regex=False)
        vdf["path"] = vdf["path"].map(lambda p: p if os.path.isabs(p) else os.path.abspath(p))
        vdf = vdf[vdf["path"].map(os.path.exists)].reset_index(drop=True)
        vdf["sensor_id"] = vdf["sensor"].astype(str).map(sensor2id).fillna(0).astype(int)
        # Same finger-level identity as the train set so val centroids are per-finger.
        if "finger_id" in vdf.columns and "roll" in vdf.columns:
            vdf["finger_id"] = vdf["finger_id"].astype(str) + "_" + vdf["roll"].astype(str)
        # q_mat for val computed from teacher within the val set (its own centroids).
        val_q_mat = compute_teacher_targets(teacher_map, vdf, norm_mode=str(args.mat_norm))
        n_vt = sum(1 for v in val_q_mat.values() if np.isfinite(v))
        val_ds = ManifestDataset(
            vdf, image_size=int(args.image_size),
            teacher_map=teacher_map, teacher_dim=teacher_dim, q_mat_map=val_q_mat,
        )
        val_dl = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False,
                            num_workers=int(args.num_workers))
        print(f"[val] {len(vdf)} images, {n_vt} with q_mat target → early stop on val L_mat")

    # Design-faithful SIFQ (sifq/model.py): spatial concept head, strict bottleneck,
    # single GRL D, + restored ArcFace identity metric (n_ids) so the backbone learns
    # fingerprint structure (the frozen teacher only passes a scalar Q-target).
    model = SIFQModel(backbone="mobilenet_v2", n_sensors=len(sensor2id), n_ids=n_ids).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    huber = nn.SmoothL1Loss()

    stages = [
        StageCfg("stage1_id_warmup", epochs=int(args.epochs1), lr=1e-3, lambda_adv=0.0, enable_id=True, enable_q=False, enable_deg=True),
        StageCfg("stage2_add_q", epochs=int(args.epochs2), lr=5e-4, lambda_adv=0.0, enable_id=True, enable_q=True, enable_deg=True),
        StageCfg(
            "stage3_add_invariance",
            epochs=int(args.epochs3),
            lr=5e-4,
            lambda_adv=float(args.lambda_adv),
            enable_id=True,
            enable_q=True,
        ),
        StageCfg("stage4_finetune", epochs=int(args.epochs4), lr=2e-4,
                 lambda_adv=float(args.lambda_adv), enable_id=True, enable_q=True),
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "q_hist"), exist_ok=True)
    with open(os.path.join(args.out_dir, "sensor_map.json"), "w", encoding="utf-8") as f:
        json.dump(sensor2id, f, indent=2)

    if resume_state is not None:
        model.load_state_dict(resume_state["model"], strict=False)
        if "opt" in resume_state:
            opt.load_state_dict(resume_state["opt"])

    run_stages = {int(x) for x in str(args.run_stages).split(",") if x.strip()}

    def grl_lambda_for(step_idx: int, total_steps: int, base: float) -> float:
        if base <= 0:
            return 0.0
        if args.grl_ramp == "none" or total_steps <= 1:
            return float(base)
        p = float(step_idx) / float(max(1, total_steps - 1))
        if args.grl_ramp == "linear":
            return float(base) * p
        # sigmoid ramp (DANN-style)
        return float(base) * float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

    def stage_ramp(stage_num: int, epoch: int, stage_epochs: int, first_active_stage: int) -> float:
        """Design L(t) = alpha(t)*L_mat + beta(t)*L_sens + gamma(t)*L_deg + L_ortho.

        Smooth 0->1 ramp of a loss-family weight across the stage it first activates in,
        then held at 1. Replaces the old binary stage on/off gating (deviation #10c).
        """
        if stage_num < first_active_stage:
            return 0.0
        if stage_num > first_active_stage:
            return 1.0
        return float(min(1.0, (epoch + 1) / max(1, int(stage_epochs))))

    for stage in stages:
        stage_num = int(stage.name.split("_", 1)[0].replace("stage", ""))
        if stage_num not in run_stages:
            continue
        for pg in opt.param_groups:
            pg["lr"] = stage.lr

        _best_stage_loss = float("inf")
        _patience_count = 0

        for epoch in range(stage.epochs):
            model.train()
            # Design L(t) loss-family ramps (deviation #10c fix): L_mat from stage2,
            # L_pair from stage3 (invariance), L_deg from stage1. Adversarial L_adv is
            # ramped separately by the GRL lambda, so it is not re-scaled here.
            alpha = stage_ramp(stage_num, epoch, stage.epochs, 2)   # L_mat
            beta = stage_ramp(stage_num, epoch, stage.epochs, 3)    # L_pair
            gamma = stage_ramp(stage_num, epoch, stage.epochs, 1)   # L_deg
            losses = []
            deg_iter = iter(deg_dl) if deg_dl is not None else None
            total_steps = int(args.steps_per_epoch) if int(args.steps_per_epoch) > 0 else len(dl)
            it = dl
            if bool(args.tqdm) and _HAS_TQDM:
                it = tqdm(dl, desc=f"{stage.name} train e{epoch}", total=total_steps)
            for bi, (x, fid, sens, roll, q_gt, t_emb, t_ok, q_mat_batch) in enumerate(it):
                x = x.to(device)
                sens = sens.to(device)
                fid = fid.to(device)
                roll = roll.to(device)
                q_gt_t = q_gt.to(device, dtype=torch.float32)

                lam = grl_lambda_for(bi, total_steps, float(stage.lambda_adv))
                # Light direct concept-GRL active only at invariance stages (when lam>0).
                lam_c = grl_lambda_for(bi, total_steps, float(args.lambda_adv_concept)) \
                    if stage.lambda_adv > 0 else 0.0
                out = model(x, grl_lambda=lam, sensor_ids=sens, grl_lambda_concept=lam_c)

                loss_id = torch.tensor(0.0, device=device)
                # Identity supervision = ArcFace/CosFace on emb + id_metric_weight. (The
                # design has no student CE head; the CE-on-id_logits path was always dead
                # — id_logits stayed None — and was removed.)
                if stage.enable_id and getattr(model, "id_metric_weight", None) is not None:
                    ok_id = fid >= 0
                    if ok_id.any():
                        loss_id = margin_classification_loss(
                            out.emb[ok_id],
                            fid[ok_id],
                            model.id_metric_weight,
                            scale=float(args.id_scale),
                            margin=float(args.id_margin),
                            mode=str(args.id_loss_mode),
                            easy_margin=bool(args.id_easy_margin),
                        )

                loss_q = torch.tensor(0.0, device=device)
                if float(args.lambda_q) > 0 and stage.enable_q:
                    ok_q = torch.isfinite(q_gt_t)
                    if ok_q.any():
                        loss_q = mse(out.q[ok_q], q_gt_t[ok_q])

                # GRL already scales encoder gradients by lam; keep CE weight = 1 for stability.
                loss_adv = ce(out.sensor_logits, sens) if lam > 0 else torch.tensor(0.0, device=device)
                # Light direct concept adversary (GRL scales encoder grad by lam_c, kept small).
                loss_adv_concept = (
                    ce(out.concept_sensor_logits, sens)
                    if (lam_c > 0 and out.concept_sensor_logits is not None)
                    else torch.tensor(0.0, device=device)
                )

                # L_pair (design 4.3.2): cross-sensor invariance on Q. Part of L_sens,
                # active from the invariance stage (3+). Ramped by beta.
                loss_qpair = torch.tensor(0.0, device=device)
                if float(args.lambda_qpair) > 0 and stage.enable_q and stage_num >= 3:
                    loss_qpair = q_pair_margin_loss(
                        out.q, fid, roll, sens, delta=float(args.delta),
                        q_mat=q_mat_batch.to(device, dtype=torch.float32),
                    )

                loss_deg = torch.tensor(0.0, device=device)
                loss_ortho = torch.tensor(0.0, device=device)
                if (
                    deg_dl is not None
                    and float(args.lambda_deg) > 0
                    and stage.enable_deg
                    and (int(args.deg_every) <= 1 or (bi % int(args.deg_every) == 0))
                ):
                    # one batch of degradation samples per train step (cheap + stable)
                    try:
                        clean, xi, xj, deg, li, lj, deg_sensor = next(deg_iter)  # type: ignore[name-defined]
                    except Exception:
                        deg_iter = iter(deg_dl)  # type: ignore[assignment]
                        clean, xi, xj, deg, li, lj, deg_sensor = next(deg_iter)
                    clean = clean.to(device)
                    xi = xi.to(device)
                    xj = xj.to(device)
                    deg_sensor = deg_sensor.to(device)
                    out_c = model(clean, grl_lambda=0.0, sensor_ids=deg_sensor)
                    out_i = model(xi, grl_lambda=0.0, sensor_ids=deg_sensor)
                    out_j = model(xj, grl_lambda=0.0, sensor_ids=deg_sensor)

                    m = float(args.margin_m)
                    # ranking: Q(clean) > Q(i) > Q(j)
                    l_rank = torch.relu(out_j.q - out_i.q + m).mean() + torch.relu(out_i.q - out_c.q + m).mean()

                    # concept supervision for degraded images i and j.
                    # target concepts follow an interpolated monotonic target anchored
                    # to the clean prediction. Non-target concepts are pulled back toward
                    # clean ONLY for single-target degradations (deviation #9b fix):
                    # multi-target degradations physically perturb other concepts too, so
                    # forcing those "non-targets" to clean would teach a false invariance.
                    l_c = torch.tensor(0.0, device=device)
                    l_nt = torch.tensor(0.0, device=device)
                    l_mono = torch.tensor(0.0, device=device)
                    # Softened rails keep concepts off the saturating sigmoid ends so the
                    # bottleneck cannot collapse to a constant [0,0,0,1,0,0] corner.
                    for b in range(xi.shape[0]):
                        d = str(deg[b])
                        targets = DEG_TO_TARGETS.get(d, [])
                        target_names = {cname for cname, _direction in targets}
                        level_i = float(max(0, min(3, int(li[b])))) / 3.0
                        level_j = float(max(0, min(3, int(lj[b])))) / 3.0
                        for cname, direction in targets:
                            ci = CONCEPT_INDEX[cname]
                            # RELATIVE grounding: the target for a degraded image is the CLEAN
                            # image's OWN concept value (detached) pushed toward the `bad` end by
                            # `level`. We do NOT pin the clean concept to an absolute rail (0.8),
                            # so L_deg teaches only the DELTA (how far the concept drops as the
                            # degradation rises) and stops fighting the teacher over the absolute
                            # concept/Q level — the rail-vs-teacher conflict that parked
                            # continuity/minutiae at a constant. Anti-collapse is now carried by the
                            # teacher (Q must vary -> concepts must vary) + l_mono + L_ortho.
                            anchor = out_c.concepts[b, ci].detach()
                            bad = anchor.new_tensor(0.8 if direction == "increase" else 0.2)
                            ti = anchor * (1.0 - level_i) + bad * level_i
                            tj = anchor * (1.0 - level_j) + bad * level_j
                            l_c = l_c + huber(out_i.concepts[b, ci], ti)
                            l_c = l_c + huber(out_j.concepts[b, ci], tj)
                            if direction == "increase":
                                l_mono = l_mono + torch.relu(out_i.concepts[b, ci] - out_j.concepts[b, ci] + m)
                            else:
                                l_mono = l_mono + torch.relu(out_j.concepts[b, ci] - out_i.concepts[b, ci] + m)
                        if len(targets) <= 1:
                            for cname in CONCEPT_INDEX.keys():
                                if cname in target_names:
                                    continue
                                ci = CONCEPT_INDEX[cname]
                                clean_val = out_c.concepts[b, ci].detach()
                                l_nt = l_nt + huber(out_i.concepts[b, ci], clean_val)
                                l_nt = l_nt + huber(out_j.concepts[b, ci], clean_val)
                    l_c = l_c / float(max(1, xi.shape[0]))
                    l_nt = l_nt / float(max(1, xi.shape[0]))
                    l_mono = l_mono / float(max(1, xi.shape[0]))

                    loss_deg = (
                        l_rank
                        + float(args.gamma_deg) * l_c
                        + float(args.gamma_deg_nontarget) * l_nt
                        + float(args.gamma_deg_mono) * l_mono
                    )

                if float(args.lambda_ortho) > 0:
                    # decorrelate concepts in current batch
                    c = out.concepts
                    c = c - c.mean(dim=0, keepdim=True)
                    c = c / (c.std(dim=0, keepdim=True) + 1e-6)
                    corr = (c.T @ c) / float(max(1, c.shape[0] - 1))
                    off = corr - torch.diag(torch.diag(corr))
                    loss_ortho = torch.mean(off**2)

                # L_mat: teacher-as-quality Huber loss (design doc Section 4.3.1)
                loss_mat = torch.tensor(0.0, device=device)
                if float(args.lambda_mat) > 0 and stage.enable_q:
                    q_mat_t = q_mat_batch.to(device, dtype=torch.float32)
                    loss_mat = matcher_teacher_loss(
                        out.q, q_mat_t, huber_delta=float(args.lambda_mat_huber_delta)
                    )

                # L(t) = L_id + L_q + L_adv + alpha*L_mat + beta*L_pair + gamma*L_deg + L_ortho
                # (alpha/beta/gamma are the design's stage ramps; L_adv ramps via GRL lambda).
                loss = (
                    float(args.lambda_id) * loss_id
                    + float(args.lambda_q) * loss_q
                    + loss_adv
                    + loss_adv_concept
                    + beta * float(args.lambda_qpair) * loss_qpair
                    + gamma * float(args.lambda_deg) * loss_deg
                    + float(args.lambda_ortho) * loss_ortho
                    + alpha * float(args.lambda_mat) * loss_mat
                )
                scaled_loss = loss / float(args.grad_accum)
                scaled_loss.backward()
                if (bi + 1) % int(args.grad_accum) == 0 or (bi + 1) == total_steps:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                sensor_acc = float((torch.argmax(out.sensor_logits.detach(), dim=1) == sens).float().mean().item())
                losses.append(
                    {
                        "loss": float(loss.detach().cpu().item()),
                        "loss_id": float(loss_id.detach().cpu().item()),
                        "loss_q": float(loss_q.detach().cpu().item()),
                        "loss_adv": float(loss_adv.detach().cpu().item()),
                        "loss_adv_concept": float(loss_adv_concept.detach().cpu().item()),
                        "loss_qpair": float(loss_qpair.detach().cpu().item()),
                        "loss_deg": float(loss_deg.detach().cpu().item()),
                        "loss_ortho": float(loss_ortho.detach().cpu().item()),
                        "loss_mat": float(loss_mat.detach().cpu().item()),
                        "sensor_acc": float(sensor_acc),
                        "grl_lambda": float(lam),
                    }
                )
                if int(args.steps_per_epoch) > 0 and (bi + 1) >= int(args.steps_per_epoch):
                    break

            # eval pass for Q distribution logging (cheap, reuse train set)
            model.eval()
            all_q = []
            all_s = []
            with torch.no_grad():
                eval_total = int(args.eval_batches) if int(args.eval_batches) > 0 else len(dl)
                ite = dl
                if bool(args.tqdm) and _HAS_TQDM:
                    ite = tqdm(dl, desc=f"{stage.name} eval e{epoch}", total=eval_total)
                for ebi, (x, _, sens, _, _, _, _, _) in enumerate(ite):
                    x = x.to(device)
                    out = model(x, grl_lambda=0.0, sensor_ids=sens.to(device))
                    all_q.append(out.q.detach().cpu().numpy())
                    all_s.append(sens.numpy())
                    if int(args.eval_batches) > 0 and (ebi + 1) >= int(args.eval_batches):
                        break
            q_np = np.concatenate(all_q, axis=0)
            s_np = np.concatenate(all_s, axis=0)

            stats = {
                "stage": stage.name,
                "epoch": epoch,
                "lr": stage.lr,
                "lambda_adv": stage.lambda_adv,
                "alpha_mat": alpha,
                "beta_pair": beta,
                "gamma_deg": gamma,
                "loss_mean": float(np.mean([x["loss"] for x in losses])),
                "loss_id_mean": float(np.mean([x["loss_id"] for x in losses])),
                "loss_q_mean": float(np.mean([x["loss_q"] for x in losses])),
                "loss_adv_mean": float(np.mean([x["loss_adv"] for x in losses])),
                "loss_adv_concept_mean": float(np.mean([x["loss_adv_concept"] for x in losses])),
                "loss_qpair_mean": float(np.mean([x["loss_qpair"] for x in losses])),
                "loss_deg_mean": float(np.mean([x["loss_deg"] for x in losses])),
                "loss_ortho_mean": float(np.mean([x["loss_ortho"] for x in losses])),
                "loss_mat_mean": float(np.mean([x["loss_mat"] for x in losses])),
                "sensor_acc_mean": float(np.mean([x["sensor_acc"] for x in losses])) if losses else float("nan"),
                "grl_lambda_last": float(losses[-1]["grl_lambda"]) if losses else 0.0,
                "q_by_sensor": per_sensor_q_stats(q_np, s_np),
            }

            # Validation L_mat on held-out subjects (generalisation of Q). Falls back to
            # train loss for early stopping when no val set is supplied.
            val_mat = float("nan")
            if val_dl is not None:
                val_mat = eval_val_mat(model, val_dl, device, float(args.lambda_mat_huber_delta))
                stats["val_mat"] = val_mat
                model.train()

            log_path = os.path.join(args.out_dir, "train_log.jsonl")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(stats) + "\n")

            # Live per-epoch console summary (mirrors monitor_train.py columns). Qspread =
            # cross-sensor std of per-sensor mean Q; Qwithin = within-sensor std (x100). Both
            # ~0 => Q collapsed; Qspread small + Qwithin healthy => sensor-invariant.
            _qs = stats.get("q_by_sensor", {}) or {}
            _means = [v["mean"] for v in _qs.values()] if _qs else [0.0]
            _stds = [v.get("std", 0.0) for v in _qs.values()] if _qs else [0.0]
            _qspread = float(np.std(_means)) * 100.0
            _qwithin = float(np.mean(_stds)) * 100.0
            vm = stats.get("val_mat")
            _vm = f" val_mat {vm:.4f}" if isinstance(vm, float) and np.isfinite(vm) else ""
            print(
                f"[epoch] {stage.name} e{epoch} | loss {stats['loss_mean']:.3f} | "
                f"id {stats['loss_id_mean']:.2f} mat {stats['loss_mat_mean']:.4f} "
                f"deg {stats['loss_deg_mean']:.3f} adv {stats['loss_adv_mean']:.3f} "
                f"advc {stats['loss_adv_concept_mean']:.3f} qpair {stats['loss_qpair_mean']:.4f} "
                f"ortho {stats['loss_ortho_mean']:.3f} | sensAcc {stats['sensor_acc_mean']:.3f} "
                f"Qspread {_qspread:.2f} Qwithin {_qwithin:.2f}{_vm}",
                flush=True,
            )

            # Early stopping. Only monitor val L_mat when Q is actually being trained
            # (enable_q stages). In stage 1 (L_mat off) Q is untrained, so val_mat is
            # flat/noisy and would stop the concept-grounding stage prematurely — there
            # we fall back to mean train loss (L_id + L_deg), which decreases as concepts
            # ground and identity warms up.
            train_loss = float(np.mean([x["loss"] for x in losses]))
            use_val = (val_dl is not None) and np.isfinite(val_mat) and bool(stage.enable_q)
            monitor = val_mat if use_val else train_loss
            monitor_name = "val_mat" if use_val else "train_loss"
            if int(args.patience) > 0:
                if _best_stage_loss - monitor > float(args.min_delta):
                    _best_stage_loss = monitor
                    _patience_count = 0
                else:
                    _patience_count += 1
                    if _patience_count >= int(args.patience):
                        print(f"[EarlyStopping] {stage.name} e{epoch}: {monitor_name} "
                              f"không cải thiện {_patience_count} epoch (best={_best_stage_loss:.4f})")
                        break

            plot_q_hist(
                q_np,
                s_np,
                out_png=os.path.join(args.out_dir, "q_hist", f"{stage.name}_epoch{epoch:03d}.png"),
            )

            ckpt = {
                "stage": stage.name,
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sensor2id": sensor2id,
            }
            torch.save(ckpt, os.path.join(args.out_dir, "checkpoints", f"{stage.name}_epoch{epoch:03d}.pt"))

        # stage boundary checkpoint
        torch.save(
            {
                "stage": stage.name,
                "model": model.state_dict(),
                "sensor2id": sensor2id,
            },
            os.path.join(args.out_dir, "checkpoints", f"{stage.name}_final.pt"),
        )

    print(f"Done. Logs: {os.path.abspath(os.path.join(args.out_dir, 'train_log.jsonl'))}")


if __name__ == "__main__":
    main()
