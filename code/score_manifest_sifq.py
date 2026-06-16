#!/usr/bin/env python3
"""
score_manifest_sifq.py
======================
Chấm điểm SIFQ cho toàn bộ ảnh của MỘT split (mặc định: test) lấy từ manifest.

Script này lọc đúng split theo cột `split` trong manifest và dùng cột `sensor`
trực tiếp — nên không đoán nhầm sensor từ path (tránh lỗi sensor G) và không bị
rò rỉ dữ liệu training vào đánh giá.

Output CSV có đủ cột cho cả 4 tracks:
    path, sensor, identity, finger_id, roll, split,
    q_hat_raw (0..1), q_hat (0..100),
    ridge_valley_clarity, noise_level, contrast_uniformity,
    usable_area, ridge_frequency, orientation_coherence,
    f0..f255  (embedding — cần cho Track 1/3 ERC)

Usage:
    python SFIQ-2/code/score_manifest_sifq.py \
        --ckpt SFIQ-2/weights/stage4_finetune_final.pt \
        --manifest data/manifest_all.csv \
        --split test \
        --sensor-map SFIQ-2/weights/sensor_map.json \
        --output-csv SFIQ-2/outputs/sifq_test_scores.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Allow running from anywhere (repo root, code/, etc.) by making the sifq package importable.
_CODE_ROOT = Path(__file__).resolve().parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from sifq.model import SIFQModel
from sifq.degradations import CONCEPTS as CONCEPT_NAMES  # single source of truth (v2 concept set)


def resolve_device(s: str) -> str:
    if s == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return s


class ManifestScoreDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sensor2id: dict, image_size: int = 224):
        self.df = df.reset_index(drop=True)
        self.sensor2id = sensor2id
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
        img = Image.open(row["path"]).convert("RGB")
        x = self.tf(img)
        sensor_id = int(self.sensor2id.get(str(row["sensor"]), 0))
        return x, sensor_id, idx


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Score one manifest split with SIFQ.")
    ap.add_argument("--ckpt", default="SFIQ-2/weights/stage4_finetune_final.pt")
    ap.add_argument("--manifest", default="data/manifest_all.csv")
    ap.add_argument("--split", default="test",
                    help="Split to score: test|val|train|fvc_train|all (default: test).")
    ap.add_argument("--sensor-map", default="SFIQ-2/weights/sensor_map.json")
    ap.add_argument("--output-csv", default="SFIQ-2/outputs/sifq_test_scores.csv")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-emb", action="store_true",
                    help="Skip writing f0..fN embedding columns (smaller CSV; breaks Track1/3).")
    args = ap.parse_args()

    device = resolve_device(str(args.device))
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # --- load manifest and filter split ---
    df = pd.read_csv(args.manifest)
    for col in ("path", "sensor", "split"):
        if col not in df.columns:
            raise SystemExit(f"Manifest must contain column '{col}'. Found: {list(df.columns)}")

    if str(args.split).lower() != "all":
        df = df[df["split"].astype(str) == str(args.split)].copy()
    if df.empty:
        raise SystemExit(f"No rows for split={args.split} in {args.manifest}")

    # Manifest paths are relative to the repo root (the parent of the manifest's
    # folder, e.g. paths like "data/SD302a/..." sit next to "data/manifest_all.csv").
    # Resolve robustly: try CWD first, then anchor to the repo root, so the script
    # works regardless of which directory it is launched from.
    manifest_dir = Path(args.manifest).resolve().parent
    repo_root = manifest_dir.parent if manifest_dir.name == "data" else manifest_dir

    def resolve_path(p: str) -> str:
        p = str(p).replace("\\", "/")
        if os.path.isabs(p) and os.path.exists(p):
            return p
        cand_cwd = os.path.abspath(p)
        if os.path.exists(cand_cwd):
            return cand_cwd
        cand_root = str((repo_root / p).resolve())
        if os.path.exists(cand_root):
            return cand_root
        return cand_cwd  # fall back; will be flagged as missing below

    df["path"] = df["path"].map(resolve_path)
    exists = df["path"].map(os.path.exists)
    if not exists.all():
        n_missing = int((~exists).sum())
        print(f"[warn] dropping {n_missing} rows with missing image files "
              f"(first: {df.loc[~exists, 'path'].iloc[0]})")
        df = df[exists].copy()
    df = df.reset_index(drop=True)
    if df.empty:
        raise SystemExit(
            f"All images missing after path resolution (split={args.split}). "
            f"Check that the manifest paths exist relative to repo root: {repo_root}"
        )
    print(f"Scoring {len(df)} images (split={args.split}) from {args.manifest}")

    # --- load checkpoint + sensor map ---
    ck = torch.load(args.ckpt, map_location="cpu")
    sensor2id = ck.get("sensor2id")
    if args.sensor_map and Path(args.sensor_map).exists():
        sensor2id = json.loads(Path(args.sensor_map).read_text(encoding="utf-8"))
    if not isinstance(sensor2id, dict):
        sensor2id = {}
    id2sensor = {int(v): str(k) for k, v in sensor2id.items()} if sensor2id else {}

    # Warn about sensors in the data that the model never saw (routed to id 0).
    unknown = sorted(set(df["sensor"].astype(str)) - set(sensor2id.keys()))
    if sensor2id and unknown:
        print(f"[warn] sensors not in sensor_map (scored via id 0 fallback): {unknown}")

    sd = ck.get("model", {})
    model = SIFQModel(backbone="mobilenet_v2", n_sensors=max(1, len(id2sensor) or 8)).to(device)
    incompat = model.load_state_dict(sd, strict=False)
    if incompat.missing_keys:
        # Any missing key => those params stay at RANDOM init => Q is garbage with no
        # error. This is the v1-checkpoint-into-v2-model trap. Fail loudly instead.
        raise SystemExit(
            f"Checkpoint KHÔNG khớp model hiện tại: {len(incompat.missing_keys)} tham số bị "
            f"bỏ ở random init (vd {incompat.missing_keys[:5]}). Thường do code model.py đã đổi "
            f"sau khi train checkpoint này → q_hat sẽ là RÁC. Hãy score bằng đúng phiên bản "
            f"model.py khớp checkpoint. Unexpected keys: {incompat.unexpected_keys[:5]}"
        )
    if incompat.unexpected_keys:
        print(f"[warn] checkpoint thừa {len(incompat.unexpected_keys)} key không dùng "
              f"(ok nếu cố ý gỡ module): {incompat.unexpected_keys[:5]}")
    model.eval()

    ds = ManifestScoreDataset(df, sensor2id, image_size=int(args.image_size))
    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )

    n = len(df)
    emb_dim = int(model.proj[-1].out_features)
    q_raw = np.zeros((n,), dtype=np.float32)
    concepts = np.zeros((n, len(CONCEPT_NAMES)), dtype=np.float32)
    embeddings = np.zeros((n, emb_dim), dtype=np.float32) if not args.no_emb else None
    sensor_pred_id = np.full((n,), -1, dtype=np.int64)
    sensor_conf = np.full((n,), np.nan, dtype=np.float32)

    done = 0
    for x, sensor_id, idx in dl:
        x = x.to(device)
        sensor_id = sensor_id.to(device)
        out = model(x, grl_lambda=0.0, sensor_ids=sensor_id)
        idx_np = idx.numpy()
        q_raw[idx_np] = out.q.detach().cpu().numpy().astype(np.float32)
        concepts[idx_np] = out.concepts.detach().cpu().numpy().astype(np.float32)
        if embeddings is not None:
            embeddings[idx_np] = out.emb.detach().cpu().numpy().astype(np.float32)
        if out.sensor_logits is not None:
            prob = torch.softmax(out.sensor_logits.detach().cpu(), dim=1)
            pred = torch.argmax(prob, dim=1)
            sensor_pred_id[idx_np] = pred.numpy()
            sensor_conf[idx_np] = prob[torch.arange(prob.shape[0]), pred].numpy().astype(np.float32)
        done += x.shape[0]
        if done % 512 == 0 or done == n:
            print(f"  scored {done}/{n}")

    # --- assemble output frame ---
    out = pd.DataFrame()
    out["path"] = df["path"].values
    out["sensor"] = df["sensor"].astype(str).values
    # identity = finger_id (the unit Track 1/2/3 group by). Keep both names for compatibility.
    identity = (
        df["finger_id"].astype(str).values
        if "finger_id" in df.columns
        else df["sensor"].astype(str).values
    )
    out["identity"] = identity
    if "finger_id" in df.columns:
        out["finger_id"] = df["finger_id"].values
    out["roll"] = df["roll"].values if "roll" in df.columns else np.nan
    out["split"] = df["split"].astype(str).values
    out["q_hat_raw"] = q_raw
    out["q_hat"] = 100.0 * q_raw
    for ci, cname in enumerate(CONCEPT_NAMES):
        out[cname] = concepts[:, ci]
    out["sensor_pred_id"] = sensor_pred_id
    out["sensor_pred_label"] = [id2sensor.get(int(s), "") if s >= 0 else "" for s in sensor_pred_id]
    out["sensor_confidence"] = sensor_conf
    if embeddings is not None:
        # Build all f0..fN columns in one block to avoid fragmentation.
        emb_df = pd.DataFrame(embeddings, columns=[f"f{j}" for j in range(emb_dim)], index=out.index)
        out = pd.concat([out, emb_df], axis=1)

    out.to_csv(out_csv, index=False)
    print(f"Saved {len(out)} rows -> {out_csv.resolve()}")
    print(f"Q stats: mean={float(np.mean(out['q_hat'])):.2f} "
          f"std={float(np.std(out['q_hat'])):.2f} "
          f"min={float(np.min(out['q_hat'])):.2f} max={float(np.max(out['q_hat'])):.2f}")


if __name__ == "__main__":
    main()
