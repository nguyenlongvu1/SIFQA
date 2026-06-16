from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def q_pair_margin_loss(
    q: torch.Tensor,
    finger_id: torch.Tensor,
    roll: torch.Tensor,
    sensor: torch.Tensor,
    delta: float = 0.05,
    max_pairs_per_key: int = 8,
    q_mat: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Design 4.3.2 — L_pair: cross-sensor invariance on the scalar quality Q for
    SD302-style paired keys (finger_id, roll).

        L_pair = mean over cross-sensor pairs of  max(0, |Q(x_s1) - Q(x_s2)| - delta)

    Needs batches that actually contain the same (finger_id, roll) captured by
    >=2 sensors (use --pair-batch), otherwise it returns 0.
    """
    if q.ndim != 1:
        raise ValueError("q must be (N,)")
    n = q.shape[0]
    if n < 2:
        return q.new_tensor(0.0)

    valid = (finger_id >= 0) & (roll >= 0)
    if valid.sum().item() < 2:
        return q.new_tensor(0.0)

    keys = torch.stack([finger_id, roll], dim=1)[valid]
    qv = q[valid]
    sv = sensor[valid]
    qmv = q_mat[valid] if q_mat is not None else None

    order = torch.argsort(keys[:, 0] * 100 + keys[:, 1])
    keys = keys[order]
    qv = qv[order]
    sv = sv[order]
    if qmv is not None:
        qmv = qmv[order]

    loss = q.new_tensor(0.0)
    pairs = 0
    i = 0
    d = float(delta)
    while i < keys.shape[0]:
        j = i + 1
        while j < keys.shape[0] and torch.equal(keys[j], keys[i]):
            j += 1
        if j - i >= 2:
            idxs = torch.arange(i, j, device=q.device)
            perm = idxs[torch.randperm(idxs.numel(), device=q.device)]
            tried = 0
            for a in perm:
                for b in perm:
                    if a >= b:
                        continue
                    if sv[a] == sv[b]:
                        continue
                    # Quality-AWARE margin: allow Q to differ by as much as the teacher
                    # q_mat genuinely differs, so a genuinely-worse capture (e.g. SD302
                    # sensor H) is NOT forced equal to the finger's good captures. Enforce
                    # invariance (>= delta) ONLY for cosmetic (near-equal q_mat) pairs.
                    margin = d
                    if qmv is not None and torch.isfinite(qmv[a]) and torch.isfinite(qmv[b]):
                        margin = torch.clamp(torch.abs(qmv[a] - qmv[b]), min=d)
                    diff = torch.abs(qv[a] - qv[b]) - margin
                    loss = loss + torch.clamp(diff, min=0.0)
                    pairs += 1
                    tried += 1
                    if tried >= max_pairs_per_key:
                        break
                if tried >= max_pairs_per_key:
                    break
        i = j

    if pairs == 0:
        return q.new_tensor(0.0)
    return loss / pairs


def matcher_teacher_loss(
    q: torch.Tensor,
    q_mat: torch.Tensor,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """
    L_mat = Huber(Q(x), q_mat(x)) using pre-computed teacher targets.

    q_mat values are pre-computed via compute_teacher_targets() in train_sifq.py:
        q_mat(x) = sigmoid( (cos(teacher(x), c_y) - mu) / sigma )
    NaN entries (samples without a teacher embedding) are skipped.
    """
    if q.ndim != 1:
        raise ValueError("q must be (N,)")
    valid = torch.isfinite(q_mat)
    if valid.sum() == 0:
        return q.new_tensor(0.0)
    return F.huber_loss(q[valid], q_mat[valid].detach(), delta=float(huber_delta), reduction="mean")


def margin_logits(
    features: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    scale: float = 30.0,
    margin: float = 0.5,
    mode: str = "arcface",
    easy_margin: bool = False,
) -> torch.Tensor:
    """
    Compute ArcFace/CosFace-adjusted classification logits.
    Used only by the v1 architecture (student identity head). v2 drops L_id.
    """
    if features.ndim != 2:
        raise ValueError("features must be (N,D)")
    if labels.ndim != 1:
        raise ValueError("labels must be (N,)")
    if weight.ndim != 2:
        raise ValueError("weight must be (C,D)")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have same N")

    x = F.normalize(features, p=2, dim=1)
    w = F.normalize(weight, p=2, dim=1)
    cosine = F.linear(x, w).clamp(-1.0, 1.0)
    num_classes = cosine.shape[1]
    one_hot = F.one_hot(labels.to(torch.int64), num_classes=num_classes).to(dtype=cosine.dtype)

    if mode.lower() == "cosface":
        logits = cosine - one_hot * float(margin)
        return logits * float(scale)

    if mode.lower() != "arcface":
        raise ValueError(f"Unsupported margin mode: {mode}")

    sine = torch.sqrt(torch.clamp(1.0 - cosine**2, min=0.0))
    cos_m = math.cos(float(margin))
    sin_m = math.sin(float(margin))
    phi = cosine * cos_m - sine * sin_m
    if easy_margin:
        phi = torch.where(cosine > 0.0, phi, cosine)
    else:
        th = math.cos(math.pi - float(margin))
        mm = math.sin(math.pi - float(margin)) * float(margin)
        phi = torch.where(cosine > th, phi, cosine - mm)

    logits = one_hot * phi + (1.0 - one_hot) * cosine
    return logits * float(scale)


def margin_classification_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    scale: float = 30.0,
    margin: float = 0.5,
    mode: str = "arcface",
    easy_margin: bool = False,
) -> torch.Tensor:
    """Cross-entropy on ArcFace/CosFace logits (v1 only)."""
    logits = margin_logits(
        features=features,
        labels=labels,
        weight=weight,
        scale=scale,
        margin=margin,
        mode=mode,
        easy_margin=easy_margin,
    )
    return F.cross_entropy(logits, labels.to(torch.int64))
