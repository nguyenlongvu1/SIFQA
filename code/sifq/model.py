from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from .grl import grl

BackboneName = Literal["mobilenet_v2", "efficientnet_b0"]


@dataclass
class SIFQOutputs:
    emb: torch.Tensor
    concepts: torch.Tensor
    q: torch.Tensor
    sensor_logits: torch.Tensor
    # Light DIRECT adversary on the 6 concepts (gentle Q-invariance guarantee, off
    # unless grl_lambda_concept>0). Complements the heavy f_intermediate adversary.
    concept_sensor_logits: Optional[torch.Tensor] = None


class SIFQModel(nn.Module):
    def __init__(
        self,
        backbone: BackboneName = "mobilenet_v2",
        emb_dim: int = 256,
        n_concepts: int = 6,
        n_sensors: int = 8,
        concept_hidden: int = 64,
        n_ids: Optional[int] = None,
        **_ignored,
    ):
        super().__init__()
        self.n_sensors = int(n_sensors)
        self.n_concepts = int(n_concepts)

        if backbone == "efficientnet_b0":
            net = tvm.efficientnet_b0(weights=None)
            feat_dim = net.classifier[1].in_features
            self.features = net.features
        elif backbone == "mobilenet_v2":
            net = tvm.mobilenet_v2(weights=None)
            feat_dim = net.classifier[1].in_features
            self.features = net.features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Spatial concept head, SPLIT so the adversary can attach to the intermediate
        # features (design §4.3.2: D(sensor | f_intermediate) reads the concept-head
        # INTERMEDIATE, not the final 6-dim bottleneck — adversarial pressure on 6 dims
        # collapses the concepts and hence Q). concept_stem -> f_intermediate (concept_hidden
        # dims), concept_out -> the 6 concept maps.
        self.concept_stem = nn.Sequential(
            nn.Conv2d(feat_dim, concept_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.concept_out = nn.Conv2d(concept_hidden, n_concepts, kernel_size=1)

        # Embedding projection (kept for Track-1/3 ERC outputs + optional analysis).
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )

        # ArcFace identity metric weight (RESTORED). The frozen teacher (L_mat) only
        # passes ONE scalar per image -> it sets Q's utility LEVEL but cannot teach the
        # backbone what a ridge/minutia is. ArcFace classifies ~n_ids finger identities
        # from `emb`, giving the dense, ground-truth-labelled representation signal that
        # makes the 6 concepts meaningful. Its sensor leakage is cleanable by GRL (the
        # leak lives in the representation, where GRL acts — unlike a biased teacher label).
        # Loss = margin_classification_loss(emb, fid, id_metric_weight) in train_sifq.py.
        self.id_metric_weight = None
        if n_ids is not None and int(n_ids) > 0:
            self.id_metric_weight = nn.Parameter(torch.empty(int(n_ids), emb_dim))
            nn.init.xavier_uniform_(self.id_metric_weight)

        # Aggregator: Q = MLP(concepts) -> strict bottleneck.
        self.quality_head = nn.Sequential(
            nn.Linear(n_concepts, max(8, n_concepts * 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, n_concepts * 2), 1),
        )

        # Adversarial sensor classifier D on f_intermediate (design §4.3.2 L_adv).
        # Input dim = concept_hidden (the intermediate), NOT n_concepts.
        self.sensor_head = nn.Sequential(
            nn.Linear(concept_hidden, max(8, concept_hidden)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, concept_hidden), n_sensors),
        )

        # Optional LIGHT direct adversary on the 6 concepts. f_intermediate-GRL cleans
        # the representation but only INDIRECTLY constrains Q (a different projection of
        # inter could resurface residual sensor). This head gives a direct, low-weight
        # invariance nudge on the exact vector Q reads — kept gentle (small lambda) so it
        # does NOT collapse the bottleneck the way a heavy concept-GRL would.
        self.concept_sensor_head = nn.Sequential(
            nn.Linear(n_concepts, max(8, n_concepts * 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, n_concepts * 2), n_sensors),
        )

    def forward(
        self,
        x: torch.Tensor,
        grl_lambda: float = 0.0,
        sensor_ids: Optional[torch.Tensor] = None,  # accepted, ignored (no SSBN)
        grl_lambda_concept: float = 0.0,            # light direct concept-GRL (off=0)
    ) -> SIFQOutputs:
        feat = self.features(x)                              # (N, C, H, W)

        inter_maps = self.concept_stem(feat)                # (N, concept_hidden, H, W) = f_intermediate
        concept_maps = self.concept_out(inter_maps)         # (N, n_concepts, H, W)
        concept_logits = F.adaptive_avg_pool2d(concept_maps, (1, 1)).flatten(1)
        concepts = torch.sigmoid(concept_logits)            # (N, n_concepts) in [0,1]
        f_intermediate = F.adaptive_avg_pool2d(inter_maps, (1, 1)).flatten(1)  # (N, concept_hidden)

        pooled = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        emb = F.normalize(self.proj(pooled), p=2, dim=1)

        q = torch.sigmoid(self.quality_head(concepts).squeeze(1))

        # GRL into f_intermediate (design §4.3.2): the sensor CE then pushes the
        # intermediate to be sensor-invariant, so the concepts derived from it — and Q —
        # become invariant WITHOUT squeezing the 6-dim bottleneck. L_pair (train_sifq.py)
        # handles the direct, collapse-free Q-level invariance.
        sensor_in = grl(f_intermediate, grl_lambda) if grl_lambda > 0 else f_intermediate
        sensor_logits = self.sensor_head(sensor_in)

        # Light direct concept-GRL (only when enabled) — gentle Q-invariance guarantee.
        concept_sensor_logits = None
        if grl_lambda_concept > 0:
            concept_sensor_logits = self.concept_sensor_head(grl(concepts, grl_lambda_concept))

        return SIFQOutputs(
            emb=emb,
            concepts=concepts,
            q=q,
            sensor_logits=sensor_logits,
            concept_sensor_logits=concept_sensor_logits,
        )
