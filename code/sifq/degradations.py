from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF


DegradationType = Literal[
    "blur", "noise", "jpeg", "occlusion", "dry_skin",
    "pressure_wet", "orientation_break", "ridge_stretch",
]


# Concept set v2 (2026-06-14 redesign). Each concept is anchored to a NAMED NFIQ2 /
# ISO 29794-4 quality feature and grounded by exactly one degradation family (isolated
# grounding — see DEG_TO_TARGETS). Changes vs the design's original 6:
#   DROP continuity            -> folded into ridge_valley_clarity (LCS already covers
#                                 ridge continuity; on real data continuity was collinear
#                                 with clarity OR sensor-driven, eta2_sensor up to 0.46).
#   DROP minutiae_reliability  -> derived = f(usable_area, ridge_valley_clarity); the
#                                 design itself calls it an aggregate, and it was dead.
#   ADD  usable_area           -> ISO/IEC 29794-4 quality-zones / foreground %.
#   ADD  ridge_frequency       -> NFIQ2 FDA / RPS (Lim et al. 2002). GATED: drop if it
#                                 collapses into clarity (|corr| >= 0.5 after training).
#   orientation_coherence is also GATED: drop if eta2_sensor stays > 0.2 after training.
# NOTE: the 6 OLD names ordered differently — do NOT re-score an OLD checkpoint with this
# list (columns would be mislabeled); use that run's archived scores CSV.
CONCEPTS: List[str] = [
    "ridge_valley_clarity",   # NFIQ2 LCS  (Chen 2005; Alonso-Fernandez 2007)
    "noise_level",            # Gabor-std smudge/noise (Shen, Kot & Koo 2001)
    "contrast_uniformity",    # NFIQ2 RVU
    "usable_area",            # ISO/IEC 29794-4 quality zones / foreground %
    "ridge_frequency",        # NFIQ2 FDA / RPS (gated)
    "orientation_coherence",  # NFIQ2 OCL  (Lim et al. 2002) (gated)
]

CONCEPT_INDEX: Dict[str, int] = {c: i for i, c in enumerate(CONCEPTS)}


# Corrected SIFQ_research_design.pdf Section 4.4: ONE degradation -> ONE primary concept
# (isolated grounding). A degradation may physically perturb other concepts, but that
# secondary cross-talk is NOT supervised here — it is MEASURED in Track 4. Every concept
# is grounded by >= 1 degradation; no degradation grounds two different concepts.
DEG_TO_TARGETS: Dict[DegradationType, List[Tuple[str, str]]] = {
    "blur":              [("ridge_valley_clarity", "decrease")],
    "jpeg":              [("ridge_valley_clarity", "decrease")],
    "pressure_wet":      [("ridge_valley_clarity", "decrease")],
    "noise":             [("noise_level", "increase")],
    "dry_skin":          [("contrast_uniformity", "decrease")],
    "occlusion":         [("usable_area", "decrease")],
    "ridge_stretch":     [("ridge_frequency", "decrease")],
    "orientation_break": [("orientation_coherence", "decrease")],
}


def level_to_target(level: int, direction: str) -> float:
    """Expected monotonic response in [0,1]. level in {0,1,2,3} (0 = clean)."""
    lv = float(max(0, min(3, int(level)))) / 3.0
    if direction == "increase":
        return lv
    return 1.0 - lv


def apply_degradation_pil(img: Image.Image, deg: DegradationType, level: int) -> Image.Image:
    """Apply a degradation type at intensity level 0..3. Level 0 returns original."""
    lvl = int(max(0, min(3, int(level))))
    if lvl == 0:
        return img

    if deg == "blur":
        # Pure sharpness loss for ridge_valley_clarity. The contrast-reduction term that
        # used to be here was REMOVED — it leaked blur into contrast_uniformity. Blur now
        # perturbs clarity only (in supervision); any residual cross-talk is measured.
        sigma = 0.8 + (6.0 - 0.8) * (lvl / 3.0)
        k = 2 * int(round(2.0 * sigma)) + 1
        t = TF.to_tensor(img)
        t = TF.gaussian_blur(t, kernel_size=[k, k], sigma=[sigma, sigma])
        return TF.to_pil_image(t)

    if deg == "noise":
        sigma = 5.0 + (30.0 - 5.0) * (lvl / 3.0)
        arr = np.array(img).astype(np.float32)
        arr = arr + np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
        sp = 0.01 + 0.03 * (lvl / 3.0)
        if sp > 0:
            salt   = np.random.rand(*arr.shape[:2]) < (sp * 0.5)
            pepper = np.random.rand(*arr.shape[:2]) < (sp * 0.5)
            arr[salt]   = 255.0
            arr[pepper] = 0.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    if deg == "jpeg":
        # Heavier compression + downscale so continuity / ridge_valley_clarity actually move
        # (Track-4 jpeg range was ~0.01 = effectively no response).
        q = int(round(80.0 + (6.0 - 80.0) * (lvl / 3.0)))
        q = max(4, min(95, q))
        scale = 1.0 - 0.40 * (lvl / 3.0)
        if scale < 1.0:
            w, h = img.size
            ds  = img.resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.BILINEAR)
            img = ds.resize((w, h), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    if deg == "occlusion":
        cov = 0.10 + (0.40 - 0.10) * (lvl / 3.0)
        w, h = img.size
        arr  = np.array(img).copy()
        mask = np.zeros((h, w), dtype=np.uint8)
        n_blobs = 1 + lvl
        for _ in range(n_blobs):
            bw = max(8, int(w * (0.12 + 0.10 * np.random.rand()) * (lvl / 3.0 + 0.4)))
            bh = max(8, int(h * (0.10 + 0.12 * np.random.rand()) * (lvl / 3.0 + 0.4)))
            x0 = random.randint(0, max(0, w - bw))
            y0 = random.randint(0, max(0, h - bh))
            mask[y0 : y0 + bh, x0 : x0 + bw] = 1

        if cov > 0:
            target_pixels = int(w * h * cov)
            current = int(mask.sum())
            # Fixed: bounded loop to prevent infinite iteration when mask is nearly full
            for _ in range(500):
                if current >= target_pixels:
                    break
                cx = random.randint(0, max(0, w - 1))
                cy = random.randint(0, max(0, h - 1))
                rw = random.randint(max(8, w // 16), max(10, w // 6))
                rh = random.randint(max(8, h // 16), max(10, h // 5))
                x1 = max(0, cx - rw // 2)
                x2 = min(w, cx + rw // 2)
                y1 = max(0, cy - rh // 2)
                y2 = min(h, cy + rh // 2)
                mask[y1:y2, x1:x2] = 1
                current = int(mask.sum())

        # Mid-grey fill: black (0) created fake high-contrast edges that leaked occlusion
        # into orientation_coherence / ridge_valley_clarity. usable_area = lost area only.
        arr[mask > 0] = 128
        return Image.fromarray(arr)

    if deg == "dry_skin":
        # contrast_uniformity degradation: impose a SMOOTH spatial contrast gradient so
        # part of the foreground loses contrast (dry skin / uneven pressure). The random
        # pixel dropout + stripes that used to be here were REMOVED — they fragmented
        # ridges and leaked into continuity. Pure (non-uniform) contrast now.
        t = TF.to_tensor(img)
        h, w = t.shape[1], t.shape[2]
        lo = 1.0 - 0.6 * (lvl / 3.0)                      # contrast at the faded side
        horizontal = random.random() < 0.5
        ramp = torch.linspace(1.0, lo, w if horizontal else h)
        ramp = ramp.view(1, 1, -1) if horizontal else ramp.view(1, -1, 1)
        mean = t.mean(dim=(1, 2), keepdim=True)
        t = (t - mean) * ramp + mean                     # local contrast scaling toward mean
        return TF.to_pil_image(t.clamp(0, 1))

    if deg == "orientation_break":
        # Rotate small patches independently to disrupt orientation coherence.
        # Each patch is rotated by a random angle; boundaries create orientation discontinuities.
        arr = np.array(img)
        h, w = arr.shape[:2]
        patch_size = max(10, 32 - 6 * lvl)          # 26, 20, 14 for lvl 1,2,3
        angle_range = 30.0 * lvl                      # ±30, ±60, ±90 degrees
        flip_prob   = 0.3 + 0.2 * lvl                # 0.5, 0.7, 0.9 probability per patch
        pil_src = img.copy()
        result  = img.copy()
        for y0 in range(0, h, patch_size):
            for x0 in range(0, w, patch_size):
                if random.random() > flip_prob:
                    continue
                y1, x1 = min(y0 + patch_size, h), min(x0 + patch_size, w)
                patch   = pil_src.crop((x0, y0, x1, y1))
                fill    = int(np.mean(np.array(patch)))
                rotated = patch.rotate(
                    random.uniform(-angle_range, angle_range),
                    fillcolor=fill, resample=Image.BILINEAR,
                )
                result.paste(rotated, (x0, y0))
        return result

    if deg == "ridge_stretch":
        # ridge_frequency degradation: anisotropic resample changes the ridge PERIOD
        # (spatial frequency / dominant FDA peak) along one axis. Compress one axis then
        # restore the original size -> ridge spacing shifts. Some interpolation softening
        # is unavoidable; the freq-vs-clarity gate decides if ridge_frequency holds a
        # separate identity from clarity (drop it if not).
        w, h = img.size
        factor = 1.0 + 0.45 * (lvl / 3.0)               # 1.15, 1.30, 1.45
        if random.random() < 0.5:
            nw, nh = max(8, int(round(w / factor))), h
        else:
            nw, nh = w, max(8, int(round(h / factor)))
        small = img.resize((nw, nh), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)

    # pressure_wet
    t = TF.to_tensor(img)
    g   = (0.2989 * t[0] + 0.5870 * t[1] + 0.1140 * t[2]).unsqueeze(0).unsqueeze(0)
    k   = 1 + 2 * lvl
    dil = F.max_pool2d(g, kernel_size=k, stride=1, padding=k // 2)
    blur_k = 3 + 2 * (lvl > 1)
    sm  = F.avg_pool2d(dil, kernel_size=blur_k, stride=1, padding=blur_k // 2)
    wet = 0.65 * dil + 0.35 * sm
    wet = torch.clamp(wet * (1.0 - 0.08 * lvl), 0, 1)
    out = torch.cat([wet[0], wet[0], wet[0]], dim=0)
    return TF.to_pil_image(out.clamp(0, 1))


def sample_degradation_pair(rng: random.Random) -> Tuple[DegradationType, int, int]:
    degs: List[DegradationType] = ["blur", "noise", "jpeg", "occlusion", "dry_skin", "pressure_wet", "orientation_break", "ridge_stretch"]
    deg = rng.choice(degs)
    i = rng.randint(1, 2)
    j = rng.randint(i + 1, 3)
    return deg, i, j
