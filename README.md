# SIFQ — Sensor-Invariant Fingerprint Quality

A self-supervised, concept-grounded alternative to NFIQ2. SIFQ is a small CNN that predicts a
scalar fingerprint quality `Q` from **6 interpretable concepts** (a Concept Bottleneck Model),
trained **without any human quality labels**.

```
image → MobileNetV2 → embedding → concept head → 6 concepts → quality head → Q
```

`Q` is designed to measure **biometric utility**: it stays high across cosmetic sensor differences
and drops only when biometric content is genuinely destroyed.

## Why concept-bottleneck + self-supervised?
- **Interpretable.** `Q = quality_head(concepts)` is a strict bottleneck (≈ linear), so every score
  decomposes into 6 named factors.
- **No quality labels.** NFIQ2-style training needs human/utility labels. SIFQ learns `Q` from three
  label-free signals instead.

## The 6 concepts (v4)
| Concept | Decreases with |
|---|---|
| `ridge_valley_clarity` | blur, wet pressure, JPEG |
| `noise_level` *(inverted)* | sensor noise |
| `contrast_uniformity` | dry skin / uneven contrast |
| `usable_area` | occlusion / small capture area |
| `ridge_frequency` *(gated)* | ridge stretching / resolution loss |
| `orientation_coherence` *(gated)* | broken ridge flow |

## Training signals (all label-free)
- **L_mat — matcher-as-teacher.** Targets `q_mat = sigmoid(norm(FLaRE mean-genuine match score))`.
  Normalisation is a **blend** of global and per-sensor z-scores (`α·global + (1−α)·per-sensor`,
  α=0.5) so a *globally* bad sensor stays low instead of being rescaled to average.
- **L_sens — cross-sensor invariance.** A gradient-reversal (GRL) sensor adversary on the 64-d
  intermediate feature + a **quality-aware** paired margin (`L_pair`): same-finger captures are
  pushed to equal `Q` only as far as their teacher quality genuinely agrees.
- **L_deg — controlled synthetic degradation.** `Q(clean) > Q(degraded)` ranking + concept grounding
  (each degradation targets one concept).
- `L_ortho` (concept decorrelation) is always on. `L_id` (ArcFace on finger identity) gives the
  backbone real fingerprint understanding.

4-stage schedule: `id_warmup → add_q → add_invariance(GRL) → finetune`.
Model: MobileNetV2 (3-channel), 256-d embedding, **~2.7M params, ~20 ms / image on desktop CPU**.

## Results (NIST SD302 + FVC, 12-sensor cross-sensor test, 4010 images)

### Track 1 — Error-vs-Reject Curve (ERC): does Q improve a real matcher?
AUC_ERC (**lower = better**) — both SIFQ and NFIQ2 are deployable single-image quality predictors.

**NBIS Bozorth3 matcher** (genuine-vs-impostor ROC 0.766):

| Quality | FMR=0.1 | 0.01 | 0.001 | 1e-4 |
|---|---|---|---|---|
| **SIFQ v4** | **0.132** | **0.207** | **0.274** | **0.336** |
| NFIQ2 | 0.144 | 0.222 | 0.287 | 0.344 |

**VeriFinger matcher** (ROC 0.958):

| Quality | FMR=0.1 | 0.01 | 0.001 | 1e-4 |
|---|---|---|---|---|
| **SIFQ v4** | 0.009 | **0.012** | **0.015** | **0.020** |
| NFIQ2 | 0.007 | 0.013 | 0.021 | 0.033 |

→ **SIFQ ≥ NFIQ2 at every operating point on Bozorth3, and at strict FMR on VeriFinger.**

### Track 2 — Sensor invariance
Mean cross-sensor KS distance between per-sensor `Q` distributions (**lower = more invariant**):

| | all sensor pairs | excluding sensor H |
|---|---|---|
| **SIFQ v4** | **0.266** | **0.140** |
| NFIQ2 | 0.686 | 0.625 |

SIFQ's `Q` is far more sensor-invariant than NFIQ2. Sensor **H** is *genuinely* low-quality on every
matcher, so SIFQ's quality-aware design correctly lets H differ — hence the gap between the two columns.

Full ERC reports, KS matrices and per-sensor plots are in [`results/`](results/).

## Repository layout
```
code/        model + training + scoring + evaluation
  sifq/                    model.py, losses.py, degradations.py, grl.py
  train_sifq.py            4-stage training
  score_manifest_sifq.py   score a manifest → q_hat + 6 concepts (+ embeddings)
  make_flare_qmat.py       build FLaRE teacher targets (or --renorm-from to re-normalise)
  report.py                sensor / KS / train-log reports
  evaluate/                Track 1–4 + Bozorth/VeriFinger ERC harness
weights/     stage4_finetune_final.pt (model of record) + sensor_map.json + train_log.jsonl
results/     teacher targets, per-image scores, ERC reports, Track-2 invariance
```

## Setup & usage
The datasets, the matcher (VeriFinger) and the teacher (FLaRE) are **not** included — bring your own.

```bash
pip install -r requirements.txt
```

Score a manifest (CSV with `path,sensor,identity,roll,split`):

```bash
python code/score_manifest_sifq.py \
  --ckpt weights/stage4_finetune_final.pt \
  --manifest path/to/manifest.csv --split test \
  --sensor-map weights/sensor_map.json \
  --output-csv scores.csv
```

Re-train from the provided teacher targets (no FLaRE re-run needed):

```bash
python code/train_sifq.py --manifest path/to/manifest.csv \
  --qmat results/flare_qmat_train_fvc_blend.csv ...
```

`results/flare_qmat_train_fvc_blend.csv` is the teacher target file used to train the released model.

## Honest caveats
- **FLaRE is a third-party teacher and is *not* bundled.** It is a research dependency: install it
  yourself to regenerate `q_mat`. The provided teacher CSV lets you retrain without it.
- An earlier DeepPrint embedding teacher was found to **encode sensor, not finger identity**, and was
  dropped; the matchers of record are classical minutiae (NBIS Bozorth3) and VeriFinger.
- **Small data** (~140 training subjects) and a **synthetic→natural degradation gap**: concept
  grounding is validated on synthetic degradations, so treat absolute concept values with care.
- All results are on NIST SD302 + FVC.

## Data & licensing
- **Datasets** (NIST SD302, FVC) are not redistributed here — obtain them from their official sources
  under their own licenses.
- **VeriFinger** (Neurotec) and **FLaRE** are third-party and not included.
- Code in this repository is released under the terms in [`LICENSE`](LICENSE).
