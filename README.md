<div align="center">

# 🦜 BirdCLEF+ 2026 — EoS.7 (bz) + EfficientNet-B0 Branch

**Acoustic Species Identification in the Pantanal, South America**

*234 species · Macro ROC-AUC · Public LB **0.950***

[![Kaggle](https://img.shields.io/badge/Kaggle-BirdCLEF%202026-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/birdclef-2026)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX%20Runtime-1.24-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📌 Overview

This repository contains my iteration on the **BirdCLEF+ 2026** competition — an acoustic species-identification challenge over **234 bird species** of the Pantanal wetlands (South America). The task is to predict per-species presence probabilities for every 5-second window of 1-minute test soundscapes, scored by **macro ROC-AUC**.

The notebook in this repo, [`birdclef-2026-eos-7-bz-efficientnet-b0.ipynb`](./birdclef-2026-eos-7-bz-efficientnet-b0.ipynb), reaches **Public LB = 0.950** by combining a four-model ensemble (Perch v2 + ProtoSSM + Distilled-SED + Karnakbayev Power-Optimization) with a **custom 5-fold EfficientNet-B0 diversity branch** that I trained from scratch on 35,549 Xeno-Canto + iNaturalist clips (CV macro-AUC **0.9564**), plus a **taxonomy-aware smoothing post-processor** (genus α=0.15, class α=0.05).

> **My personal contribution:** the entire EfficientNet-B0 5-fold branch (Cell 33), its training pipeline, ONNX export, species-mapping logic, and probability-blend integration. All other ensemble members are properly credited to the original community authors below.

---

## 🏆 Results

| Submission | Public LB |
|---|---|
| Baseline (EoS.7 bz ensemble + TAX_SMOOTHING) | **0.950** |
| My EfficientNet-B0 iteration on EoS.6 bz | 0.949 |

**EfficientNet-B0 5-fold cross-validation (my training, 35,549 clips):**

| Fold | Val macro-AUC |
|:----:|:-------------:|
| 0 | 0.9562 |
| 1 | 0.9585 |
| 2 | 0.9540 |
| 3 | 0.9525 |
| 4 | **0.9610** |
| **Mean** | **0.9564** |

---

## 🏗️ Architecture

The pipeline is a **multi-branch heterogeneous ensemble** orchestrated by a master `solutions` dictionary (Cell 2). Each branch is a stand-alone inference graph; outputs are fused via weighted blending plus a final taxonomy-aware smoothing pass.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       BirdCLEF+ 2026  · 60s soundscape (.ogg)               │
│                       12 windows × 5s · SR=32 kHz · 234 classes             │
└────────────────────────────────────────────────────────────────────────────┘
                                       │
        ┌──────────────────┬───────────┴───────────┬──────────────────┐
        ▼                  ▼                       ▼                  ▼
 ┌──────────────┐   ┌──────────────┐       ┌──────────────┐   ┌──────────────┐
 │  Model_22    │   │  Model_51    │       │  Model_74    │   │  Model_1     │
 │ ProtoSSMv5   │   │ Karnakbayev  │       │ Karnakbayev  │   │ Distilled    │
 │ + Perch v2   │   │ PowerOpt     │       │ PowerOpt(74) │   │ SED EffB0    │
 │ (TF/ONNX)    │   │ LB 0.948     │       │ LB 0.948     │   │ (PyTorch)    │
 │  LB 0.928    │   │ xSED 0.6/0.4 │       │ xSED 0.6/0.4 │   │  LB 0.917    │
 └──────┬───────┘   └──────┬───────┘       └──────┬───────┘   └──────────────┘
        │ w=0.022          │ w=0.9695            │ w=0.0085
        │                  │                     │ (PSSM saved as subm_74p)
        └──────────────────┴─────────────────────┘
                                  │
                                  ▼
                  ┌──────────────────────────────┐
                  │   TAX_SMOOTHING POSTPROC     │  ← v221 (genus α=0.15,
                  │   genus + class group means  │     class α=0.05)
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │   submission.csv  (anchor)   │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │   🦅 EfficientNet-B0 5-Fold Branch   │  ← MY CONTRIBUTION
              │   (Cell 33 · ONNX · weight 0.05)      │     (Cell 32–33)
              │   Trained on 35,549 XC+iNat clips     │
              │   CV macro-AUC 0.9564                 │
              └──────────────┬───────────────────────┘
                             │
                             ▼
                  ┌──────────────────────────────┐
                  │       Final submission.csv  │  → Public LB 0.950
                  └──────────────────────────────┘
```

### Ensemble configuration (Cell 2, version 11)

```python
solutions = {
  'type_add' : 'TAX_SMOOTHING',
  'task1'    : 'run SED once',
  'task2'    : {'save PSSM for Model_74 as': 'subm_74p.csv'},
  'Models'   : [
    {'Model': 'Model_22', 'subm': 'subm_22.csv',  'weight': 0.022,  'xSED': [],          'LB': '0.928'},
    {'Model': 'Model_74', 'subm': 'subm_74p.csv', 'weight': 0.0085, 'xSED': [],          'LB': '0.949'},
    {'Model': 'Model_51', 'subm': 'subm_51.csv',  'weight': 0.9695, 'xSED': [0.60,0.40], 'LB': '0.949'},
  ]
}
```

---

## 🔬 Notebook Walkthrough (34 cells)

### Cell 0 — Title banner
Stylised HTML header identifying the iteration, baseline LB, and the EfficientNet contribution.

### Cell 1 — Competition + model catalogue
Markdown table cataloguing every public model referenced by ID (Model_1 / 21 / 22 / 51 / 52 / 73 / 74) with original Kaggle authors, LB scores, and blend recipes. Documents the **single-line patch** taken from the Japanese expert *swordsman* to the Model_51 / Model_74 post-processing:

```
before:  proto_cont = (xctx > 0.88) & (rank_proto > 0.75) & (p_sed < 0.12) & (~fake_only)
after :  proto_cont = (xctx > 0.88) & (rank_proto > 0.77) & (p_sed < 0.14) & (~fake_only)
```

### Cell 2 — Master configuration
`solutions` dict (ver.11): three-way blend of Model_22 (ProtoSSM, w=0.022), Model_74 (post-PSSM, w=0.0085) and Model_51 (Karnakbayev, w=0.9695) with `TAX_SMOOTHING` as the additive type. Model_51 receives the cross-SED schema `[0.60, 0.40]`.

### Cell 3 — Solution unpacking
Flattens the `solutions` dict into parallel lists (`_ensemble_models`, `_files_subm`, `_weights`, `_xsed`, `_lbs`) and resolves the one-time SED flag.

### Cells 4–5 — Model_1: Distilled-SED (Tucker Arrants, LB 0.917)
*Skipped under ver.11* — but fully wired for the "run SED once" task. Architecture:
- **Backbone:** `tf_efficientnet_b0.ns_jft_in1k` (timm), 256-mel spectrograms, n_fft=2048, hop=512, fmin=20, fmax=16000.
- **GeM frequency pooling** with learnable p (init 3.0) — sharpens focus on bird-vocalisation bands.
- **Attention bottleneck:** 512-dim dense → 1D conv → frame-wise → clip-level aggregation.
- **Perch v2 distillation head:** GAP + linear projection into the frozen 1536-dim Perch embedding space (MSE loss, α=1.0).
- **Augmentation:** gain jitter ±6 dB, noise SNR 10–30 dB, SpecAugment (GPU), Focal-Focal + Focal-Soundscape MixUp (β=0.4), per-class minimum sample = 20.
- **Training:** 5 folds, 25 epochs, AdamW lr=5e-4, cosine + 2-epoch warmup, batch 64.

### Cells 6–7 — Model_21: Perch v2 + ProtoSSM (yukiZ, LB 0.928)
Installs ONNX Runtime 1.24 + TensorFlow 2.20 from bundled wheels. CPU-only by competition constraint. Operates on the 1536-d Perch v2 embeddings extracted from 12 × 5-s windows per file.

### Cells 8–9 — Model_22: ProtoSSMv5 + Cross-Attention (yukiZ, LB 0.928)
Same Perch-embedding feature path as Model_21 but with the V18 sequential upgrade:
- **Metadata injection:** `site_id` + `hour_utc` → 24-dim → added to embeddings.
- **Selective SSM (Mamba-style):** 4-layer bidirectional, d_model=320, d_state=32.
- **Temporal cross-attention:** 8-head, captures long-range (dawn-chorus / counter-call) interactions across the full minute.
- **Prototypical classification:** temperature-scaled cosine similarity vs learnable class prototypes (not a linear head).
- **MLP / LightGBM probes:** class-specific stacking on the sequence embeddings.

### Cells 10–11 — Model_5 explanation excerpt
Documents the **exp019 / exp017** scalar-probe sequence used by Sun-Derek-kiz to land on LB 0.948 → 0.949:
- `apply_prior(λ_prior)`: 0.4 → 0.5 (exp017)
- `rank_aware_scaling(power)`: 0.5 → 0.6 (exp019)
- Single-scalar-at-a-time methodology, fully attributable.

### Cells 12–13 — Model_51: Karnakbayev Power-Optimization (LB 0.948 → 0.949)
Embeds the entire `Karnakbayev_PowerOptimization_LB0948` pipeline. Writes `subm_karnakbayev_power_optimization.csv` and, via cross-SED `[0.60, 0.40]`, also produces `subm_51.csv`.

### Cells 14–15 — Model_52: same code path, different xSED source
Identical to Model_51 but pulls its `xSED` schedule from `solutions['Models'][1]` instead of `[2]`. Provides the alternative blending lane used in earlier versions.

### Cells 16–17 — Model_7 explanation excerpt
Discusses the **dual-pipeline architecture** (SED + ProtoSSMv5) and the `exp_067 / exp_066_v6_prior060` scalar-probe series; `lambda_prior` 0.60 → 0.65.

### Cells 18–19 — Model_73: Karnakbayev variant
Same code as Model_51/52 but writes `subm_karnakbayev_power_optimization_73.csv`. Uses `xSED` from `solutions['Models'][1]`.

### Cells 20–21 — Model_74: Karnakbayev variant (active in ver.11)
Same code as Model_73 but uses `xSED` from `solutions['Models'][2]`. **This is the model whose PSSM is captured and re-emitted as `subm_74p.csv` per `solutions['task2']`.**

### Cells 22–24 — Division-attention bookkeeping
Reads `sample_submission.csv`, partitions the 234 species into two halves by name (`'son' not in label`), prints the cut: `half_1 = 117, half_2 = 117`. Sets up the split-weighting scheme used by `division_attention`.

### Cell 25 — `## blend` section header

### Cell 26 — Direct-add & rank-based blends
Defines `direct_add2 / 3 / 4` (weighted linear blend) and `rank_1_add2 / 3` (rank-percentile blend with ε-clipping and row_id reconciliation). All operate on the per-model `subm_*.csv` outputs.

### Cell 27 — `division_attention()`
The active blender for the `division_attention` mode. Three submissions are merged on `row_id`; each species column is reweighted by `sub_w1 = [0.014, 0.021, 0.965]` if `i < 117` (half_1) else `sub_w2 = [0.0137, 0.0213, 0.965]` (half_2).

### Cell 28 — `f_TAX_SMOOTHING_POSTPROC` — the v221 LB-0.950 lever
Post-processor that contributed the final jump from 0.949 → 0.950:

1. Calls the underlying additive blender (`func_add`, e.g. `direct` / `division_attention`).
2. Loads `taxonomy.csv`, builds species→genus and species→class maps.
3. For each multi-member genus group: `probs ← (1−α_genus) · probs + α_genus · genus_mean` with **α_genus = 0.15**.
4. For each multi-member class group: same with **α_class = 0.05**.
5. Returns the smoothed DataFrame.

This pulls correlated siblings toward their group mean — a Bayesian-style shrinkage that reduces single-class overconfidence and consistently adds ≈+0.001 macro-AUC on this dataset.

### Cell 29 — Dispatcher
Selects `f_add` based on `solutions['type_add']`:
- `'rank.1'` → `rank_1`
- `'direct'` → `direct`
- `'division_attention'` → `division_attention`
- `'TAX_SMOOTHING'` → `f_TAX_SMOOTHING_POSTPROC` ✅ (active)

### Cell 30 — `## Submit`

### Cell 31 — Write submission
```python
submission = f_add()
submission.to_csv("submission.csv", index=True)
```
This anchor file is what the EfficientNet branch in Cell 33 then blends against.

---

### 🦅 Cells 32–33 — **My contribution: EfficientNet-B0 5-fold branch**

This is the entire iteration I added to the public EoS.7 baseline.

#### Training pipeline (run offline, weights shipped as ONNX)

| | |
|---|---|
| **Backbone** | `tf_efficientnet_b0.ns_jft_in1k` (timm) |
| **Input** | 5-second 32 kHz audio → mel-spectrogram (128 mels) → resized to 224×224 |
| **Dataset** | 35,549 clips from Xeno-Canto + iNaturalist, 234 target species |
| **Augmentation** | MixUp + SpecAugment (frequency / time masking) |
| **Optimiser** | AdamW, lr=1e-3, weight-decay 1e-4 |
| **Schedule** | Cosine annealing + 3-epoch warm-up |
| **Epochs** | 30 with early stopping (patience 8) |
| **Splits** | 5-fold stratified (by primary_label) |

5-fold validation reported in the *Results* section above (mean macro-AUC **0.9564**).

#### Inference integration (Cell 33)

1. **Auto-discovery** — locates the ONNX bundle at `/kaggle/input/birdclef-2026-anuj-efficientnet-b0-onnx/` (falls back to a recursive glob).
2. **Species alignment** — reads `species.json` shipped alongside the ONNX weights, computes a permutation to match `sample_submission.csv` column ordering exactly (234 classes).
3. **Fold selection** — knob `EFFNET_USE_ALL_FOLDS`; defaults to **fold 4 only** (best CV = 0.9610) for runtime safety since the full notebook already runs ≈80–90 min on hidden test. Flip to `True` to enable the full 5-fold average when wall-time permits.
4. **Window-wise inference** — soundfile + librosa → 5-s mel → ONNX Runtime → sigmoid probabilities for every 5-second window of every test soundscape.
5. **Probability blend** with the anchor `submission.csv`:
   ```
   final = (1 − EFFNET_WEIGHT) · anchor + EFFNET_WEIGHT · effnet
   ```
   with **`EFFNET_WEIGHT = 0.05`** — conservative on a 0.95 anchor.
6. **Dry-run safety** — if `sample_submission.csv` has only the 3 placeholder rows (Kaggle dry-run), the branch *skips* the blend and leaves the anchor untouched. The hidden-test run has matching schemas and the blend applies.
7. **Overwrites** `submission.csv` in place.

---

## 🚀 How to Reproduce

The notebook is designed to run end-to-end on a **Kaggle T4×2 / CPU** kernel with the BirdCLEF 2026 competition data attached. Locally, you need:

```bash
# 1. Clone
git clone https://github.com/anujdevsingh/birdclef-2026-model-and-analysis.git
cd birdclef-2026-model-and-analysis

# 2. Environment (Python 3.12 recommended)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch torchaudio timm onnxruntime tensorflow==2.20 \
            pandas numpy scikit-learn scipy soundfile librosa lightgbm tqdm

# 3. Download competition data from Kaggle into /kaggle/input/competitions/birdclef-2026
#    (or adjust the COMP_DIR paths inside the notebook)

# 4. Launch
jupyter lab birdclef-2026-eos-7-bz-efficientnet-b0.ipynb
```

**Required Kaggle datasets** (attach when running on Kaggle):
- `competitions/birdclef-2026`
- `datasets/tuckerarrants/perch-v2-no-dft-onnx`
- `datasets/tuckerarrants/birdclef-2026-waveform-cache`
- `datasets/rishikeshjani/perch-onnx-for-birdclef-2026`
- `notebooks/ashok205/tf-wheels`
- `models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1`
- `datasets/anujdevsingh/birdclef-2026-anuj-efficientnet-b0-onnx` (my EfficientNet weights)

---

## 📂 Repository Layout

```
birdclef-2026-model-and-analysis/
├── birdclef-2026-eos-7-bz-efficientnet-b0.ipynb   # Main notebook (34 cells)
├── README.md                                       # This file
└── .gitattributes
```

---

## 🙏 Credits



 nina2025 
Tucker Arrants 


---

## 👤 Author

**Anuj Dev Singh** — Delhi, India
BS Data Science · IIT Madras

- 🌐 [codewithanuj.com](https://codewithanuj.com)
- 💻 [GitHub: @anujdevsingh](https://github.com/anujdevsingh)
- 📧 anujdev9928@gmail.com

---

## 📄 License

Released under the **MIT License**. The original community model code retained in cells 4–21 is the property of its respective authors and is included under the terms of their public Kaggle notebook licenses (Apache 2.0 / open-source). The EfficientNet-B0 branch (Cells 32–33) and this README are © 2026 Anuj Dev Singh.

---

<div align="center">

*If this work helps you, a ⭐ on the repo and an upvote on the [Kaggle notebook](https://www.kaggle.com/competitions/birdclef-2026) are deeply appreciated.*

</div>
