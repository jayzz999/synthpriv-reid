# SynthPriv-ReID

**Differentially Private Generative Framework for Synthetic Person Re-Identification**

## Problem Statement

Person Re-Identification (Re-ID) systems are critical for intelligent video surveillance, but training them requires large datasets of real pedestrian images — data that carries serious privacy risks. Identities captured in surveillance footage are sensitive personal information protected under regulations such as GDPR (EU), PDPL (Saudi Arabia), and CCPA (California).

**SynthPriv-ReID** solves this by training a GAN with formal Differential Privacy (DP) guarantees, producing synthetic pedestrian images that are provably safe to distribute. No individual from the original training set (e.g., Market-1501) can be recovered or identified from the synthetic outputs, with mathematically bounded privacy loss (ε, δ).

## How It Works

```
┌─────────────┐     ┌─────────────────────┐     ┌──────────────┐
│  Market-1501 │────►│  DP-GAN Training    │────►│  Synthetic   │
│  (Real Data) │     │                     │     │  Dataset     │
│              │     │  Discriminator:     │     │  (ε-private) │
│  Private,    │     │    DP-SGD via Opacus│     │              │
│  Regulated   │     │    • Gradient clip  │     │  Safe to     │
│              │     │    • Noise inject   │     │  share and   │
└─────────────┘     │                     │     │  publish     │
                    │  Generator:         │     └──────────────┘
                    │    Standard backprop │              │
                    └─────────────────────┘              ▼
                                                 ┌──────────────┐
                                                 │  Downstream  │
                                                 │  Re-ID Model │
                                                 │  (ResNet-50) │
                                                 └──────────────┘
```

### Privacy Mechanism

The Discriminator — the only component that touches real data — is trained with **DP-SGD** (Differentially Private Stochastic Gradient Descent):

1. **Per-sample gradient clipping**: Each training sample's gradient is clipped to a maximum L2 norm, bounding the influence of any single individual.
2. **Calibrated noise injection**: Gaussian noise (scaled by the noise multiplier) is added to the aggregated gradient before the parameter update.
3. **Privacy accounting**: An RDP (Rényi Differential Privacy) accountant tracks the cumulative privacy loss (ε) across all training steps.

The Generator never sees real data — it only receives gradient signals through the (privatized) Discriminator — so it inherits the same DP guarantee by the post-processing theorem.

## Project Structure

```
synthpriv_reid/
├── __init__.py          # Package marker
├── data_pipeline.py     # Market-1501 loader + dummy data for testing
├── models.py            # Generator (transposed CNN) + Discriminator (CNN)
├── privacy_engine.py    # Opacus DP integration, validation, budget tracking
├── train.py             # Full training loop with privacy budget monitoring
├── evaluate.py          # Synthetic image generation + FID computation
├── reid_utility.py      # Downstream Re-ID utility validation (ResNet-50)
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Package build configuration
└── README.md            # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Structural Test (Dummy Data)

Run a short training loop with random tensors to verify the full pipeline:

```bash
python -m synthpriv_reid.train --use-dummy --epochs 5 --batch-size 32 --output-dir output
```

### 3. Train on Market-1501

Download [Market-1501](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html) and point to it:

```bash
python -m synthpriv_reid.train \
  --dataset-root /path/to/Market-1501-v15.09.15 \
  --epochs 50 \
  --batch-size 64 \
  --noise-multiplier 1.1 \
  --max-grad-norm 1.0 \
  --target-epsilon 8.0 \
  --output-dir output
```

### 4. Generate Synthetic Images

```bash
python -m synthpriv_reid.evaluate \
  --checkpoint output/checkpoint_final.pt \
  --num-images 64 \
  --output-dir output
```

### 5. Compute FID

```bash
python -m synthpriv_reid.evaluate \
  --checkpoint output/checkpoint_final.pt \
  --compute-fid \
  --real-dir /path/to/Market-1501-v15.09.15/bounding_box_train \
  --fid-num-images 1000 \
  --output-dir output
```

### 6. Downstream Re-ID Utility Validation

Train a ResNet-50 Re-ID model on DP-synthetic data, then evaluate on real Market-1501:

```bash
python -m synthpriv_reid.reid_utility \
  --checkpoint output/checkpoint_final.pt \
  --dataset-root /path/to/Market-1501-v15.09.15 \
  --num-identities 100 \
  --images-per-id 50 \
  --epochs 20 \
  --output-dir output
```

This produces Rank-1 accuracy, Rank-5, Rank-10, and mAP scores that quantify the privacy-utility trade-off.

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--noise-multiplier` | 1.1 | Noise std relative to sensitivity. Higher → more private, lower image quality |
| `--max-grad-norm` | 1.0 | L2 clipping bound for per-sample gradients |
| `--target-epsilon` | 8.0 | Privacy budget cap. Training stops if exceeded |
| `--target-delta` | 1e-5 | Probability of privacy guarantee failure |
| `--nz` | 128 | Latent vector dimensionality |
| `--batch-size` | 64 | Training batch size (affects privacy cost per step) |

### Privacy–Utility Trade-off

- **Lower ε** (stronger privacy): increase `noise-multiplier`, decrease `epochs` or `batch-size`.
- **Higher image quality**: decrease `noise-multiplier` (weaker privacy) or increase model capacity (`ngf`, `ndf`).
- Typical publishable DP guarantee: ε < 10 with δ = 1/N.

## Design Decisions

**Why GroupNorm instead of BatchNorm?** Opacus requires per-sample gradient computation. BatchNorm computes statistics across the batch, making each sample's output depend on its peers — this violates the independence assumption needed for DP-SGD. GroupNorm operates per-sample.

**Why only privatize the Discriminator?** The Generator never accesses real data directly. By the post-processing theorem of differential privacy, any function of a DP output is also DP. Since the Generator only receives gradients through the privatized Discriminator, it automatically satisfies the same (ε, δ)-DP guarantee.

**Why BCEWithLogitsLoss?** Opacus requires a single `loss.backward()` call per optimizer step. BCE with logits is numerically stable and compatible with this constraint (unlike Wasserstein loss with gradient penalty, which requires multiple backward passes).

## Relevance to Privacy-Preserving Surveillance Research

This framework directly addresses the core challenge in deploying Re-ID systems responsibly:

1. **Regulatory compliance**: Synthetic datasets generated under (ε, δ)-DP satisfy the "data minimization" and "purpose limitation" principles of GDPR Art. 5 and similar regulations.
2. **Dataset sharing**: Researchers can share DP-synthetic datasets publicly without re-identification risk, accelerating collaboration.
3. **Downstream utility**: The privacy–utility trade-off is quantified by training a standard Re-ID model (e.g., ResNet-50) on synthetic data and evaluating on real benchmarks.

## References

- Dwork, C., & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy*. Foundations and Trends in Theoretical Computer Science.
- Abadi, M., et al. (2016). *Deep Learning with Differential Privacy*. CCS '16.
- Opacus library: https://opacus.ai/
- Market-1501 dataset: Zheng, L., et al. (2015). *Scalable Person Re-identification: A Benchmark*. ICCV '15.
