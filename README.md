# Sinkhorn-Drifting Generative Models

Official code for the ImageNet-256 experiments of **Drifting with Sinkhorn Coupling**.

This repository extends [Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770) (Deng et al., 2026) with **Sinkhorn optimal transport couplings** for the drift loss, while fully aligning the architecture and training hyperparameters with the original paper's JAX release.

## Architecture: LightningDiT (aligned with Drifting)

The DiT implementation (`imagenet/dit.py`) is a PyTorch port of Drifting's **LightningDiT**, with identical components:

| Component | Drifting (JAX) | This repo (PyTorch) |
|-----------|---------------|-------------------|
| **Attention** | QK-Norm + RoPE | ✅ QK-Norm + RoPE |
| **FFN** | SwiGLU | ✅ SwiGLU |
| **Normalization** | RMSNorm | ✅ RMSNorm |
| **Pos Embed** | 2D sincos | ✅ 2D sincos |
| **Class Tokens** | 16 cls tokens | ✅ 16 cls tokens |
| **Conditioning** | class + noise + CFG | ✅ class + noise + CFG |
| **Noise Embed** | 64 classes × 32 coords | ✅ 64 × 32 |
| **CFG Embed** | TimestepEmbedder + RMSNorm × 0.02 | ✅ same |
| **AdaLN Zero Init** | ✅ | ✅ |

### Two Conditioning Modes

- **"drift" mode** (default): `cond = y_embed(y) + Σnoise_embed_i(nl_i) + cfg_norm(cfg_embed(cfg)) × 0.02`
- **"flow" mode** (backward compat): `cond = t_embed(t) + h_embed(h) + y_embed(y)`

## ImageNet-256 Training

### Environment Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install pytorch-fid diffusers tensorboard numpy scipy tqdm einops
```

### Data Preparation

Download ImageNet-1k and precompute VAE latents. The dataset should be organized as a `.npy` file:

```
/path/to/imagenet256/cache_full/
├── train_latents.npy    # (N, 4, 32, 32) float32
├── train_labels.npy     # (N,) int64
├── meta.json
```

Copy the numpy cache from the official Drifting JAX release or encode with SD-VAE.

### Training: Ablation (DiT-B/2, ~134M params, 30k steps)

This matches Drifting's ablation setting (`configs/gen/latent_ablation.yaml`):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
HF_ENDPOINT=https://hf-mirror.com \
torchrun --standalone --nproc_per_node=4 \
    -m imagenet.train_drifting \
    --cache-dir /path/to/imagenet256/cache_full \
    --backbone dit_b_2 \
    --coupling sinkhorn \
    --drift-form split \
    --sinkhorn-iters 20 \
    --sinkhorn-marginal weighted_cols \
    --temps 0.02 0.05 0.2 \
    --nneg 64 --npos 64 --nuncond 16 \
    --classes-per-step 64 \
    --epochs 96 --lr 2e-4 --weight-decay 0.01 --warmup-steps 5000 --ema-decay 0.999 \
    --noise-classes 64 --noise-coords 32 \
    --feature-mode encoder --encoder-every-n-blocks 2 --gen-mode direct \
    --run-name my_drift_exp \
    --sample-every-epochs 0.5 --decode-rgb-every-epochs 0.5 \
    --fid-every-epochs 5 --fid-num-samples 10000 \
    --fid-ref-stats /path/to/imagenet_256_fid_stats.npz \
    --tensorboard
```

**Training hyperparameters (aligned with Drifting ablation):**

| Parameter | Value | Equivalent in Drifting |
|-----------|-------|----------------------|
| `--classes-per-step` | 64 | `train_batch_size: 64` |
| `--nneg` | 64 | `gen_per_label: 64` |
| `--npos` | 64 | `pos_per_sample: 64` |
| `--nuncond` | 16 | `neg_per_sample: 16` |
| `--lr` | 2e-4 | `learning_rate: 0.0002` |
| `--weight-decay` | 0.01 | `weight_decay: 0.01` |
| `--warmup-steps` | 5000 | `warmup_steps: 5000` |
| `--ema-decay` | 0.999 | `ema_decay: 0.999` |
| `--temps` | 0.02, 0.05, 0.2 | `R_list: [0.02, 0.05, 0.2]` |
| `--epochs 96` | ~30048 steps | `total_steps: 30000` |
| `--noise-classes 64 --noise-coords 32` | noise diversity | `noise_classes: 64, noise_coords: 32` |

### Training: SOTA-L (~310M params, 200k steps)

To match Drifting's SOTA-L setting (`configs/gen/latent_sota_L.yaml`), use `dit_l_2` backbone:

```bash
# 4× GPU (adjust classes-per-step for your GPU count)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
HF_ENDPOINT=https://hf-mirror.com \
torchrun --standalone --nproc_per_node=4 \
    -m imagenet.train_drifting \
    --cache-dir /path/to/imagenet256/cache_full \
    --backbone dit_l_2 \
    --coupling sinkhorn \
    --drift-form split \
    --sinkhorn-iters 20 \
    --sinkhorn-marginal weighted_cols \
    --temps 0.02 0.05 0.2 \
    --nneg 64 --npos 64 --nuncond 32 \
    --classes-per-step 32 \
    --epochs 640 --lr 4e-4 --weight-decay 0.01 --warmup-steps 10000 --ema-decay 0.999 \
    --noise-classes 64 --noise-coords 32 \
    --feature-mode encoder --encoder-every-n-blocks 2 --gen-mode direct \
    --run-name my_sotaL_exp \
    --sample-every-epochs 1 --decode-rgb-every-epochs 1 \
    --fid-every-epochs 5 --fid-num-samples 10000 \
    --fid-ref-stats /path/to/imagenet_256_fid_stats.npz \
    --tensorboard
```

**SOTA-L hyperparameters (aligned with Drifting SOTA-L):**

| Parameter | Value | Equivalent in Drifting SOTA-L |
|-----------|-------|------------------------------|
| `--backbone` | `dit_l_2` | `hidden_size: 1024, depth: 24` |
| `--classes-per-step` | 32 | `train_batch_size: 128` (adjust for GPU count) |
| `--nneg` | 64 | `gen_per_label: 64` |
| `--npos` | 64 | `pos_per_sample: 64` |
| `--nuncond` | 32 | `neg_per_sample: 32` |
| `--lr` | 4e-4 | `learning_rate: 0.0004` |
| `--warmup-steps` | 10000 | `warmup_steps: 10000` |
| `--epochs 640` | ~200,000 steps | `total_steps: 200000` |
| `--fid-every-epochs 5` | eval every 5 epochs | `eval_per_step: ~1562` |

### Model Size Comparison

| Model | Params | Depth | Hidden | Heads | Tokens |
|-------|--------|-------|--------|-------|--------|
| DiT-S/2 | ~33M | 12 | 384 | 6 | 256 |
| **DiT-B/2** (ablation) | **~134M** | **12** | **768** | **12** | **256** |
| **DiT-L/2** (SOTA) | **~310M** | **24** | **1024** | **16** | **256** |
| DiT-XL/2 | ~675M | 28 | 1152 | 16 | 256 |

### CFG Sampling (Drifting-aligned)

The training uses **uniform CFG sampling** (matching Drifting), not power-law:

```python
# During each training step:
# 50%: cfg_scale = 1.0 (unconditional, dropout)
# 50%: cfg_scale ~ Uniform[1.0, 4.0]
_raw = 1.0 + torch.rand(classes_per_step) * 3.0
_drop = torch.rand(classes_per_step) < 0.5
omegas = torch.where(_drop, torch.ones_like(_raw), _raw)
```

This replaces the previous power-law sampling (`sample_power_law_omega` with exponent=3.0).

### Logging and Monitoring

- **TensorBoard**: Metrics (loss, grad_norm, FID) + generated RGB samples
- **Periodic samples**: `samples_rgb_epochN_stepM.png` — real vs generated side-by-side
- **Checkpoints**: Saved every epoch in `runs/imagenet256_drift/<run_name>/`

### FID Evaluation

Requires `pytorch-fid` and a reference `.npz` file:

```bash
pip install pytorch-fid
# Download reference: https://drive.google.com/drive/folders/1Tr_6PXF2WMYkSlCbbkP_0FRhEjAXx5gb
```

The training script evaluates FID periodically (set by `--fid-every-epochs`) and at the end.

---

## Sinkhorn-specific Improvements

Beyond Drifting's baseline, this codebase adds:

| Improvement | Description |
|-------------|-------------|
| **Sinkhorn coupling** | Full optimal transport balancing (vs. partial two-sided heuristic) |
| **Split drift form** | Decoupled positive/negative couplings (vs. joint) |
| **Unconditional negatives** | Real uniform random samples as negatives (vs. memory bank) |
| **Weighted column marginals** | CFG weighting via sinkhorn marginals (vs. kernel bias) |
| **Configurable coupling** | `partial_two_sided`, `row`, or `sinkhorn` |
| **Multi-temperature aggregation** | Configurable ρ values, feature normalization |

---

## Codebase Structure

```
imagenet/                  # ImageNet-256 experiments (main focus)
├── dit.py                 # LightningDiT (drifting-aligned architecture)
├── train_drifting.py      # Training loop + sinkhorn drift loss
├── eval_fid.py            # FID evaluation (InceptionV3)
├── latent_encoder.py      # ConvNet feature encoder (Appendix A.5)
├── conv_generator.py      # Conv baseline generator
├── flow_gen.py            # Flow-mode ablation
├── cache_latents.py       # VAE latent cache builder
├── models.py              # Model definitions
├── sample_decode.py       # RGB decoding utilities
└── unet.py                # UNet baseline
core/                      # Core drift loss implementations
├── drifting_loss.py       # Baseline + Sinkhorn drift loss
└── models/ema.py          # EMA helper
scripts/
├── train_sinkhorn_aligned.sh  # Launch script for aligned training
├── train_drift_80ep_5kipe.sh  # Original drift training
└── train_flow_80ep_5kipe.sh   # Flow ablation training
toy/                       # Toy experiments (Section 5.2)
mnist/                     # MNIST experiments (Section 5.3)
ffhq/                      # FFHQ experiments (Section 5.4)
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{deng2026generative,
  title={Generative Modeling via Drifting},
  author={Deng, Mingyang and Li, He and Li, Tianhong and Du, Yilun and He, Kaiming},
  journal={arXiv preprint arXiv:2602.04770},
  year={2026}
}
```

---

## Acknowledgements

This codebase builds on the official [Drifting JAX release](https://github.com/lambertae/drifting) and extends it with Sinkhorn optimal transport couplings.
