"""FID evaluation for ImageNet-256 latent generators (EMA, VAE decode, Inception)."""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable

import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm

try:
    from pytorch_fid.inception import InceptionV3
except ImportError as exc:
    raise ImportError(
        "Install pytorch-fid: PYTHONNOUSERSITE=1 pip install pytorch-fid"
    ) from exc


def load_ref_stats(path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        if "ref_mu" in data:
            return data["ref_mu"], data["ref_sigma"]
        if "mu" in data:
            return data["mu"], data["sigma"]
        raise KeyError(f"Expected mu/sigma or ref_mu/ref_sigma in {path}")


@torch.no_grad()
def generate_rgb_batch(
    gen: torch.nn.Module,
    labels: torch.Tensor,
    *,
    backbone: str,
    noise_dim: int,
    device: torch.device,
    gen_mode: str = "direct",
    flow_steps: int = 1,
    flow_time_scale: float = 999.0,
    gen_timestep: int = 999,
    noise_classes: int = 0,
    noise_coords: int = 0,
    cfg_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    from imagenet.train_drifting import generate_latents

    n = labels.shape[0]
    noise_labels = torch.randint(0, max(1, noise_classes), (n, max(1, noise_coords)), device=device) if noise_classes > 0 else None
    return generate_latents(
        gen, n, labels, backbone=backbone, noise_dim=noise_dim, device=device,
        gen_mode=gen_mode, flow_steps=flow_steps, flow_time_scale=flow_time_scale,
        gen_timestep=gen_timestep,
        cfg_scale=cfg_scale or (torch.ones(n, device=device) if noise_classes > 0 else None),
        noise_labels=noise_labels,
        noise_classes=noise_classes, noise_coords=noise_coords,
    )


@torch.no_grad()
def decode_to_uint8(vae, latents: torch.Tensor) -> np.ndarray:
    z = latents / 0.18215
    imgs = vae.decode(z).sample
    imgs = (imgs.clamp(-1, 1) + 1) * 127.5
    imgs = imgs.round().clamp(0, 255).to(torch.uint8)
    return imgs.permute(0, 2, 3, 1).cpu().numpy()


def compute_activation_stats(
    images_nhwc: np.ndarray,
    model: InceptionV3,
    device: torch.device,
    batch_size: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    n = images_nhwc.shape[0]
    pred_arr = np.empty((n, 2048), dtype=np.float64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = images_nhwc[start:end].astype(np.float32) / 255.0
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
        pred = model(batch)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred_arr[start:end] = pred.squeeze(-1).squeeze(-1).cpu().numpy()
    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    return mu, sigma


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


@torch.no_grad()
def evaluate_fid(
    gen: torch.nn.Module,
    *,
    num_samples: int,
    num_classes: int,
    backbone: str,
    noise_dim: int,
    device: torch.device,
    vae_id: str,
    ref_stats_path: str,
    batch_size: int = 32,
    sample_omega: float = 2.0,
    gen_mode: str = "direct",
    flow_steps: int = 1,
    flow_time_scale: float = 999.0,
    gen_timestep: int = 999,
    noise_classes: int = 0,
    noise_coords: int = 0,
) -> float:
    from diffusers import AutoencoderKL

    if not os.path.isfile(ref_stats_path):
        raise FileNotFoundError(f"FID ref stats not found: {ref_stats_path}")

    ref_mu, ref_sigma = load_ref_stats(ref_stats_path)
    vae = AutoencoderKL.from_pretrained(vae_id).to(device)
    vae.eval()
    gen.eval()

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device)
    inception.eval()

    per_class = num_samples // num_classes
    remainder = num_samples - per_class * num_classes
    images = []
    for c in tqdm(range(num_classes), desc="fid/generate", leave=False):
        n_c = per_class + (1 if c < remainder else 0)
        if n_c == 0:
            continue
        labels = torch.full((n_c,), c, dtype=torch.long, device=device)
        latents = generate_rgb_batch(
            gen, labels, backbone=backbone, noise_dim=noise_dim, device=device,
            gen_mode=gen_mode, flow_steps=flow_steps, flow_time_scale=flow_time_scale,
            gen_timestep=gen_timestep,
            noise_classes=noise_classes, noise_coords=noise_coords,
            cfg_scale=torch.full((n_c,), sample_omega, device=device),
        )
        rgb = decode_to_uint8(vae, latents)
        images.append(rgb)
    all_imgs = np.concatenate(images, axis=0)[:num_samples]

    mu, sigma = compute_activation_stats(all_imgs, inception, device, batch_size=batch_size)
    return frechet_distance(mu, sigma, ref_mu, ref_sigma)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ref-stats", required=True, help="npz with mu/sigma or ref_mu/ref_sigma")
    p.add_argument("--num-samples", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--vae-id", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--gen-mode", default=None, choices=["flow", "direct"])
    p.add_argument("--flow-steps", type=int, default=None)
    p.add_argument("--flow-time-scale", type=float, default=None)
    p.add_argument("--gen-timestep", type=int, default=None, help="Only for direct mode")
    return p.parse_args()


def _cfg_gen_kw(gen_cfg: dict, ckpt_args: dict, cli: argparse.Namespace) -> dict[str, object]:
    args = ckpt_args or {}
    return {
        "gen_mode": cli.gen_mode or gen_cfg.get("gen_mode", args.get("gen_mode", "direct")),
        "flow_steps": int(cli.flow_steps if cli.flow_steps is not None else gen_cfg.get("flow_steps", args.get("flow_steps", 1))),
        "flow_time_scale": float(
            cli.flow_time_scale if cli.flow_time_scale is not None
            else gen_cfg.get("flow_time_scale", args.get("flow_time_scale", 999.0))
        ),
        "gen_timestep": int(
            cli.gen_timestep if cli.gen_timestep is not None
            else gen_cfg.get("gen_timestep", args.get("gen_timestep", 999))
        ),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    from imagenet.train_drifting import build_generator, load_ema_weights

    gen_cfg = ckpt.get("gen_config", {})
    ckpt_args = ckpt.get("args", {})
    backbone = gen_cfg.get("backbone", ckpt_args.get("backbone", "dit_b_4"))
    num_classes = gen_cfg.get("num_classes", 1000)
    noise_dim = gen_cfg.get("noise_dim", 128)
    hidden_dim = gen_cfg.get("hidden_dim", 512)
    gen_kw = _cfg_gen_kw(gen_cfg, ckpt_args, args)

    gen = build_generator(backbone, num_classes, hidden_dim, noise_dim).to(device)
    load_ema_weights(gen, ckpt)
    fid = evaluate_fid(
        gen,
        num_samples=args.num_samples,
        num_classes=num_classes,
        backbone=backbone,
        noise_dim=noise_dim,
        device=device,
        vae_id=args.vae_id,
        ref_stats_path=args.ref_stats,
        batch_size=args.batch_size,
        **gen_kw,
    )
    print(f"FID@ {args.num_samples}: {fid:.4f}")

    run_dir = os.path.dirname(os.path.abspath(args.ckpt))
    result_path = os.path.join(run_dir, "fid_results.json")
    records = []
    if os.path.isfile(result_path):
        with open(result_path) as f:
            records = json.load(f)
    records.append({
        "method": "sinkhorn_drift",
        "num_samples": args.num_samples,
        "fid": fid,
        "tag": "end",
        "ref_stats": args.ref_stats,
        "ckpt": args.ckpt,
    })
    with open(result_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved -> {result_path}")


if __name__ == "__main__":
    main()
