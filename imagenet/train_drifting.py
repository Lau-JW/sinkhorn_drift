"""Train Sinkhorn drifting generator on ImageNet-256 VAE latents."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

from core.drifting_loss import (
    drifting_loss_for_feature_set,
    drifting_loss_over_feature_sets,
    feature_sets_from_latents,
    flatten_latents_as_feature_set,
    sample_power_law_omega,
)
from core.models.ema import EMA
from imagenet.conv_generator import LatentConvGenerator
from imagenet.dit import dit_b_2, dit_b_4, dit_s_2
from imagenet.latent_encoder import build_latent_encoder
from imagenet.unet import unet_base

IMAGENET_TRAIN_SIZE = 1_281_167
_VAE = None


def init_distributed() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, local_rank, world_size
    return 0, 0, 1


def is_main(rank: int) -> bool:
    return rank == 0


def log(msg: str, rank: int = 0) -> None:
    if is_main(rank):
        print(msg, flush=True)


def build_generator(backbone: str, num_classes: int, hidden_dim: int, noise_dim: int,
                    noise_classes: int = 64, noise_coords: int = 32) -> torch.nn.Module:
    if backbone == "conv":
        return LatentConvGenerator(num_classes=num_classes, noise_dim=noise_dim, hidden_dim=hidden_dim)
    if backbone == "dit_s_2":
        return dit_s_2(num_classes=num_classes, noise_classes=noise_classes, noise_coords=noise_coords)
    if backbone == "dit_b_2":
        return dit_b_2(num_classes=num_classes, noise_classes=noise_classes, noise_coords=noise_coords)
    if backbone == "dit_b_4":
        return dit_b_4(num_classes=num_classes, noise_classes=noise_classes, noise_coords=noise_coords)
    if backbone == "unet_base":
        return unet_base(num_classes=num_classes, noise_classes=noise_classes, noise_coords=noise_coords)
    raise ValueError(f"unknown backbone: {backbone}")


def generate_latents(
    gen: torch.nn.Module,
    n: int,
    labels: torch.Tensor,
    *,
    backbone: str,
    noise_dim: int,
    device: torch.device,
    gen_mode: str = "direct",
    flow_steps: int = 1,
    flow_time_scale: float = 999.0,
    gen_timestep: int = 999,
    train: bool = False,
    cfg_scale: Optional[torch.Tensor] = None,
    noise_classes: int = 0,
    noise_coords: int = 0,
    noise_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """G(ε, y) → x_fake in one forward (default). Optional flow mode for ablation only."""
    if backbone == "conv":
        noise = torch.randn(n, noise_dim, device=device)
        omega = torch.full((n,), 2.0, device=device)
        return gen(noise, labels, omega)
    if gen_mode == "flow":
        from imagenet.flow_gen import integrate_flow_dit

        return integrate_flow_dit(
            gen, n, labels,
            num_steps=flow_steps,
            time_scale=flow_time_scale,
            train=train,
        )
    noise = torch.randn(n, 4, 32, 32, device=device)

    # Drift-mode: use cfg_scale and noise_labels for conditioning (drifting-aligned)
    # Fallback to original t-slot convention for backward compat
    if cfg_scale is not None and noise_classes > 0:
        return gen(
            noise,
            t=torch.full((n,), 999, device=device, dtype=torch.float32),  # unused in drift mode
            y=labels,
            cfg_scale=cfg_scale,
            noise_labels=noise_labels,
            train=train,
        )
    # Original direct mode (backward compat)
    t = torch.full((n,), gen_timestep, device=device, dtype=torch.float32)
    return gen(noise, t, labels, h=torch.zeros_like(t), train=train)


def generation_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "gen_mode": args.gen_mode,
        "flow_steps": args.flow_steps,
        "flow_time_scale": args.flow_time_scale,
        "gen_timestep": args.gen_timestep,
    }


class LatentDataset:
    def __init__(
        self,
        latents: np.ndarray,
        labels: np.ndarray,
        device: torch.device,
        rank: int = 0,
        *,
        resample_latents: bool = True,
    ):
        self.device = device
        self.size = len(latents)
        self.resample_latents = resample_latents
        if isinstance(latents, np.memmap):
            self.latents = latents
        else:
            self.latents = np.asarray(latents, dtype=np.float32)
        self.latent_channels = int(self.latents.shape[1])
        self.has_std = self.latent_channels >= 8
        if resample_latents and not self.has_std:
            log("dataset: 4ch mean-only cache; disable latent resampling", rank)
            self.resample_latents = False
        self.labels = labels.astype(np.int64)
        self.num_classes = int(labels.max()) + 1
        buckets: list[list[int]] = [[] for _ in range(self.num_classes)]
        for i, c in enumerate(self.labels):
            buckets[int(c)].append(i)
        self.class_indices = [np.asarray(b, dtype=np.int64) for b in buckets]
        self.valid_classes = [c for c in range(self.num_classes) if len(self.class_indices[c]) > 0]
        fmt = "mean+std resample" if self.resample_latents else ("8ch mean+std" if self.has_std else "mean only")
        log(f"dataset: {len(latents)} samples, {len(self.valid_classes)} classes, {fmt} (CPU/mmap)", rank)

    def _to_tensor(self, batch: np.ndarray) -> torch.Tensor:
        if batch.ndim == 3:
            batch = batch[:, None]
        if self.resample_latents and batch.shape[1] >= 8:
            mean = batch[:, :4]
            std = batch[:, 4:8]
            eps = np.random.randn(*mean.shape).astype(np.float32)
            batch = mean + std * eps
        elif batch.shape[1] > 4:
            batch = batch[:, :4]
        return torch.from_numpy(np.ascontiguousarray(batch)).float().to(self.device)

    def sample_class(self, c: int, n: int) -> torch.Tensor:
        idx = self.class_indices[c]
        if len(idx) == 0:
            raise ValueError(f"class {c} has no samples")
        sel = np.random.choice(idx, size=n, replace=True)
        return self._to_tensor(self.latents[sel].copy())

    def sample_random(self, n: int) -> torch.Tensor:
        sel = np.random.randint(0, len(self.latents), size=n)
        return self._to_tensor(self.latents[sel].copy())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="/home/liushuai/data1/liujiawei/imagenet256/cache_full")
    p.add_argument(
        "--backbone",
        default="dit_b_2",
        choices=["conv", "dit_s_2", "dit_b_2", "dit_b_4", "unet_base"],
    )
    p.add_argument("--coupling", default="sinkhorn", choices=["row", "partial_two_sided", "sinkhorn"])
    p.add_argument("--drift-form", default="split", choices=["alg2_joint", "split"])
    p.add_argument("--sinkhorn-iters", type=int, default=20)
    p.add_argument("--sinkhorn-marginal", default="weighted_cols", choices=["none", "weighted_cols", "post_guidance"])
    p.add_argument("--temps", type=float, nargs="+", default=[0.02, 0.05, 0.2])
    p.add_argument("--dist-metric", default="l2_sq", choices=["l2", "l2_sq"])
    p.add_argument("--nneg", type=int, default=64)
    p.add_argument("--npos", type=int, default=64)
    p.add_argument("--nuncond", type=int, default=16)
    p.add_argument(
        "--classes-per-step",
        type=int,
        default=64,
        help="Global number of classes per optimizer step (split across GPUs under DDP)",
    )
    p.add_argument("--epochs", type=float, default=96.0, help="Train for this many ImageNet-scale epochs")
    p.add_argument("--steps", type=int, default=None, help="Override: fixed step count instead of --epochs")
    p.add_argument(
        "--iters-per-epoch",
        type=int,
        default=None,
        help="Drift steps counted as one epoch (default: ceil(train_size/(classes_per_step*npos)), ~501)",
    )
    p.add_argument(
        "--imagenet-train-size",
        type=int,
        default=IMAGENET_TRAIN_SIZE,
        help="Dataset size used to convert epochs -> iterations (paper: 1,281,167)",
    )
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=5000)
    p.add_argument("--grad-clip", type=float, default=2.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--noise-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--sample-every-epochs", type=float, default=1.0)
    p.add_argument("--decode-rgb-every-epochs", type=float, default=1.0)
    p.add_argument("--vae-id", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--sample-classes", type=int, default=8)
    p.add_argument(
        "--feature-mode",
        default="encoder",
        choices=["encoder", "vanilla"],
        help="encoder=multi-scale conv features (Appendix A.5); vanilla=flatten latent",
    )
    p.add_argument("--encoder-every-n-blocks", type=int, default=2)
    p.add_argument("--noise-classes", type=int, default=64, help="Noise embedding dict size (drifting-aligned)")
    p.add_argument("--noise-coords", type=int, default=32, help="Number of noise embedding coords (drifting-aligned)")
    p.add_argument(
        "--gen-mode",
        default="direct",
        choices=["direct", "flow"],
        help="direct: G(ε,y)→latent one forward (default); flow: ablation only",
    )
    p.add_argument(
        "--flow-steps",
        type=int,
        default=1,
        help="Number of flow integration steps (1 = single step t:1->0, like MeanFlow default)",
    )
    p.add_argument(
        "--flow-time-scale",
        type=float,
        default=999.0,
        help="Scale [0,1] flow times to DiT timestep embedding (999 ≈ noise end)",
    )
    p.add_argument(
        "--gen-timestep",
        type=int,
        default=999,
        help="Only for --gen-mode direct: fixed t embedding for one-shot G(noise)",
    )
    p.add_argument(
        "--resample-latents",
        action="store_true",
        default=True,
        help="Resample z = mean + std*eps when cache has 8 channels",
    )
    p.add_argument("--no-resample-latents", action="store_false", dest="resample_latents")
    p.add_argument("--device", default="cuda:0", help="Used only for single-GPU mode")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fid-every-epochs", type=float, default=0.0, help="Periodic FID every N epochs (0=off; use --fid-at-end for final eval)")
    p.add_argument("--fid-at-end", action="store_true", default=True, help="Run FID at the last training step")
    p.add_argument("--no-fid-at-end", action="store_false", dest="fid_at_end")
    p.add_argument("--fid-num-samples", type=int, default=10000, help="Samples for periodic inline FID")
    p.add_argument("--fid-end-num-samples", type=int, default=50000, help="Samples for final FID at training end")
    p.add_argument(
        "--fid-ref-stats",
        default=os.environ.get(
            "FID_REF",
            "/home/liushuai/data1/liujiawei/imagenet256/fid_ref/imagenet_256_fid_stats.npz",
        ),
        help="npz with ref mu/sigma (ref_mu/ref_sigma or mu/sigma); skip FID if missing",
    )
    p.add_argument("--run-root", default="runs/imagenet256_drift")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--tensorboard", action="store_true", default=True)
    p.add_argument("--no-tensorboard", action="store_false", dest="tensorboard")
    return p.parse_args()


def iters_per_epoch(imagenet_train_size: int, classes_per_step: int, npos: int) -> int:
    global_pos = classes_per_step * npos
    return max(1, math.ceil(imagenet_train_size / global_pos))


def append_fid_result(run_dir: str, record: dict) -> None:
    path = os.path.join(run_dir, "fid_results.json")
    records: list = []
    if os.path.isfile(path):
        with open(path) as f:
            records = json.load(f)
    records.append(record)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def resolve_total_steps(args: argparse.Namespace) -> tuple[int, int]:
    if args.iters_per_epoch is not None:
        ipe = max(1, int(args.iters_per_epoch))
    else:
        ipe = iters_per_epoch(args.imagenet_train_size, args.classes_per_step, args.npos)
    if args.steps is not None:
        return args.steps, ipe
    return int(math.ceil(args.epochs * ipe)), ipe


def partition_step_classes(step_classes: np.ndarray, rank: int, world_size: int) -> np.ndarray:
    if world_size == 1:
        return step_classes
    splits = np.array_split(step_classes, world_size)
    return splits[rank]


def set_lr(optimizer, step: int, warmup: int, base_lr: float) -> float:
    if warmup > 0 and step < warmup:
        lr = base_lr * (step + 1) / warmup
    else:
        lr = base_lr
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


@torch.no_grad()
def make_latent_grid_tensor(
    gen: torch.nn.Module,
    dataset: LatentDataset,
    classes: list[int],
    nrow: int,
    device: torch.device,
    *,
    backbone: str,
    noise_dim: int,
    gen_mode: str,
    flow_steps: int,
    flow_time_scale: float,
    gen_timestep: int,
    noise_classes: int = 0,
    noise_coords: int = 0,
) -> torch.Tensor:
    tiles = []
    has_noise = noise_classes > 0
    for c in classes:
        labels = torch.full((nrow,), c, dtype=torch.long, device=device)
        noise_lbl = torch.randint(0, noise_classes, (nrow, noise_coords), device=device) if has_noise else None
        fake = generate_latents(
            gen, nrow, labels, backbone=backbone, noise_dim=noise_dim,
            device=device, gen_mode=gen_mode, flow_steps=flow_steps,
            flow_time_scale=flow_time_scale, gen_timestep=gen_timestep,
            cfg_scale=torch.ones(nrow, device=device) if has_noise else None,
            noise_labels=noise_lbl,
            noise_classes=noise_classes, noise_coords=noise_coords,
        )
        real = dataset.sample_class(c, nrow)
        for z in (real, fake):
            vis = z.mean(dim=1, keepdim=True).clamp(-3, 3)
            vis = (vis + 3) / 6
            tiles.append(vis)
    return make_grid(torch.cat(tiles, dim=0), nrow=nrow, padding=1)


@torch.no_grad()
def save_latent_grid(
    gen: torch.nn.Module,
    dataset: LatentDataset,
    classes: list[int],
    nrow: int,
    device: torch.device,
    out_path: str,
    *,
    backbone: str,
    noise_dim: int,
    gen_mode: str,
    flow_steps: int,
    flow_time_scale: float,
    gen_timestep: int,
    noise_classes: int = 0,
    noise_coords: int = 0,
) -> torch.Tensor:
    grid = make_latent_grid_tensor(
        gen, dataset, classes, nrow, device, backbone=backbone, noise_dim=noise_dim,
        gen_mode=gen_mode, flow_steps=flow_steps, flow_time_scale=flow_time_scale,
        gen_timestep=gen_timestep,
        noise_classes=noise_classes, noise_coords=noise_coords,
    )
    save_image(grid, out_path)
    return grid


@torch.no_grad()
def make_rgb_grid_tensor(
    gen: torch.nn.Module,
    dataset: LatentDataset,
    classes: list[int],
    nrow: int,
    device: torch.device,
    *,
    backbone: str,
    noise_dim: int,
    vae_id: str,
    gen_mode: str,
    flow_steps: int,
    flow_time_scale: float,
    gen_timestep: int,
    noise_classes: int = 0,
    noise_coords: int = 0,
) -> torch.Tensor:
    vae = get_vae(vae_id, device)
    rows = []
    has_noise = noise_classes > 0
    for c in classes:
        labels = torch.full((nrow,), c, dtype=torch.long, device=device)
        noise_lbl = torch.randint(0, noise_classes, (nrow, noise_coords), device=device) if has_noise else None
        fake = generate_latents(
            gen, nrow, labels, backbone=backbone, noise_dim=noise_dim,
            device=device, gen_mode=gen_mode, flow_steps=flow_steps,
            flow_time_scale=flow_time_scale, gen_timestep=gen_timestep,
            cfg_scale=torch.ones(nrow, device=device) if has_noise else None,
            noise_labels=noise_lbl,
            noise_classes=noise_classes, noise_coords=noise_coords,
        )
        real = dataset.sample_class(c, nrow)
        rows.append(decode_latents(vae, real))
        rows.append(decode_latents(vae, fake))
    return make_grid(torch.cat(rows, dim=0), nrow=nrow, padding=2)


def get_vae(vae_id: str, device: torch.device):
    global _VAE
    if _VAE is None:
        from diffusers import AutoencoderKL

        log(f"loading VAE {vae_id} ...")
        try:
            _VAE = AutoencoderKL.from_pretrained(vae_id, local_files_only=True).to(device)
        except OSError:
            log("VAE not in local cache, trying online download (HF_ENDPOINT / mirror)...")
            _VAE = AutoencoderKL.from_pretrained(vae_id).to(device)
        _VAE.eval()
    return _VAE


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    z = latents / 0.18215
    imgs = vae.decode(z).sample
    return (imgs.clamp(-1, 1) + 1) * 0.5


@torch.no_grad()
def save_rgb_grid(
    gen: torch.nn.Module,
    dataset: LatentDataset,
    classes: list[int],
    nrow: int,
    device: torch.device,
    out_path: str,
    *,
    backbone: str,
    noise_dim: int,
    vae_id: str,
    gen_mode: str,
    flow_steps: int,
    flow_time_scale: float,
    gen_timestep: int,
    noise_classes: int = 0,
    noise_coords: int = 0,
) -> torch.Tensor:
    grid = make_rgb_grid_tensor(
        gen, dataset, classes, nrow, device,
        backbone=backbone, noise_dim=noise_dim, vae_id=vae_id,
        gen_mode=gen_mode, flow_steps=flow_steps, flow_time_scale=flow_time_scale,
        gen_timestep=gen_timestep,
        noise_classes=noise_classes, noise_coords=noise_coords,
    )
    save_image(grid, out_path)
    return grid


def sample_step_classes(valid_classes: list[int], n: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.choice(valid_classes, size=min(n, len(valid_classes)), replace=False)


def reduce_mean(value: float, device: torch.device, world_size: int) -> float:
    if world_size == 1:
        return value
    t = torch.tensor(value, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / world_size).item()


def epoch_at_step(step: int, iters_per_ep: int) -> float:
    return step / iters_per_ep


def load_ema_weights(model: torch.nn.Module, ckpt: dict) -> None:
    state = ckpt.get("ema_state") or ckpt.get("model")
    if state is None:
        raise KeyError("checkpoint missing model/ema_state")
    model.load_state_dict(state, strict=False)


def compute_class_drift_loss(
    x_lat: torch.Tensor,
    pos_lat: torch.Tensor,
    unc_lat: torch.Tensor,
    *,
    feature_mode: str,
    encoder: torch.nn.Module | None,
    encoder_every_n_blocks: int,
    omega: torch.Tensor,
    args: argparse.Namespace,
    class_stats: dict[str, float],
) -> torch.Tensor:
    drift_kwargs = dict(
        omega=omega,
        temps=args.temps,
        vanilla=False,
        drift_form=args.drift_form,
        coupling=args.coupling,
        sinkhorn_iters=args.sinkhorn_iters,
        sinkhorn_marginal=args.sinkhorn_marginal,
        mask_self_neg=(args.coupling != "sinkhorn"),
        dist_metric=args.dist_metric,
        normalize_drift_theta=True,
    )
    if feature_mode == "vanilla":
        x_feat = x_lat.view(x_lat.shape[0], 1, -1)
        y_pos_feat = pos_lat.view(pos_lat.shape[0], 1, -1)
        y_unc_feat = unc_lat.view(unc_lat.shape[0], 1, -1) if unc_lat.numel() else unc_lat
        return drifting_loss_for_feature_set(
            x_feat, y_pos_feat, y_unc_feat, stats=class_stats, **drift_kwargs,
        )

    assert encoder is not None
    x_sets = feature_sets_from_latents(
        encoder, x_lat, every_n_blocks=encoder_every_n_blocks,
    )
    with torch.no_grad():
        pos_sets = feature_sets_from_latents(
            encoder, pos_lat, every_n_blocks=encoder_every_n_blocks,
        )
        if unc_lat.numel():
            unc_sets = feature_sets_from_latents(
                encoder, unc_lat, every_n_blocks=encoder_every_n_blocks,
            )
        else:
            unc_sets = []
    return drifting_loss_over_feature_sets(
        x_sets, pos_sets, unc_sets, stats=class_stats, **drift_kwargs,
    )


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = init_distributed()
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)

    if args.classes_per_step < world_size:
        raise ValueError(f"--classes-per-step ({args.classes_per_step}) must be >= world_size ({world_size})")

    total_steps, ipe = resolve_total_steps(args)
    sample_every = max(1, int(round(args.sample_every_epochs * ipe)))
    decode_rgb_every = max(1, int(round(args.decode_rgb_every_epochs * ipe)))
    fid_every = max(1, int(round(args.fid_every_epochs * ipe))) if args.fid_every_epochs > 0 else 0
    has_fid_ref = bool(args.fid_ref_stats) and os.path.isfile(args.fid_ref_stats)
    fid_periodic = fid_every > 0 and has_fid_ref
    fid_at_end = args.fid_at_end and has_fid_ref

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    latents = np.load(os.path.join(args.cache_dir, "train_latents.npy"), mmap_mode="r")
    labels = np.load(os.path.join(args.cache_dir, "train_labels.npy"), mmap_mode="r")
    dataset = LatentDataset(
        latents, labels, device, rank=rank, resample_latents=args.resample_latents,
    )

    feature_encoder = None
    if args.feature_mode == "encoder":
        feature_encoder = build_latent_encoder(device=device, freeze=True)
        log(f"feature encoder: LatentFeatureEncoder (every_n_blocks={args.encoder_every_n_blocks})", rank)

    raw_gen = build_generator(
        args.backbone, dataset.num_classes, args.hidden_dim, args.noise_dim,
        noise_classes=args.noise_classes, noise_coords=args.noise_coords,
    ).to(device)
    ema = EMA(raw_gen, decay=args.ema_decay)
    opt = torch.optim.AdamW(raw_gen.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        raw_gen.load_state_dict(ckpt["model"], strict=False)
        if "ema_state" in ckpt:
            for k, v in ckpt["ema_state"].items():
                ema.shadow[k] = v.clone().to(device)
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        log(f"resumed from {args.resume} at step {start_step} (epoch {epoch_at_step(start_step, ipe):.3f})", rank)

    gen = DDP(raw_gen, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) if world_size > 1 else raw_gen

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.run_name or f"{ts}_sinkhorn_imagenet256"
    run_dir = os.path.join(args.run_root, name)
    if args.resume and args.run_name:
        os.makedirs(run_dir, exist_ok=True)
    elif not args.resume:
        os.makedirs(run_dir, exist_ok=True)
    elif args.resume:
        run_dir = os.path.dirname(os.path.abspath(args.resume))

    per_rank_classes = max(1, args.classes_per_step // world_size)
    global_pos = args.classes_per_step * args.npos
    if is_main(rank):
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(
                {
                    **vars(args),
                    "world_size": world_size,
                    "total_steps": total_steps,
                    "iters_per_epoch": ipe,
                    "global_pos_per_step": global_pos,
                    "classes_per_rank": per_rank_classes,
                    "sample_every_steps": sample_every,
                    "decode_rgb_every_steps": decode_rgb_every,
                },
                f,
                indent=2,
            )
    if world_size > 1:
        dist.barrier()

    log(f"run_dir: {run_dir}", rank)
    log(f"world_size={world_size}, device={device}", rank)
    log(
        f"schedule: {args.epochs} epochs = {total_steps} steps "
        f"({ipe} iters/epoch, global {args.classes_per_step} classes x {args.npos} pos = {global_pos} samples/step)",
        rank,
    )
    if world_size > 1:
        log(f"DDP split: {per_rank_classes} classes/GPU (wall-clock ~1/{world_size} of single-GPU)", rank)
    if fid_periodic:
        log(f"FID eval every {args.fid_every_epochs} epochs ({fid_every} steps), n={args.fid_num_samples}", rank)
    if fid_at_end:
        log(f"FID at end: n={args.fid_end_num_samples}, ref={args.fid_ref_stats}", rank)
    if (args.fid_every_epochs > 0 or args.fid_at_end) and not has_fid_ref:
        log(
            f"FID disabled: ref stats not found at {args.fid_ref_stats!r} "
            f"(run scripts/download_imagenet256_fid_ref.sh)",
            rank,
        )
    log(f"training step {start_step + 1} .. {total_steps}", rank)
    gen_desc = (
        f"G(ε,y) one-shot, t={args.gen_timestep}"
        if args.gen_mode == "direct"
        else f"flow steps={args.flow_steps}, scale={args.flow_time_scale}"
    )
    log(
        f"features: {args.feature_mode}, {gen_desc}, resample_latents={dataset.resample_latents}",
        rank,
    )

    tb_writer: SummaryWriter | None = None
    if args.tensorboard and is_main(rank):
        tb_dir = os.path.join(run_dir, "tensorboard")
        tb_writer = SummaryWriter(log_dir=tb_dir)
        log(f"tensorboard: {tb_dir}", rank)
        tb_writer.add_text("config", json.dumps(vars(args), indent=2), 0)

    sample_classes = dataset.valid_classes[: args.sample_classes]
    gen.train()

    for step in range(start_step + 1, total_steps + 1):
        lr = set_lr(opt, step - 1, args.warmup_steps, args.lr)
        opt.zero_grad()

        # Drifting-aligned CFG sampling: uniform [1.0, 4.0], 50% dropout to 1.0
        _raw = 1.0 + torch.rand(args.classes_per_step, device=device) * 3.0
        _drop = torch.rand(args.classes_per_step, device=device) < 0.5
        omegas = torch.where(_drop, torch.ones_like(_raw), _raw)

        step_classes = sample_step_classes(
            dataset.valid_classes,
            args.classes_per_step,
            seed=args.seed + step * 1009,
        )
        my_classes = partition_step_classes(step_classes, rank, world_size)
        n_global = len(step_classes)

        total_loss = 0.0
        step_stats: dict[str, float] = {}
        for ci, c in enumerate(my_classes):
            c = int(c)
            c_labels = torch.full((args.nneg,), c, dtype=torch.long, device=device)

            # Drifting-aligned: noise labels for diversity
            noise_labels = None
            if args.noise_classes > 0:
                noise_labels = torch.randint(0, args.noise_classes, (args.nneg, args.noise_coords), device=device)

            # Drifting-aligned: cfg_scale per class
            cfg_scale_val = float(omegas[min(ci, len(omegas) - 1)].item())
            cfg_scales = torch.full((args.nneg,), cfg_scale_val, dtype=torch.float32, device=device)

            x_lat = generate_latents(
                gen, args.nneg, c_labels, backbone=args.backbone, noise_dim=args.noise_dim,
                device=device, train=True, cfg_scale=cfg_scales,
                noise_labels=noise_labels, **generation_kwargs(args),
            )
            pos_lat = dataset.sample_class(c, args.npos)
            unc_lat = dataset.sample_random(args.nuncond) if args.nuncond > 0 else torch.empty(0, 4, 32, 32, device=device)

            class_stats: dict[str, float] = {}
            loss = compute_class_drift_loss(
                x_lat, pos_lat, unc_lat,
                feature_mode=args.feature_mode,
                encoder=feature_encoder,
                encoder_every_n_blocks=args.encoder_every_n_blocks,
                omega=torch.tensor(cfg_scale_val, device=device),
                args=args,
                class_stats=class_stats,
            )
            (loss / n_global).backward()
            total_loss += loss.item()
            for k, v in class_stats.items():
                step_stats[k] = step_stats.get(k, 0.0) + v / max(len(my_classes), 1)

        grad_norm = 0.0
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_gen.parameters(), args.grad_clip).item()
        opt.step()
        ema.update(raw_gen)

        local_denom = max(len(my_classes), 1)
        mean_loss = reduce_mean(total_loss / local_denom, device, world_size)
        ep = epoch_at_step(step, ipe)
        if step % args.log_every == 0 or step == start_step + 1:
            log(
                f"[epoch {ep:.3f}/{args.epochs}] step {step}/{total_steps} "
                f"loss={mean_loss:.6f} ({len(my_classes)} cls/rank)",
                rank,
            )
            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", mean_loss, step)
                tb_writer.add_scalar("train/lr", lr, step)
                tb_writer.add_scalar("train/epoch", ep, step)
                tb_writer.add_scalar("train/omega", omegas[:min(1, len(omegas))].mean().item(), step)
                if grad_norm > 0:
                    tb_writer.add_scalar("train/grad_norm", grad_norm, step)
                for k, v in step_stats.items():
                    if k.startswith("drift_"):
                        tb_writer.add_scalar(f"drift/{k}", v, step)

        need_fid_periodic = fid_periodic and step % fid_every == 0
        need_fid_end = fid_at_end and step == total_steps
        need_fid = need_fid_periodic or need_fid_end
        need_sample = step % sample_every == 0 or step == total_steps
        need_rgb = decode_rgb_every > 0 and (step % decode_rgb_every == 0 or step == total_steps)
        if need_sample or need_rgb or need_fid:
            if world_size > 1:
                dist.barrier()
            if is_main(rank):
                raw_gen.eval()
                orig = {n: p.clone() for n, p in raw_gen.named_parameters() if p.requires_grad}
                ema.copy_to(raw_gen)
                ep_tag = f"{ep:.2f}".replace(".", "p")
                if need_sample:
                    out = os.path.join(run_dir, f"latent_grid_epoch{ep_tag}_step{step}.png")
                    latent_grid = save_latent_grid(
                    raw_gen, dataset, sample_classes, nrow=4, device=device, out_path=out,
                    backbone=args.backbone, noise_dim=args.noise_dim,
                    noise_classes=args.noise_classes, noise_coords=args.noise_coords,
                    **generation_kwargs(args),
                    )
                    if tb_writer is not None:
                        tb_writer.add_image("samples/latent_grid", latent_grid.cpu(), step)
                if need_rgb:
                    try:
                        rgb_out = os.path.join(run_dir, f"samples_rgb_epoch{ep_tag}_step{step}.png")
                        rgb_grid = save_rgb_grid(
                            raw_gen, dataset, sample_classes, nrow=4, device=device, out_path=rgb_out,
                            backbone=args.backbone, noise_dim=args.noise_dim, vae_id=args.vae_id,
                            noise_classes=args.noise_classes, noise_coords=args.noise_coords,
                            **generation_kwargs(args),
                        )
                        log(f"  -> {rgb_out}", rank)
                        if tb_writer is not None:
                            tb_writer.add_image("samples/rgb", rgb_grid.cpu(), step)
                    except Exception as exc:
                        log(f"  RGB decode skipped (training continues): {exc}", rank)
                if need_fid and has_fid_ref:
                    try:
                        from imagenet.eval_fid import evaluate_fid

                        n_fid = args.fid_end_num_samples if need_fid_end else args.fid_num_samples
                        fid_score = evaluate_fid(
                            raw_gen,
                            num_samples=n_fid,
                            num_classes=dataset.num_classes,
                            backbone=args.backbone,
                            noise_dim=args.noise_dim,
                            device=device,
                            vae_id=args.vae_id,
                            ref_stats_path=args.fid_ref_stats,
                            batch_size=32,
                            noise_classes=args.noise_classes,
                            noise_coords=args.noise_coords,
                            **generation_kwargs(args),
                        )
                        tag = "end" if need_fid_end else "periodic"
                        log(f"  FID@{n_fid} ({tag}) = {fid_score:.4f}", rank)
                        if tb_writer is not None:
                            tb_writer.add_scalar("eval/fid", fid_score, step)
                        append_fid_result(
                            run_dir,
                            {
                                "method": "sinkhorn_drift",
                                "step": step,
                                "epoch": ep,
                                "num_samples": n_fid,
                                "fid": fid_score,
                                "tag": tag,
                                "ref_stats": args.fid_ref_stats,
                            },
                        )
                    except Exception as exc:
                        log(f"  FID eval failed: {exc}", rank)
                for n, p in raw_gen.named_parameters():
                    if n in orig:
                        p.data.copy_(orig[n])
                raw_gen.train()
            if world_size > 1:
                dist.barrier()

        save_ckpt = step % ipe == 0 or step == total_steps
        if save_ckpt:
            if world_size > 1:
                dist.barrier()
            if is_main(rank):
                ckpt = {
                    "step": step,
                    "epoch": ep,
                    "model": raw_gen.state_dict(),
                    "ema_state": {n: ema.shadow[n].clone() for n in ema.shadow},
                    "optimizer": opt.state_dict(),
                    "gen_config": {
                        "backbone": args.backbone,
                        "num_classes": dataset.num_classes,
                        "noise_dim": args.noise_dim,
                        "hidden_dim": args.hidden_dim,
                        **generation_kwargs(args),
                        "feature_mode": args.feature_mode,
                    },
                    "args": {**vars(args), "world_size": world_size, "total_steps": total_steps, "iters_per_epoch": ipe},
                    "sample_classes": sample_classes,
                }
                if step == total_steps:
                    tag = "ckpt_final.pt"
                else:
                    ep_int = int(round(ep))
                    tag = f"ckpt_epoch{ep_int}.pt"
                path = os.path.join(run_dir, tag)
                torch.save(ckpt, path)
                log(f"saved {path} (epoch {ep:.3f})", rank)
            if world_size > 1:
                dist.barrier()

    log("done", rank)
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
