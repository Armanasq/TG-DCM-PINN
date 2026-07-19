"""Train TG-DCM with structured logging and checkpointing."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_optimizer
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from tsfast.models.scaling import ScaledModel, StandardScaler
from tsfast.quaternions import QuaternionAugmentation, augmentation_groups
from tsfast.training import ignore_nan
from tsfast.training.learners import RNNLearner
from tsfast.training.schedulers import sched_flat_cos
from tsfast.training.transforms import bias, noise_grouped

from tg_dcm.data import get_dls, get_files
from tg_dcm.losses import (
    RiannV2Loss,
    TargetDCM3Loss,
    finite_target_quat,
    inclination_error_rad,
    masked_geodesic_mean_deg,
    masked_pitch_rmse_deg,
    masked_roll_rmse_deg,
    masked_tilt_mean_deg,
    masked_tilt_rmse_deg,
    qnorm,
    roll_pitch_deg,
)


def prefer_venv_nvidia_libs() -> None:
    if os.environ.get("RIANN_VENV_NVIDIA_LIBS") == "1":
        return
    site_packages = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    nvidia_root = site_packages / "nvidia"
    lib_dirs = [
        nvidia_root / name / "lib"
        for name in (
            "cuda_runtime",
            "cublas",
            "cudnn",
            "cuda_nvrtc",
            "curand",
            "cusolver",
            "cusparse",
            "cufft",
            "nccl",
            "nvtx",
        )
    ]
    existing = [str(path) for path in lib_dirs if path.exists()]
    if not existing:
        return
    current = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    if current[: len(existing)] == existing:
        os.environ["RIANN_VENV_NVIDIA_LIBS"] = "1"
        return
    os.environ["RIANN_VENV_NVIDIA_LIBS"] = "1"
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        existing + [part for part in current if part not in existing]
    )
    os.execv(sys.executable, [sys.executable, *sys.argv])


if __name__ == "__main__":
    prefer_venv_nvidia_libs()


DLS_MEAN = np.array([-0.6845, -0.4654, 0.2514, -0.0715, 0.0343, 0.1019, 0.0035])
DLS_STD = np.array([7.1055, 7.0015, 7.1748, 2.0147, 1.9856, 2.0225, 1.8847e-03])
PBAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, ETA {remaining}, {rate_fmt}{postfix}]"


@dataclass
class TrainConfig:
    epochs: int = 512
    lr: float = 3e-3
    pct_start: float = 0.15
    weight_decay: float = 0.01
    batch_size: int = 64
    win_sz: int = 9000
    stp_sz: int = 60
    n_batches_train: int | None = 300
    targ_fs_count: int = 427
    sub_seq_len: int = 1800
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = False
    seed: int = 42
    out_dir: Path = Path("runs")
    run_name: str | None = "tg_dcm"
    data_dir: Path | None = None
    resume: Path | None = None
    resume_optimizer: bool = False
    no_resume_scheduler: bool = False
    device: str = "auto"
    cuda_graph: bool = True
    use_cudnn: bool = True
    stop_on_nonfinite: bool = True
    save_every: int = 1
    metric_every: int = 10
    metric_samples: int = 8
    compile_aux_losses: bool = True
    smoke: bool = False


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train TG-DCM.")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--out-dir", type=Path, default=TrainConfig.out_dir)
    parser.add_argument("--run-name", type=str, default=TrainConfig.run_name)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=TrainConfig.device)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument(
        "--no-cudnn", action="store_false", dest="use_cudnn", default=TrainConfig.use_cudnn
    )
    parser.add_argument("--metric-every", type=int, default=TrainConfig.metric_every)
    parser.add_argument("--metric-samples", type=int, default=TrainConfig.metric_samples)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        out_dir=args.out_dir,
        run_name=args.run_name,
        data_dir=args.data_dir,
        resume=args.resume,
        resume_optimizer=args.resume_optimizer,
        device=args.device,
        cuda_graph=not args.no_cuda_graph,
        use_cudnn=args.use_cudnn,
        metric_every=max(1, args.metric_every),
        metric_samples=max(1, args.metric_samples),
        smoke=args.smoke,
    )
    if cfg.smoke:
        cfg.epochs = 1
        cfg.batch_size = min(cfg.batch_size, 2)
        cfg.win_sz = min(cfg.win_sz, 128)
        cfg.stp_sz = min(cfg.stp_sz, 64)
        cfg.n_batches_train = 1
        cfg.targ_fs_count = 1
        cfg.sub_seq_len = 0
        cfg.cuda_graph = False
    return cfg


def setup_run_dir(cfg: TrainConfig) -> Path:
    run_name = cfg.run_name or datetime.now().strftime("tg_dcm_%Y%m%d_%H%M%S")
    run_dir = cfg.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("tg_dcm.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(run_dir / "train.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=jsonable)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_backend(cfg: TrainConfig, logger: logging.Logger) -> None:
    if not cfg.use_cudnn:
        torch.backends.cudnn.enabled = False
        logger.info(
            "cuDNN disabled; using native PyTorch RNN kernels for stable RIANN GRU training."
        )
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        # TF32 for cuDNN ops (GRU gates); separate from matmul precision
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        logger.info("TF32 matmul + cuDNN TF32 + cuDNN benchmark autotuning enabled.")


def cuda_training_is_usable(cfg: TrainConfig | None = None) -> tuple[bool, str | None]:
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is false"
    try:
        seq_len = min(max((cfg.sub_seq_len if cfg else 128) or 128, 16), 256)
        batch_size = min(max((cfg.batch_size if cfg else 2), 2), 4)
        model = torch.nn.GRU(input_size=7, hidden_size=200, num_layers=2, batch_first=True).cuda()
        head = torch.nn.Linear(200, 4).cuda()
        x = torch.randn(batch_size, seq_len, 7, device="cuda")
        y, _ = model(x)
        head(y).square().mean().backward()
        torch.cuda.synchronize()
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def resolve_device(cfg: TrainConfig, logger: logging.Logger) -> torch.device:
    if cfg.device == "auto":
        ok, reason = cuda_training_is_usable(cfg)
        if ok:
            return torch.device("cuda")
        logger.warning("CUDA training preflight failed; using CPU for this run. Reason: %s", reason)
        return torch.device("cpu")
    if cfg.device == "cuda":
        ok, reason = cuda_training_is_usable(cfg)
        if not ok:
            raise RuntimeError(
                f"--device cuda requested, but CUDA training preflight failed: {reason}"
            )
    return torch.device(cfg.device)


def summarize_data(cfg: TrainConfig) -> dict[str, Any]:
    splits = get_files(cfg.data_dir)
    return {
        "train_files": len(splits["train"]),
        "valid_files": len(splits["valid"]),
        "test_files": len(splits["test"]),
        "train_examples": [p.name for p in splits["train"][:5]],
        "valid_examples": [p.name for p in splits["valid"][:5]],
    }


def build_dls(cfg: TrainConfig, win_sz: int | None = None):
    return get_dls(
        data_dir=cfg.data_dir,
        win_sz=win_sz if win_sz is not None else cfg.win_sz,
        stp_sz=cfg.stp_sz,
        bs=cfg.batch_size,
        n_batches_train=cfg.n_batches_train,
        targ_fs_count=cfg.targ_fs_count,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
        seed=cfg.seed,
    )


def build_learner(cfg: TrainConfig, device: torch.device):
    dls = build_dls(cfg)
    augmentations = [
        QuaternionAugmentation(augmentation_groups([3, 3])),
        bias(std=[0, 0, 0, 0.006556, 0.006461, 0.006581, 0], p=0.1067),
        noise_grouped(std_std=[0.6410, 0.00712, 0.0], std_idx=[0, 0, 0, 1, 1, 1, 2], p=0.6123),
    ]
    loss_func = RiannV2Loss(inc_weight=1.0, norm_weight=0.01)
    aux_losses = [
        TargetDCM3Loss(
            dyn_weight=0.10,
            gravity_weight=0.05,
            target_gate_deg=2.0,
            dyn_deadzone_deg=0.15,
            dyn_delta_deg=0.5,
            gravity_sigma=0.25,
            jerk_sigma=45.0,
            rate_sigma=1.4,
            gate_floor=1e-4,
        )
    ]
    # Both paths use Ranger — it is stable for long 512-epoch runs (Lookahead
    # prevents the high-LR divergence that AdamW exhibits at epoch ~127).
    # Accuracy gains come from data quality and the target-gated loss, not from
    # changing the optimizer.
    opt_func = partial(
        pytorch_optimizer.Ranger,
        betas=(0.95, 0.99),
        eps=1e-6,
        weight_decay=cfg.weight_decay,
        use_gc=False,
    )
    _use_cuda_graph = cfg.cuda_graph and bool(cfg.sub_seq_len) and device.type == "cuda"
    lrn = RNNLearner(
        dls,
        loss_func=loss_func,
        metrics=[
            ignore_nan(masked_tilt_mean_deg),
            ignore_nan(masked_tilt_rmse_deg),
            ignore_nan(masked_roll_rmse_deg),
            ignore_nan(masked_pitch_rmse_deg),
            ignore_nan(masked_geodesic_mean_deg),
        ],
        aux_losses=aux_losses,
        num_layers=2,
        hidden_size=200,
        n_skip=0,
        sub_seq_len=cfg.sub_seq_len or None,
        rnn_type="gru",
        normalization="",
        input_norm=None,
        opt_func=opt_func,
        augmentations=augmentations,
        grad_clip=1.0,
        cuda_graph=_use_cuda_graph,
        device=device,
        show_bar=True,
    )
    lrn.model = ScaledModel(lrn.model, input_norm=StandardScaler(DLS_MEAN, DLS_STD))
    return lrn


def install_finite_backward_guard(lrn, logger: logging.Logger) -> None:
    lrn._finite_guard_skipped = {"count": 0}
    # Hook fires on each parameter after backward() but before optimizer.step().
    # Zeroing NaN/Inf in-place prevents a single bad batch from corrupting model
    # weights via NaN propagation through the optimizer momentum buffers.
    count = 0
    for p in lrn.model.parameters():
        if p.requires_grad:
            p.register_hook(lambda g: g.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0))
            count += 1
    logger.info("Finite backward guard: NaN/Inf gradients will be zeroed on %d parameters.", count)


class ModelEMA:
    """Exponential Moving Average of model weights (post-2021 standard practice).

    Shadow copy lives on the same device as the model — no GPU→CPU transfer per
    batch. Uses warmup decay so early random-init contamination is flushed out
    quickly: decay(t) = min(max_decay, (1+t)/(10+t)), which reaches 0.999 by
    step 10000 and asymptotes to max_decay=0.9999.  A fixed decay of 0.9999 from
    step 0 leaves 14.8% of random-init weights in the shadow after 19k steps,
    causing catastrophic drift at inference time.
    """

    def __init__(self, model: torch.nn.Module, max_decay: float = 0.9999) -> None:
        self.max_decay = float(max_decay)
        self._step = 0
        self.shadow: dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items() if torch.is_tensor(v)
        }

    @property
    def decay(self) -> float:
        return min(self.max_decay, (1.0 + self._step) / (10.0 + self._step))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        self._step += 1
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    def cpu_shadow(self) -> dict[str, torch.Tensor]:
        """Return a CPU copy of the shadow weights for checkpoint serialisation."""
        return {k: v.cpu().clone() for k, v in self.shadow.items()}


def apply_orthogonal_recurrent_init(model: torch.nn.Module, logger: logging.Logger) -> None:
    """Orthogonal init for GRU recurrent weight matrices (Henaff et al. 2016).

    PyTorch GRU defaults to Kaiming uniform for weight_hh, whose singular values
    spread over [0, sqrt(2/n)]. Orthogonal matrices have all singular values = 1,
    placing the initial gradient norm at the edge of stability rather than inside
    the vanishing regime. Safe to apply after model construction; does not change
    the model API or loss scale.
    """
    count = 0
    for name, param in model.named_parameters():
        if "weight_hh" in name and param.dim() == 2:
            torch.nn.init.orthogonal_(param)
            count += 1
    if count:
        logger.info("Orthogonal recurrent init applied to %d weight_hh matrices.", count)


def model_state_is_finite(model: torch.nn.Module) -> bool:
    for value in model.state_dict().values():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            return False
    return True


def save_checkpoint(
    path: Path,
    lrn,
    scheduler: LambdaLR,
    cfg: TrainConfig,
    epoch: int,
    best_metric: float,
    history_row: dict[str, float],
    ema: ModelEMA | None = None,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model": lrn.model.state_dict(),
        "optimizer": lrn.opt.state_dict() if lrn.opt is not None else None,
        "scheduler": scheduler.state_dict(),
        "config": {key: jsonable(value) for key, value in asdict(cfg).items()},
        "best_metric": best_metric,
        "history_row": history_row,
    }
    if ema is not None:
        checkpoint["ema"] = ema.cpu_shadow()
    torch.save(checkpoint, path)


def load_checkpoint(
    lrn, scheduler: LambdaLR, cfg: TrainConfig, logger: logging.Logger
) -> tuple[int, float]:
    if cfg.resume is None:
        return 0, float("inf")
    checkpoint = torch.load(cfg.resume, map_location="cpu", weights_only=True)
    missing, unexpected = lrn.model.load_state_dict(checkpoint["model"], strict=False)
    if missing:
        raise RuntimeError(f"resume checkpoint mismatch — missing keys: {missing}")
    if unexpected:
        logger.warning(
            "resume: ignoring %d unexpected checkpoint keys (e.g. CUDA-graph duplicates: %s ...)",
            len(unexpected),
            unexpected[0],
        )
    if cfg.resume_optimizer:
        if checkpoint.get("optimizer") is None:
            raise RuntimeError(
                "--resume-optimizer requested, but checkpoint has no optimizer state"
            )
        lrn.opt.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("scheduler") is not None and not cfg.no_resume_scheduler:
        scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = int(checkpoint.get("epoch", 0))
    best_metric = float(checkpoint.get("best_metric", float("inf")))
    logger.info(
        "Resumed %s at epoch %d with best metric %.6f", cfg.resume, start_epoch, best_metric
    )
    return start_epoch, best_metric


def append_history(path: Path, row: dict[str, float]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def metric_value(metrics: dict[str, float], name: str) -> float:
    value = metrics.get(name)
    return float("nan") if value is None else float(value)


_NAN_METRICS = {"tilt": float("nan"), "roll": float("nan"), "pitch": float("nan")}


def _batch_metric_sums(pred: torch.Tensor, target: torch.Tensor, n_skip: int) -> torch.Tensor:
    """Accumulate squared errors on GPU; caller does the single .cpu() call."""
    pred_skip = pred[:, n_skip:] if n_skip > 0 else pred
    tgt_skip = target[:, n_skip:] if n_skip > 0 else target
    q_pred = qnorm(pred_skip)
    q_target, mask = finite_target_quat(tgt_skip, q_pred)
    tilt = torch.rad2deg(inclination_error_rad(q_pred, q_target))
    pred_roll, pred_pitch = roll_pitch_deg(q_pred)
    tgt_roll, tgt_pitch = roll_pitch_deg(q_target)
    roll_err = torch.remainder(pred_roll - tgt_roll + 180.0, 360.0) - 180.0
    pitch_err = torch.remainder(pred_pitch - tgt_pitch + 180.0, 360.0) - 180.0
    v = mask.to(dtype=pred.dtype)
    return torch.stack(
        [
            (tilt.square() * v).sum(),
            (roll_err.square() * v).sum(),
            (pitch_err.square() * v).sum(),
            v.sum(),
        ]
    )


def _metric_sums_to_dict(sums: torch.Tensor) -> dict[str, float]:
    """Single .cpu() sync per batch to read accumulated GPU metric sums."""
    count = sums[3].clamp_min(1.0)
    vals = torch.sqrt(sums[:3] / count).cpu().tolist()
    return {"tilt": float(vals[0]), "roll": float(vals[1]), "pitch": float(vals[2])}


def _batch_metrics_nograd(
    lrn,
    xb: torch.Tensor,
    yb: torch.Tensor,
    use_bf16: bool = False,
    metric_samples: int = 8,
) -> dict[str, float]:
    """Low-overhead live metrics from a small sample of the last sub-sequence."""
    sub = getattr(lrn, "sub_seq_len", None)
    xb = xb[:metric_samples]
    yb = yb[:metric_samples]
    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16, cache_enabled=False),
    ):
        xb_last = xb[:, -sub:] if sub else xb
        yb_last = yb[:, -sub:] if sub else yb
        result = lrn.model(xb_last)
        pred = result[0] if isinstance(result, tuple) else result
    return _metric_sums_to_dict(_batch_metric_sums(pred.float(), yb_last.float(), 0))


def train_batch_with_metrics(
    lrn,
    xb: torch.Tensor,
    yb: torch.Tensor,
    use_bf16: bool = False,
    compute_metrics: bool = False,
    metric_samples: int = 8,
) -> tuple[float, dict[str, float]]:
    # Keep the installed tsfast TBPTT/CUDA-graph training path intact.
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16, cache_enabled=False):
        loss_val = lrn.training_step(xb, yb)
    if math.isnan(loss_val):
        return loss_val, _NAN_METRICS
    if compute_metrics:
        return loss_val, _batch_metrics_nograd(
            lrn, xb, yb, use_bf16=use_bf16, metric_samples=metric_samples
        )
    return loss_val, _NAN_METRICS


def train_one_epoch_with_progress(
    lrn,
    scheduler: LambdaLR,
    pbar: tqdm,
    epoch: int,
    n_epoch: int,
    use_bf16: bool = False,
    compute_metrics: bool = False,
    metric_every: int = 10,
    metric_samples: int = 8,
    ema: ModelEMA | None = None,
) -> float:
    lrn.model.train()
    train_losses = []
    n_batches = len(lrn.dls.train)
    total_steps = max(1, n_epoch * n_batches)
    last_metrics = _NAN_METRICS

    for batch_idx, batch in enumerate(lrn.dls.train):
        xb, yb = lrn.prepare_batch(batch, training=True)
        lrn.pct_train = (epoch * n_batches + batch_idx) / total_steps
        for _al in getattr(lrn, "aux_losses", []):
            if hasattr(_al, "pct_train"):
                _al.pct_train = lrn.pct_train
        should_update_metrics = compute_metrics and (
            batch_idx == 0 or (batch_idx + 1) % metric_every == 0 or batch_idx + 1 == n_batches
        )
        loss_val, metrics = train_batch_with_metrics(
            lrn,
            xb,
            yb,
            use_bf16=use_bf16,
            compute_metrics=should_update_metrics,
            metric_samples=metric_samples,
        )
        if not math.isnan(metrics["tilt"]):
            last_metrics = metrics
        if not math.isnan(loss_val):
            train_losses.append(loss_val)
        if ema is not None:
            ema.update(lrn.model)

        scheduler.step()
        if compute_metrics and not math.isnan(last_metrics["tilt"]):
            pbar.set_postfix_str(
                f"loss={loss_val:.4f}, "
                f"tilt_rmse={last_metrics['tilt']:.2f}, "
                f"roll_rmse={last_metrics['roll']:.2f}, "
                f"pitch_rmse={last_metrics['pitch']:.2f}, "
                f"lr={lrn.opt.param_groups[0]['lr']:.2e}",
                refresh=False,
            )
        else:
            pbar.set_postfix_str(
                f"loss={loss_val:.4f}, lr={lrn.opt.param_groups[0]['lr']:.2e}",
                refresh=False,
            )
        pbar.update(1)

    return sum(train_losses) / max(1, len(train_losses))


def validate_with_progress(
    lrn, desc: str, use_bf16: bool = False
) -> tuple[float, dict[str, float]]:
    lrn.model.eval()
    preds, targs = [], []
    with torch.inference_mode():
        pbar = tqdm(
            lrn.dls.valid,
            total=len(lrn.dls.valid),
            desc=desc,
            unit="batch",
            dynamic_ncols=True,
            bar_format=PBAR_FORMAT,
            mininterval=0.5,
        )
        for batch in pbar:
            xb, yb = lrn.prepare_batch(batch, training=False)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=use_bf16, cache_enabled=False
            ):
                result = lrn.model(xb)
            pred = result[0] if isinstance(result, tuple) else result
            preds.append(pred.float().cpu())
            targs.append(yb.cpu())
    pred_all = torch.cat(preds, dim=0)
    targ_all = torch.cat(targs, dim=0)
    pred_skip = pred_all[:, lrn.n_skip :] if lrn.n_skip > 0 else pred_all
    targ_skip = targ_all[:, lrn.n_skip :] if lrn.n_skip > 0 else targ_all
    val_loss = lrn.loss_func(pred_skip, targ_skip).item()
    metrics = {}
    for metric in lrn.metrics:
        name = getattr(metric, "__name__", type(metric).__name__)
        metrics[name] = metric(pred_skip, targ_skip).item()
    return val_loss, metrics


def train(cfg: TrainConfig) -> Path:
    run_dir = setup_run_dir(cfg)
    logger = setup_logging(run_dir)
    set_seed(cfg.seed)
    configure_torch_backend(cfg, logger)
    device = resolve_device(cfg, logger)
    logger.info("Run directory: %s", run_dir)
    logger.info("Device: %s", device)
    data_summary = summarize_data(cfg)
    logger.info("Data summary: %s", data_summary)
    logger.info(
        "RIANN contract: input=acc/gyr/dt (7), output=quaternion (4), model=2-layer GRU hidden=200"
    )
    logger.info(
        "TG-DCM stack: supervised tilt loss + target-gated DCM auxiliary loss + "
        "Ranger + BF16-on-CUDA + CUDA graph + EMA."
    )
    # TG-DCM worker auto-tuning overlaps CPU preprocessing with GPU compute.
    # Keep worker persistence disabled: this training script uses a finite
    # replacement sampler for train plus a separate validation loader each epoch.
    # On recent PyTorch builds those persistent worker pools can block the next
    # train iterator at epoch boundaries, which presents as "Train 2/... 0/60".
    if cfg.num_workers == 0 and torch.cuda.is_available():
        cfg.num_workers = min(4, os.cpu_count() or 4)
        cfg.persistent_workers = False
        logger.info(
            "auto-set DataLoader workers=%d persistent_workers=%s (PyTorch %s)",
            cfg.num_workers,
            cfg.persistent_workers,
            torch.__version__,
        )
    write_json(run_dir / "config.json", {"config": asdict(cfg), "data": data_summary})

    use_bf16 = device.type == "cuda"
    compute_metrics = True

    build_start = time.perf_counter()
    logger.info(
        "Building DataLoaders/model: batch_size=%d win_sz=%d targ_fs_count=%d "
        "train_batches=%s workers=%d",
        cfg.batch_size,
        cfg.win_sz,
        cfg.targ_fs_count,
        cfg.n_batches_train if cfg.n_batches_train is not None else "full",
        cfg.num_workers,
    )
    lrn = build_learner(cfg, device)
    logger.info(
        "DataLoaders/model built in %.1fs. train_batches=%d valid_batches=%d",
        time.perf_counter() - build_start,
        len(lrn.dls.train),
        len(lrn.dls.valid),
    )
    setup_start = time.perf_counter()
    lrn.setup(lr=cfg.lr)
    logger.info("Learner setup completed in %.1fs.", time.perf_counter() - setup_start)
    install_finite_backward_guard(lrn, logger)
    apply_orthogonal_recurrent_init(lrn.model, logger)

    # torch.compile fuses the loss's quaternion math chain into Triton kernels.
    # PyTorch 2.1.x has a Triton codegen bug (boolean int1/int8 loop-carried type
    # mismatch) that crashes on boolean reductions in the loss. Fixed in 2.3+.
    _torch_ver = tuple(int(x) for x in torch.__version__.split(".")[:2])
    _compile_ok = _torch_ver >= (2, 3) and device.type == "cuda"
    if _compile_ok:
        lrn.loss_func = torch.compile(lrn.loss_func, mode="default")
        if cfg.compile_aux_losses and getattr(lrn, "aux_losses", None):
            lrn.aux_losses = [torch.compile(aux, mode="default") for aux in lrn.aux_losses]
        logger.info(
            "loss_func compiled with torch.compile(mode='default'); aux_compile=%s aux_losses=%d.",
            cfg.compile_aux_losses,
            len(getattr(lrn, "aux_losses", [])),
        )
    elif device.type == "cuda":
        logger.info(
            "torch.compile skipped (PyTorch %s < 2.3; Triton boolean reduction bug).",
            torch.__version__,
        )
    logger.info(
        "TG-DCM backend: BF16=%s fused-AdamW=True TF32=True cuDNN-TF32=True cuDNN-benchmark=True "
        "torch.compile=%s EMA=True workers=%d persistent=%s orthogonal-init=True",
        use_bf16,
        "loss" if _compile_ok else "off",
        cfg.num_workers,
        cfg.persistent_workers,
    )
    n_batches = len(lrn.dls.train)
    # Both paths use the full epoch schedule. Ranger's Lookahead component is
    # stable at high LR for 512 epochs; AdamW at 3e-3 diverged at epoch ~127.
    _sched_epochs = cfg.epochs
    _sched_steps = _sched_epochs * n_batches
    scheduler = LambdaLR(
        lrn.opt,
        lambda step, _s=_sched_steps, _p=cfg.pct_start: max(
            0.0, sched_flat_cos(min(1.0, step / max(1, _s)), _p)
        ),
    )
    ema = ModelEMA(lrn.model, max_decay=0.9999) if device.type == "cuda" else None
    if ema is not None:
        logger.info(
            "EMA enabled (warmup decay: min(0.9999, (1+t)/(10+t)); "
            "flushes random-init contamination in first ~1k steps)."
        )

    start_epoch, best_metric = load_checkpoint(lrn, scheduler, cfg, logger)
    history_path = run_dir / "history.csv"
    stopped = False
    consecutive_bad = 0
    try:
        for epoch in range(start_epoch, cfg.epochs):
            epoch_start = time.perf_counter()

            logger.info("Starting epoch %d/%d", epoch + 1, cfg.epochs)

            # Snapshot model + optimizer before the epoch so we can roll back if NaN occurs.
            pre_epoch_model_state = {k: v.cpu().clone() for k, v in lrn.model.state_dict().items()}
            pre_epoch_opt_state = copy.deepcopy(lrn.opt.state_dict())

            with tqdm(
                total=n_batches,
                desc=f"Train {epoch + 1}/{cfg.epochs}",
                unit="batch",
                dynamic_ncols=True,
                bar_format=PBAR_FORMAT,
                mininterval=0.5,
            ) as pbar:
                train_loss = train_one_epoch_with_progress(
                    lrn,
                    scheduler,
                    pbar,
                    epoch=epoch,
                    n_epoch=cfg.epochs,
                    use_bf16=use_bf16,
                    compute_metrics=compute_metrics,
                    metric_every=cfg.metric_every,
                    metric_samples=cfg.metric_samples,
                    ema=ema,
                )
            # Validate with actual model weights — EMA shadow lags behind early in
            # training (epoch 1: ~99% initial weights at decay=0.9999 × 60 batches)
            # and would give misleading validation scores. EMA weights are saved in
            # the checkpoint for deployment-time inference only.
            val_loss, metrics = validate_with_progress(
                lrn, desc=f"Valid {epoch + 1}/{cfg.epochs}", use_bf16=use_bf16
            )
            metric_name = "masked_tilt_rmse_deg"
            score = float(metrics.get(metric_name, val_loss))
            row = {
                "epoch": epoch + 1,
                "lr": float(lrn.opt.param_groups[0]["lr"]),
                "train_loss": float(train_loss),
                "valid_loss": float(val_loss),
                **{k: float(v) for k, v in sorted(metrics.items())},
                "skipped_steps": int(getattr(lrn, "_finite_guard_skipped", {}).get("count", 0)),
            }
            append_history(history_path, row)
            write_json(run_dir / "last_metrics.json", row)
            logger.info(
                "Epoch %d/%d | train=%.6f valid=%.6f tilt_rmse=%.6f "
                "roll_rmse=%.6f pitch_rmse=%.6f geo_mean=%.6f lr=%.3e skipped=%d",
                epoch + 1,
                cfg.epochs,
                row["train_loss"],
                row["valid_loss"],
                metric_value(metrics, "masked_tilt_rmse_deg"),
                metric_value(metrics, "masked_roll_rmse_deg"),
                metric_value(metrics, "masked_pitch_rmse_deg"),
                metric_value(metrics, "masked_geodesic_mean_deg"),
                row["lr"],
                row["skipped_steps"],
            )
            logger.info("Epoch %d completed in %.1fs", epoch + 1, time.perf_counter() - epoch_start)

            epoch_bad = (
                not np.isfinite(row["train_loss"])
                or not np.isfinite(row["valid_loss"])
                or not model_state_is_finite(lrn.model)
            )
            if epoch_bad:
                consecutive_bad += 1
                logger.warning(
                    "Epoch %d non-finite (consecutive=%d): rolling back "
                    "model/optimizer and halving LR.",
                    epoch + 1,
                    consecutive_bad,
                )
                lrn.model.load_state_dict(
                    {k: v.to(device) for k, v in pre_epoch_model_state.items()}
                )
                lrn.opt.load_state_dict(pre_epoch_opt_state)
                for pg in lrn.opt.param_groups:
                    pg["lr"] *= 0.5
                if consecutive_bad >= 5:
                    logger.error(
                        "5 consecutive non-finite epochs despite recovery; stopping training."
                    )
                    stopped = True
                    break
                continue

            consecutive_bad = 0

            # Early stopping: once LR has been at the floor for 5 epochs, the
            # schedule is exhausted and no further weight updates will occur.
            # Stop rather than burning compute on dead epochs.
            if row["lr"] < cfg.lr * 1e-3:
                epochs_at_floor = getattr(train, "_epochs_at_lr_floor", 0) + 1
                # type: ignore[attr-defined]
                train._epochs_at_lr_floor = epochs_at_floor
                if epochs_at_floor >= 5:
                    logger.info(
                        "LR has been at floor (<%.1e) for 5 consecutive epochs; stopping early.",
                        cfg.lr * 1e-3,
                    )
                    stopped = True
            else:
                train._epochs_at_lr_floor = 0  # type: ignore[attr-defined]

            if score < best_metric:
                best_metric = score
                save_checkpoint(
                    run_dir / "best.pt", lrn, scheduler, cfg, epoch + 1, best_metric, row, ema=ema
                )
                write_json(run_dir / "best_metrics.json", row)
                logger.info("Saved new best checkpoint with %s=%.6f", metric_name, best_metric)
            save_checkpoint(
                run_dir / "last.pt", lrn, scheduler, cfg, epoch + 1, best_metric, row, ema=ema
            )
            if cfg.save_every > 0 and (epoch + 1) % cfg.save_every == 0:
                save_checkpoint(
                    run_dir / f"epoch_{epoch + 1:04d}.pt",
                    lrn,
                    scheduler,
                    cfg,
                    epoch + 1,
                    best_metric,
                    row,
                    ema=ema,
                )
    finally:
        lrn._teardown_composables()

    write_json(
        run_dir / "summary.json",
        {"stopped": stopped, "best_metric": best_metric, "run_dir": str(run_dir)},
    )
    logger.info("Training finished. best_%s=%.6f", "masked_tilt_rmse_deg", best_metric)
    return run_dir


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
