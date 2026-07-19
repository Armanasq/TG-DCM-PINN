#!/usr/bin/env python3
"""Train the TG-DCM configuration.

This wrapper fixes the training stack to the released TG-DCM configuration:
2-layer GRU, supervised tilt loss, and target-gated DCM auxiliary loss.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tg_dcm.train_impl import TrainConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TG-DCM.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--run-name", default="tg_dcm")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_batches_train=300,
        targ_fs_count=427,
        sub_seq_len=1800,
        num_workers=args.num_workers,
        prefetch_factor=2,
        persistent_workers=False,
        seed=args.seed,
        out_dir=args.out_dir,
        run_name=args.run_name,
        data_dir=args.data_dir,
        resume=args.resume,
        device=args.device,
        cuda_graph=True,
        use_cudnn=True,
        stop_on_nonfinite=True,
        save_every=1,
        metric_every=10,
        metric_samples=8,
        compile_aux_losses=True,
        smoke=args.smoke,
    )
    train(cfg)


if __name__ == "__main__":
    main()
