"""Regenerate TG-DCM evaluation figures from a saved per-sequence table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tg_dcm.evaluation import generate_evaluation_figures, results_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate repository and poster figures from per_sequence.csv."
    )
    parser.add_argument(
        "results",
        nargs="?",
        default=ROOT / "evaluation" / "tables" / "per_sequence.csv",
        help="Per-sequence CSV produced by scripts/evaluate.py.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Figure root. Default: two levels above the input table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results).resolve()
    if not results_path.is_file():
        raise FileNotFoundError(f"evaluation table not found: {results_path}")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else results_path.parent.parent
    frame = results_frame(pd.read_csv(results_path).to_dict(orient="records"))
    figures = generate_evaluation_figures(frame, out_dir)
    for figure in figures:
        print(figure)


if __name__ == "__main__":
    main()
