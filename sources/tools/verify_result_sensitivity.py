from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"seq_id", "dataset", "trial_group", "model_tilt_rmse", "riann_tilt_rmse", "vqf_tilt_rmse", "duration_s", "n_valid", "gate_frac"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Missing required per-sequence columns or rows")
    return rows


def estimate(rows: list[dict[str, str]], group, reps: int, seed: int) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        model = float(row["model_tilt_rmse"])
        baseline = float(row["riann_tilt_rmse"])
        if math.isfinite(model) and math.isfinite(baseline):
            grouped[group(row)].append(model - baseline)
    groups = list(grouped.values())
    if not groups:
        return {"n": 0, "clusters": 0, "mean_delta_deg": None, "ci95_deg": None}
    sums = [sum(g) for g in groups]
    counts = [len(g) for g in groups]
    observed = sum(sums) / sum(counts)
    rng = random.Random(seed)
    samples = []
    for _ in range(reps):
        numer = 0.0
        denom = 0
        for _ in groups:
            i = rng.randrange(len(groups))
            numer += sums[i]
            denom += counts[i]
        samples.append(numer / denom)
    samples.sort()

    def pct(x: float) -> float:
        pos = (len(samples) - 1) * x
        lo = int(pos)
        hi = min(lo + 1, len(samples) - 1)
        return samples[lo] + (samples[hi] - samples[lo]) * (pos - lo)

    return {"n": sum(counts), "clusters": len(groups), "mean_delta_deg": observed, "ci95_deg": [pct(0.025), pct(0.975)]}


def caruso_pairs(rows: list[dict[str, str]]) -> dict:
    canonical = {row["seq_id"]: row for row in rows if row["dataset"] == "Caruso-Sassari"}
    paired = [(canonical[row["seq_id"]], row) for row in rows if row["dataset"] == "Caruso-Sassari_orig" and row["seq_id"] in canonical]
    if not paired:
        return {"pairs": 0}

    def absdiff(field: str) -> float:
        return sum(abs(float(a[field]) - float(b[field])) for a, b in paired) / len(paired)

    return {
        "pairs": len(paired),
        "mean_abs_delta_deg": {"tg": absdiff("model_tilt_rmse"), "riann": absdiff("riann_tilt_rmse"), "vqf": absdiff("vqf_tilt_rmse")},
        "mean_abs_gate_fraction_delta": absdiff("gate_frac"),
        "nonidentical_valid_count_pairs": sum(int(a["n_valid"]) != int(b["n_valid"]) for a, b in paired),
        "nonidentical_tg_rmse_pairs": sum(abs(float(a["model_tilt_rmse"]) - float(b["model_tilt_rmse"])) > 1e-6 for a, b in paired),
    }


def run(path: Path, reps: int, seed: int) -> dict:
    rows = read_rows(path)
    standard = [r for r in rows if r["dataset"].lower() != "diodem"]
    without_orig = [r for r in standard if r["dataset"] != "Caruso-Sassari_orig"]
    diodem = [r for r in rows if r["dataset"].lower() == "diodem"]

    def experiment(row: dict[str, str]) -> str:
        hit = re.match(r"exp\d+", row["trial_group"], re.IGNORECASE)
        if hit is None:
            raise ValueError("Unrecognized DIODEM trial group: " + row["trial_group"])
        return hit.group(0).lower()

    return {
        "source": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bootstrap": {"method": "paired cluster resampling, size-preserving cluster ratios", "replications": reps, "seed": seed},
        "standard_all": estimate(standard, lambda r: r["dataset"] + ":" + r["seq_id"], reps, seed),
        "standard_without_caruso_orig": estimate(without_orig, lambda r: r["dataset"] + ":" + r["seq_id"], reps, seed),
        "caruso": caruso_pairs(rows),
        "diodem_motion_cluster": estimate(diodem, lambda r: r["trial_group"], reps, seed),
        "diodem_experiment_cluster": estimate(diodem, experiment, reps, seed),
        "diodem_experiment_count": len({experiment(r) for r in diodem}),
        "diodem_motion_count": len({r["trial_group"] for r in diodem}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("results/full_evaluation/tables/per_sequence.csv"))
    parser.add_argument("--reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.reps < 100:
        raise ValueError("Expected at least 100 bootstrap replications")
    print(json.dumps(run(args.csv, args.reps, args.seed), indent=2))


if __name__ == "__main__":
    main()
