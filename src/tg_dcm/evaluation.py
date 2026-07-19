"""Evaluation summaries and publication figures for TG-DCM.

The functions in this module operate on one row per evaluated sequence. They
are independent of model inference so an evaluation can be replotted without
loading a checkpoint or rerunning a baseline.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

METHODS: Mapping[str, tuple[str, str]] = {
    "TG-DCM": ("model_tilt_rmse", "#0072B2"),
    "RIANN": ("riann_tilt_rmse", "#D55E00"),
    "VQF": ("vqf_tilt_rmse", "#009E73"),
}
GRID = "#D7D7D7"
TEXT = "#252525"
MUTED = "#666666"
ZERO = "#8A1C1C"


def configure_matplotlib() -> None:
    """Set deterministic, vector-export-friendly figure defaults."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.75,
            "axes.edgecolor": "#555555",
            "axes.labelcolor": TEXT,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def infer_trial_group(dataset: str, seq_id: str) -> str:
    """Return the independent resampling unit for one sequence.

    DIODEM has five segment files per raw motion. Its confidence intervals are
    therefore clustered by experiment and motion. Other datasets use one
    sequence per cluster.
    """
    if dataset.lower() == "diodem":
        match = re.match(r"(exp\d+_motion\d+)", seq_id, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return f"{dataset}:{seq_id}"


def infer_diodem_motion_family(seq_id: str) -> str:
    """Map the DIODEM motion label to a compact analysis taxonomy."""
    match = re.match(r"exp\d+_motion\d+_(.+)_seg\d+$", seq_id, flags=re.IGNORECASE)
    label = match.group(1).lower() if match else seq_id.lower()
    if "pause" in label:
        return "Pause"
    if "dangle" in label:
        return "Dangle"
    if "gait" in label:
        return "Gait"
    if "canonical" in label:
        return "Canonical"
    if "slow" in label and "fast" in label:
        return "Mixed rate"
    if "quasistatic" in label or "freeze" in label:
        return "Static / freeze"
    if "slow" in label:
        return "Slow"
    if "fast" in label or "explosiv" in label or "shaking" in label:
        return "Fast / impulsive"
    return "Other"


def results_frame(results: Iterable[object]) -> pd.DataFrame:
    """Convert result dataclasses or dictionaries to the stable table schema."""
    records = []
    for result in results:
        if is_dataclass(result):
            record = asdict(result)
        elif isinstance(result, Mapping):
            record = dict(result)
        else:
            raise TypeError(f"unsupported result type: {type(result)!r}")
        records.append(record)
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    required = {"dataset", "seq_id", *(column for column, _ in METHODS.values())}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing evaluation columns: {sorted(missing)}")
    if "trial_group" not in frame:
        frame["trial_group"] = [
            infer_trial_group(dataset, seq_id)
            for dataset, seq_id in zip(frame["dataset"], frame["seq_id"])
        ]
    return frame.sort_values(["dataset", "seq_id"], kind="stable").reset_index(drop=True)


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def cluster_bootstrap_mean_ci(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    reps: int = 8000,
    seed: int = 7,
) -> tuple[float, float]:
    """Bootstrap a mean while preserving within-trial dependence."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    valid = np.isfinite(values)
    values, groups = values[valid], groups[valid]
    if values.size == 0:
        return math.nan, math.nan

    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse).astype(np.float64)
    if unique.size == 1:
        mean = float(sums[0] / counts[0])
        return mean, mean

    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, unique.size, size=(reps, unique.size))
    means = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _summary_row(frame: pd.DataFrame, dataset: str) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "dataset": dataset,
        "n_sequences": int(len(frame)),
        "n_trials": int(frame["trial_group"].nunique()),
    }
    for method, (column, _) in METHODS.items():
        values = _finite(frame[column])
        key = method.lower().replace("-", "_")
        row[f"{key}_mean_deg"] = float(np.mean(values)) if values.size else math.nan
        row[f"{key}_median_deg"] = float(np.median(values)) if values.size else math.nan
        row[f"{key}_p95_deg"] = float(np.percentile(values, 95)) if values.size else math.nan
        row[f"{key}_max_deg"] = float(np.max(values)) if values.size else math.nan
    model = frame["model_tilt_rmse"].to_numpy(float)
    for baseline in ("riann", "vqf"):
        base = frame[f"{baseline}_tilt_rmse"].to_numpy(float)
        valid = np.isfinite(model) & np.isfinite(base)
        row[f"win_rate_vs_{baseline}_pct"] = (
            float(np.mean(model[valid] < base[valid]) * 100.0) if valid.any() else math.nan
        )
        row[f"mean_delta_vs_{baseline}_deg"] = (
            float(np.mean(model[valid] - base[valid])) if valid.any() else math.nan
        )
    return row


def build_summary_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build dataset, overall, and method-reliability tables."""
    per_dataset = pd.DataFrame(
        [
            _summary_row(group, str(dataset))
            for dataset, group in frame.groupby("dataset", sort=True)
        ]
    )
    overall = pd.DataFrame([_summary_row(frame, "overall")])

    reliability_rows = []
    for dataset, group in [("overall", frame), *frame.groupby("dataset", sort=True)]:
        for method, (column, _) in METHODS.items():
            values = _finite(group[column])
            if not values.size:
                continue
            reliability_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n_sequences": int(values.size),
                    "mean_deg": float(np.mean(values)),
                    "std_deg": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "median_deg": float(np.median(values)),
                    "p95_deg": float(np.percentile(values, 95)),
                    "max_deg": float(np.max(values)),
                    "cv": float(np.std(values, ddof=1) / np.mean(values))
                    if values.size > 1 and np.mean(values) > 0
                    else 0.0,
                }
            )
    return per_dataset, overall, pd.DataFrame(reliability_rows)


def build_effects_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired TG-DCM effects with cluster-bootstrap intervals."""
    standard = frame[frame["dataset"].str.lower() != "diodem"]
    scopes = [("overall", frame), ("standard", standard)]
    scopes.extend((str(dataset), group) for dataset, group in frame.groupby("dataset", sort=True))
    rows = []
    for scope_index, (scope, group) in enumerate(scopes):
        if group.empty:
            continue
        model = group["model_tilt_rmse"].to_numpy(float)
        for baseline_index, (baseline, column) in enumerate(
            [("RIANN", "riann_tilt_rmse"), ("VQF", "vqf_tilt_rmse")]
        ):
            reference = group[column].to_numpy(float)
            valid = np.isfinite(model) & np.isfinite(reference)
            delta = model[valid] - reference[valid]
            groups = group.loc[valid, "trial_group"].to_numpy()
            if delta.size == 0:
                continue
            low, high = cluster_bootstrap_mean_ci(
                delta,
                groups,
                seed=3100 + scope_index * 67 + baseline_index * 19,
            )
            rows.append(
                {
                    "scope": scope,
                    "baseline": baseline,
                    "n_sequences": int(valid.sum()),
                    "n_trials": int(pd.Series(groups).nunique()),
                    "mean_delta_deg": float(np.mean(delta)),
                    "ci95_low_deg": low,
                    "ci95_high_deg": high,
                    "median_delta_deg": float(np.median(delta)),
                    "win_rate_pct": float(np.mean(delta < 0) * 100.0),
                }
            )
    return pd.DataFrame(rows)


def write_evaluation_artifacts(frame: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Write machine-readable evaluation outputs and return their paths."""
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    per_dataset, overall, reliability = build_summary_tables(frame)
    effects = build_effects_table(frame)
    paths = {
        "per_sequence": tables_dir / "per_sequence.csv",
        "per_dataset": tables_dir / "per_dataset.csv",
        "overall": tables_dir / "overall.csv",
        "reliability": tables_dir / "reliability.csv",
        "effects": tables_dir / "effects.csv",
        "summary": tables_dir / "summary.json",
    }
    frame.to_csv(paths["per_sequence"], index=False, float_format="%.8g")
    per_dataset.to_csv(paths["per_dataset"], index=False, float_format="%.8g")
    overall.to_csv(paths["overall"], index=False, float_format="%.8g")
    reliability.to_csv(paths["reliability"], index=False, float_format="%.8g")
    effects.to_csv(paths["effects"], index=False, float_format="%.8g")
    payload = {
        "schema_version": 1,
        "primary_metric": "yaw-invariant tilt RMSE in degrees",
        "n_sequences": int(len(frame)),
        "n_datasets": int(frame["dataset"].nunique()),
        "datasets": per_dataset.to_dict(orient="records"),
        "overall": overall.iloc[0].to_dict(),
        "effects": effects.to_dict(orient="records"),
    }

    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        if isinstance(value, np.integer):
            return int(value)
        return value

    paths["summary"].write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n")
    return paths


def _academic_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.7)
    ax.tick_params(which="minor", length=2.0, width=0.55)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.985,
        0.96,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontweight="bold",
        fontsize=10.0,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
        zorder=10,
    )


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_dataset_trials(frame: pd.DataFrame, out_dir: Path) -> Path:
    """Plot every sequence, grouped by dataset, with method distributions."""
    configure_matplotlib()
    datasets = list(frame.groupby("dataset", sort=True).groups)
    fig, axes = plt.subplots(
        len(datasets), 1, figsize=(10.8, max(7.0, 1.55 * len(datasets))), layout="constrained"
    )
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(17)
    offsets = [-0.22, 0.0, 0.22]
    for row, (ax, dataset) in enumerate(zip(axes, datasets)):
        group = frame[frame["dataset"] == dataset]
        for offset, (method, (column, color)) in zip(offsets, METHODS.items()):
            values = group[column].to_numpy(float)
            jitter = rng.uniform(-0.055, 0.055, size=len(values))
            ax.scatter(
                np.full(len(values), offset) + jitter,
                values,
                s=12,
                facecolor=color,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.58,
                rasterized=True,
            )
            finite = values[np.isfinite(values)]
            if finite.size:
                q1, median, q3 = np.percentile(finite, [25, 50, 75])
                ax.vlines(offset, q1, q3, color=color, linewidth=4.5, zorder=4)
                ax.scatter(
                    offset,
                    median,
                    s=28,
                    marker="D",
                    color=color,
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=5,
                )
        ax.set_yscale("log")
        ax.set_xlim(-0.48, 0.48)
        ax.set_xticks(offsets, METHODS)
        ax.set_ylabel("Tilt RMSE (deg)")
        ax.set_title(
            f"{dataset}  |  {len(group)} sequences, {group['trial_group'].nunique()} trials",
            loc="left",
            fontweight="bold",
        )
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(NullFormatter())
        _academic_axes(ax)
        _panel_label(ax, f"({chr(97 + row)})")
    fig.suptitle(
        "Sequence-level tilt error across evaluation datasets", fontsize=13.0, fontweight="bold"
    )
    stem = Path(out_dir) / "repository" / "all_trials_by_dataset"
    _save_figure(fig, stem)
    return stem.with_suffix(".png")


def _bootstrap_for_column(
    group: pd.DataFrame, column: str, seed: int
) -> tuple[float, float, float]:
    values = group[column].to_numpy(float)
    if not np.isfinite(values).any():
        return math.nan, math.nan, math.nan
    mean = float(np.nanmean(values))
    low, high = cluster_bootstrap_mean_ci(values, group["trial_group"].to_numpy(), seed=seed)
    return mean, low, high


def plot_poster_summary(frame: pd.DataFrame, out_dir: Path) -> Path:
    """Create a compact four-panel figure sized for the A0 poster results block."""
    configure_matplotlib()
    display_name = {
        "TG-DCM": "TG-DCM-PINN",
        "RIANN": "Reference NN",
        "VQF": "Reference filter",
    }
    fig = plt.figure(figsize=(12.0, 8.0), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.15, 1.0])
    ax_mean = fig.add_subplot(gs[0, 0])
    ax_delta = fig.add_subplot(gs[0, 1])
    ax_ecdf = fig.add_subplot(gs[1, 0])
    ax_regime = fig.add_subplot(gs[1, 1])

    datasets = list(frame.groupby("dataset", sort=True).groups)
    ypos = np.arange(len(datasets))[::-1]
    method_offsets = {"TG-DCM": 0.22, "RIANN": 0.0, "VQF": -0.22}
    for method_index, (method, (column, color)) in enumerate(METHODS.items()):
        for y, dataset in zip(ypos, datasets):
            group = frame[frame["dataset"] == dataset]
            mean, low, high = _bootstrap_for_column(group, column, 100 + method_index * 31 + y)
            if not np.isfinite(mean):
                continue
            ax_mean.errorbar(
                mean,
                y + method_offsets[method],
                xerr=[[mean - low], [high - mean]],
                fmt="o",
                ms=4.2,
                capsize=2.0,
                elinewidth=1.15,
                color=color,
                label=display_name[method] if y == ypos[0] else None,
            )
    ax_mean.set_xscale("log")
    ax_mean.set_yticks(ypos, datasets)
    ax_mean.set_xlabel("Mean sequence tilt RMSE (deg, log scale)")
    ax_mean.set_title("Dataset performance", loc="left", fontweight="bold")
    ax_mean.legend(frameon=False, ncol=3, loc="lower right")
    _academic_axes(ax_mean, grid_axis="x")
    _panel_label(ax_mean, "(a)")

    baseline_specs = [
        ("Reference NN", "riann_tilt_rmse", "#D55E00", 0.13),
        ("Reference filter", "vqf_tilt_rmse", "#009E73", -0.13),
    ]
    for baseline_index, (label, column, color, offset) in enumerate(baseline_specs):
        for y, dataset in zip(ypos, datasets):
            group = frame[frame["dataset"] == dataset]
            delta = group["model_tilt_rmse"].to_numpy(float) - group[column].to_numpy(float)
            if not np.isfinite(delta).any():
                continue
            mean = float(np.nanmean(delta))
            low, high = cluster_bootstrap_mean_ci(
                delta, group["trial_group"].to_numpy(), seed=800 + baseline_index * 43 + y
            )
            ax_delta.errorbar(
                mean,
                y + offset,
                xerr=[[mean - low], [high - mean]],
                fmt="o",
                ms=4.2,
                capsize=2.0,
                elinewidth=1.15,
                color=color,
                label=f"vs {label}" if y == ypos[0] else None,
            )
    ax_delta.axvline(0.0, color=ZERO, linewidth=0.9, linestyle=(0, (4, 3)))
    ax_delta.set_xscale("symlog", linthresh=0.5, linscale=1.0)
    ax_delta.set_xticks([-30, -10, -3, -1, 0, 1, 3])
    ax_delta.set_yticks(ypos, datasets)
    ax_delta.set_xlabel("Paired difference (TG-DCM-PINN - reference), deg")
    ax_delta.set_title("Paired effect", loc="left", fontweight="bold")
    handles, labels = ax_delta.get_legend_handles_labels()
    if handles:
        ax_delta.legend(handles, labels, frameon=False, ncol=2, loc="lower right")
    ax_delta.text(
        0.01, 0.02, "TG-DCM-PINN better", transform=ax_delta.transAxes, color=MUTED, fontsize=8.0
    )
    _academic_axes(ax_delta, grid_axis="x")
    _panel_label(ax_delta, "(b)")

    for method, (column, color) in METHODS.items():
        values = np.sort(_finite(frame[column]))
        if values.size == 0:
            continue
        probability = np.arange(1, len(values) + 1) / len(values)
        ax_ecdf.step(
            values,
            probability,
            where="post",
            color=color,
            linewidth=1.6,
            label=display_name[method],
        )
    ax_ecdf.set_xscale("log")
    all_errors = np.concatenate([_finite(frame[column]) for column, _ in METHODS.values()])
    ax_ecdf.set_xlim(left=max(0.1, float(np.min(all_errors)) * 0.8))
    ax_ecdf.set_ylim(0, 1)
    ax_ecdf.set_xlabel("Sequence tilt RMSE (deg, log scale)")
    ax_ecdf.set_ylabel("Fraction of sequences at or below error")
    ax_ecdf.set_title("Performance profile", loc="left", fontweight="bold")
    ax_ecdf.legend(frameon=False, loc="lower right")
    _academic_axes(ax_ecdf)
    _panel_label(ax_ecdf, "(c)")

    diodem = frame[frame["dataset"].str.lower() == "diodem"].copy()
    diodem["motion_family"] = diodem["seq_id"].map(infer_diodem_motion_family)
    family_stats = []
    for family, group in diodem.groupby("motion_family", sort=False):
        delta = group["model_tilt_rmse"].to_numpy(float) - group["riann_tilt_rmse"].to_numpy(float)
        valid_delta = np.isfinite(delta)
        delta = delta[valid_delta]
        if delta.size == 0:
            continue
        low, high = cluster_bootstrap_mean_ci(
            delta,
            group.loc[valid_delta, "trial_group"].to_numpy(),
            seed=1900 + len(family_stats) * 37,
        )
        family_stats.append(
            (
                family,
                float(np.nanmean(delta)),
                low,
                high,
                int(group.loc[valid_delta, "trial_group"].nunique()),
                float(np.mean(delta < 0) * 100.0),
            )
        )
    family_stats.sort(key=lambda item: item[1])
    family_y = np.arange(len(family_stats))[::-1]
    for y, (family, mean, low, high, n_trials, win_rate) in zip(family_y, family_stats):
        color = "#0072B2" if mean < 0 else "#D55E00"
        ax_regime.errorbar(
            mean,
            y,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            ms=4.5,
            capsize=2.0,
            elinewidth=1.2,
            color=color,
        )
        ax_regime.text(
            0.985,
            y,
            f"{n_trials} trials | {win_rate:.0f}% wins",
            transform=ax_regime.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7.0,
            color=MUTED,
        )
    ax_regime.axvline(0.0, color=ZERO, linewidth=0.9, linestyle=(0, (4, 3)))
    ax_regime.set_yticks(family_y, [item[0] for item in family_stats])
    if not family_stats:
        ax_regime.text(
            0.5,
            0.5,
            "TG-DCM result unavailable",
            transform=ax_regime.transAxes,
            ha="center",
            color=MUTED,
        )
    ax_regime.set_xlabel("Paired difference vs reference NN (deg)")
    ax_regime.set_title("DIODEM transfer by motion family", loc="left", fontweight="bold")
    ax_regime.text(
        0.01, 0.02, "TG-DCM-PINN better", transform=ax_regime.transAxes, color=MUTED, fontsize=8.0
    )
    ax_regime.xaxis.set_minor_locator(AutoMinorLocator(2))
    _academic_axes(ax_regime, grid_axis="x")
    _panel_label(ax_regime, "(d)")

    fig.suptitle(
        "TG-DCM-PINN evaluation across six-axis inertial datasets", fontsize=13.2, fontweight="bold"
    )
    fig.text(
        0.5,
        -0.015,
        "Points are sequence-level results. Error bars are 95% cluster-bootstrap "
        "intervals; DIODEM is clustered by raw motion. Motion-family labels are "
        "derived from DIODEM metadata names; family effects are descriptive.",
        ha="center",
        va="top",
        fontsize=8.0,
        color=MUTED,
    )
    stem = Path(out_dir) / "poster" / "evaluation_summary"
    _save_figure(fig, stem)
    return stem.with_suffix(".png")


def generate_evaluation_figures(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Generate repository and poster figure sets from one result table."""
    return [plot_dataset_trials(frame, out_dir), plot_poster_summary(frame, out_dir)]
