from __future__ import annotations

import json

import numpy as np

from tg_dcm.evaluation import (
    build_effects_table,
    build_summary_tables,
    cluster_bootstrap_mean_ci,
    infer_diodem_motion_family,
    infer_trial_group,
    results_frame,
    write_evaluation_artifacts,
)


def _records():
    return [
        {
            "dataset": "A",
            "seq_id": "a1",
            "model_tilt_rmse": 1.0,
            "riann_tilt_rmse": 2.0,
            "vqf_tilt_rmse": 3.0,
        },
        {
            "dataset": "A",
            "seq_id": "a2",
            "model_tilt_rmse": 3.0,
            "riann_tilt_rmse": 2.0,
            "vqf_tilt_rmse": 4.0,
        },
    ]


def test_diodem_group_and_motion_taxonomy() -> None:
    assert infer_trial_group("diodem", "exp04_motion07_dangle_global2_seg5") == "exp04_motion07"
    assert infer_diodem_motion_family("exp04_motion07_dangle_global2_seg5") == "Dangle"
    assert infer_diodem_motion_family("exp02_motion07_slow_fast_freeze_mix_seg1") == "Mixed rate"
    assert infer_diodem_motion_family("exp01_motion13_pause4_seg1") == "Pause"


def test_cluster_bootstrap_respects_single_cluster() -> None:
    low, high = cluster_bootstrap_mean_ci(
        np.array([1.0, 3.0]), np.array(["trial", "trial"]), reps=100, seed=1
    )
    assert low == high == 2.0


def test_summary_tables_report_paired_wins() -> None:
    frame = results_frame(_records())
    per_dataset, overall, reliability = build_summary_tables(frame)
    assert per_dataset.loc[0, "tg_dcm_mean_deg"] == 2.0
    assert per_dataset.loc[0, "win_rate_vs_riann_pct"] == 50.0
    assert overall.loc[0, "mean_delta_vs_vqf_deg"] == -1.5
    assert set(reliability["method"]) == {"TG-DCM", "RIANN", "VQF"}
    effects = build_effects_table(frame)
    row = effects[(effects["scope"] == "standard") & (effects["baseline"] == "RIANN")].iloc[0]
    assert row["mean_delta_deg"] == 0.0
    assert row["win_rate_pct"] == 50.0


def test_artifacts_use_strict_json(tmp_path) -> None:
    frame = results_frame(_records())
    paths = write_evaluation_artifacts(frame, tmp_path)
    payload = json.loads(paths["summary"].read_text())
    assert payload["schema_version"] == 1
    assert payload["n_sequences"] == 2
    assert all(path.is_file() for path in paths.values())


def test_no_model_results_still_write_artifacts(tmp_path) -> None:
    records = _records()
    for record in records:
        record["model_tilt_rmse"] = np.nan
    paths = write_evaluation_artifacts(results_frame(records), tmp_path)
    payload = json.loads(paths["summary"].read_text())
    assert payload["overall"]["tg_dcm_mean_deg"] is None
