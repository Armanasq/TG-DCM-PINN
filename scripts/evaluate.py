"""Evaluate TG-DCM against RIANN and VQF.

Primary metric: tilt RMSE = 2*arccos(sqrt(w^2 + z^2)).
The metric is yaw-invariant and uses scalar-first Hamilton quaternions.

Secondary metrics: roll RMSE and pitch RMSE after heading alignment.

Examples:
    python scripts/evaluate.py --data-dir ../data --datasets OxIOD Myon
    python scripts/evaluate.py --data-dir ../data --datasets diodem --diodem-root ../diodem
    python scripts/evaluate.py --data-dir ../data --datasets OxIOD EuRoC-MAV diodem
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as rt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tg_dcm.data import DATA_DIR, get_files
from tg_dcm.evaluation import (
    configure_matplotlib,
    generate_evaluation_figures,
    infer_trial_group,
    results_frame,
    write_evaluation_artifacts,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CKPT = ROOT / "checkpoints" / "tg_dcm.pt"


def default_riann_onnx() -> Path:
    """Resolve the official RIANN ONNX file from RIANN_ONNX or site-packages."""
    env_path = os.environ.get("RIANN_ONNX")
    if env_path:
        return Path(env_path)
    spec = importlib.util.find_spec("riann")
    if spec is None or spec.origin is None:
        return Path("riann.onnx")
    return Path(spec.origin).resolve().parent / "riann.onnx"


RIANN_ONNX = default_riann_onnx()

# Same normalization constants as training (DLS_MEAN / DLS_STD from train.py).
# Both the official RIANN ONNX and our ScaledModel use these; the ONNX has
# them baked in, our model applies them via input_norm.
INPUT_MEAN = np.array([-0.6845, -0.4654, 0.2514, -0.0715, 0.0343, 0.1019, 0.0035], dtype=np.float32)
INPUT_STD = np.array([7.1055, 7.0015, 7.1748, 2.0147, 1.9856, 2.0225, 1.8847e-03], dtype=np.float32)

GRAVITY_MAG = 9.81
STATIC_THR = 0.5
N_ALIGN = 100  # frames for world-frame heading alignment (roll/pitch only)
CHUNK_SIZE = 50_000  # GRU inference chunk — keeps GPU memory bounded
BACKEND_WORKERS = 3  # RIANN ONNX + VQF + our model can run concurrently.

_RIANN_SESSION: rt.InferenceSession | None = None
_RIANN_SESSION_LOCK = threading.Lock()

COLORS = {
    "model": "#0072B2",
    "riann": "#D55E00",
    "vqf": "#009E73",
    "gt": "#222222",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading  (channels stored separately: acc_x, acc_y, … opt_a … dt)
# ─────────────────────────────────────────────────────────────────────────────


def load_h5(path: Path) -> dict:
    """Load one HDF5 sequence into a dict with stacked numpy arrays."""
    with h5py.File(path, "r") as f:
        if all(
            k in f
            for k in (
                "acc_x",
                "acc_y",
                "acc_z",
                "gyr_x",
                "gyr_y",
                "gyr_z",
                "opt_a",
                "opt_b",
                "opt_c",
                "opt_d",
                "dt",
            )
        ):
            acc = np.stack([f["acc_x"][:], f["acc_y"][:], f["acc_z"][:]], axis=-1).astype(
                np.float32
            )
            gyr = np.stack([f["gyr_x"][:], f["gyr_y"][:], f["gyr_z"][:]], axis=-1).astype(
                np.float32
            )
            quat = np.stack(
                [f["opt_a"][:], f["opt_b"][:], f["opt_c"][:], f["opt_d"][:]], axis=-1
            ).astype(np.float32)
            dt = f["dt"][:].astype(np.float32)  # per-sample Δt [s]
        elif all(k in f for k in ("acc", "gyr", "quat", "timestamps")):
            acc = f["acc"][:].astype(np.float32)
            gyr = f["gyr"][:].astype(np.float32)
            quat = f["quat"][:].astype(np.float32)  # DIODEM stores wxyz, body_to_world, ENU.
            ts = f["timestamps"][:].astype(np.float64)
            if len(ts) > 1:
                dts = np.diff(ts)
                good_dt = dts[np.isfinite(dts) & (dts > 0.0)]
                dt_med = float(np.median(good_dt)) if len(good_dt) else 1.0 / 40.0
                dt = np.empty(len(ts), dtype=np.float32)
                dt[0] = dt_med
                dt[1:] = np.where(np.isfinite(dts) & (dts > 0.0), dts, dt_med).astype(np.float32)
            else:
                dt = np.full(len(ts), 1.0 / 40.0, dtype=np.float32)
        else:
            keys = ", ".join(sorted(f.keys()))
            raise KeyError(f"unsupported HDF5 schema: {keys}")
    fs = float(1.0 / float(np.median(dt)))
    return {"acc": acc, "gyr": gyr, "quat": quat, "dt": dt, "fs": fs, "T": len(acc)}


def discover_diodem_files(root: Path, include_questionable: bool = False) -> list[Path]:
    """Discover DIODEM segment files from either the repo root or data subdir."""
    root = Path(root)
    data_root = root / "diodem" if (root / "diodem").is_dir() else root
    paths = sorted(data_root.glob("exp*/*.h5"))
    if include_questionable:
        paths.extend(sorted(data_root.glob("exp*/questionable/*.h5")))
    return sorted(paths)


def validate_sequence(seq: dict, seq_id: str) -> dict:
    """Basic sanity-checks + nan masking."""
    acc, gyr, quat, dt = seq["acc"], seq["gyr"], seq["quat"], seq["dt"]
    n = min(len(acc), len(gyr), len(quat), len(dt))
    if n == 0:
        raise ValueError(f"{seq_id}: empty sequence")
    acc, gyr, quat, dt = acc[:n], gyr[:n], quat[:n], dt[:n]

    finite = (
        np.isfinite(acc).all(axis=1) & np.isfinite(gyr).all(axis=1) & np.isfinite(quat).all(axis=1)
    )
    # Keep raw acc/gyr even for optical-dropout frames — the IMU never stops recording
    # and the GRU hidden state must receive real sensor data to track orientation correctly.
    # Zeroing acc/gyr corrupts the hidden state and causes large tilt spikes on subsequent frames.
    acc = np.where(np.isfinite(acc), acc, 0.0).astype(np.float32)
    gyr = np.where(np.isfinite(gyr), gyr, 0.0).astype(np.float32)
    quat = qnorm_np(
        np.where(finite[:, None], quat, np.array([1, 0, 0, 0], np.float32)).astype(np.float32)
    )
    valid_mask = finite

    acc_med = (
        float(np.median(np.linalg.norm(acc[finite], axis=1))) if finite.any() else float("nan")
    )
    if np.isfinite(acc_med) and not (6.0 <= acc_med <= 13.5):
        print(f"  [warn] {seq_id}: median |acc|={acc_med:.2f} m/s² (expected ~9.81)")
    gyr_p99 = float(np.percentile(np.abs(gyr[finite]), 99)) if finite.any() else float("nan")
    if np.isfinite(gyr_p99) and gyr_p99 > 50.0:
        print(f"  [warn] {seq_id}: gyro p99={gyr_p99:.2f} (expected rad/s)")

    seq.update({"acc": acc, "gyr": gyr, "quat": quat, "dt": dt, "T": n, "valid_mask": valid_mask})
    return seq


# ─────────────────────────────────────────────────────────────────────────────
# Model loading  (ScaledModel wrapping SimpleRNN, same arch as train.py)
# ─────────────────────────────────────────────────────────────────────────────


def _build_model(
    mean: np.ndarray,
    std: np.ndarray,
    cfg: dict | None = None,
) -> torch.nn.Module:
    """Reconstruct the correct ScaledModel architecture from the checkpoint config.

    Supported architecture: ScaledModel(SimpleRNN).
    """
    from tsfast.models.scaling import ScaledModel, StandardScaler

    scaler = StandardScaler(
        torch.as_tensor(mean, dtype=torch.float32),
        torch.as_tensor(std, dtype=torch.float32),
    )
    from tsfast.models.rnn import SimpleRNN

    inner = SimpleRNN(
        input_size=7,
        output_size=4,
        num_layers=2,
        hidden_size=200,
        rnn_type="gru",
        return_state=True,  # needed for chunked stateful inference
    )

    return ScaledModel(inner, input_norm=scaler)


def _remap_state_dict(state: dict) -> dict:
    """Normalise checkpoint state dicts to match ScaledModel(SimpleRNN).

    Training with cuda_graph=True wraps the model as:
        ScaledModel(GraphedStatefulModel(SimpleRNN))
    which produces keys like  model.model.rnn.rnns.0.*
    We want                   model.rnn.rnns.0.*
    (ScaledModel(SimpleRNN) for eval — no CUDA-graph wrapper needed).
    """
    out: dict = {}
    for k, v in state.items():
        # Newer training wraps ScaledModel(UnitQuaternionOutput(SimpleRNN)),
        # producing model.model.rnn.* or model.model.model.rnn.* when combined
        # with CUDA-graph wrappers. Evaluation uses ScaledModel(SimpleRNN).
        while k.startswith("model.model."):
            k = "model." + k[len("model.model.") :]
        out[k] = v
    return out


def load_model(
    ckpt_path: Path,
    device: torch.device,
    use_ema: bool = False,
) -> torch.nn.Module | None:
    """Load checkpoint and return ScaledModel in eval mode, or None on failure."""
    if not ckpt_path.exists():
        print(f"  [model] checkpoint not found: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        print(f"  [model] unrecognised checkpoint format in {ckpt_path}")
        return None

    state_key = "ema" if (use_ema and "ema" in ckpt) else "model"
    source_weights = ckpt.get("source_weights", state_key)
    state = _remap_state_dict(ckpt[state_key])

    # Read config first — determines which architecture to instantiate.
    cfg = ckpt.get("config", {})

    # input_norm stats come from the checkpoint (either training-computed or DLS
    # constants set in train.py's build_learner).  Fall back to the known constants.
    mean = state.get("input_norm.mean", torch.as_tensor(INPUT_MEAN)).numpy()
    std = state.get("input_norm.std", torch.as_tensor(INPUT_STD)).numpy()

    model = _build_model(mean, std, cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [model] missing keys: {missing}")
    if unexpected:
        print(f"  [model] unexpected keys: {unexpected}")

    model.to(device).eval()

    epoch = ckpt.get("epoch", "?")
    best_value = ckpt.get("best_metric", float("nan"))
    metric_label = ckpt.get("selection_metric")
    if not metric_label:
        metric_label = "best_tilt"
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"  [model] epoch={epoch}  {metric_label}={best_value:.3f}°  "
        f"params={n_params / 1e6:.3f}M  weights={source_weights}  ({ckpt_path.name})"
    )
    print(f"          input_norm mean={np.round(mean, 4).tolist()}")
    print(f"          input_norm std ={np.round(std, 4).tolist()}")
    return model


def find_best_checkpoint(runs_dir: Path) -> Path | None:
    """Return the best.pt with the lowest best_metric across all runs."""
    candidates = sorted(runs_dir.rglob("best.pt"))
    if not candidates:
        return None
    best_path, best_val = None, float("inf")
    for p in candidates:
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=True)
            v = ckpt.get("best_metric", float("inf")) if isinstance(ckpt, dict) else float("inf")
            if v < best_val:
                best_val, best_path = v, p
        except Exception:
            pass
    return best_path


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────


def get_riann_session() -> rt.InferenceSession:
    """Return a process-local official RIANN ONNX session.

    Creating an ONNXRuntime session is expensive; doing it for every sequence
    dominates evaluation time on short files. The model is immutable, so a
    single reused CPU session is equivalent and much faster.
    """
    global _RIANN_SESSION
    if _RIANN_SESSION is None:
        with _RIANN_SESSION_LOCK:
            if _RIANN_SESSION is None:
                _RIANN_SESSION = rt.InferenceSession(
                    str(RIANN_ONNX), providers=["CPUExecutionProvider"]
                )
    return _RIANN_SESSION


def run_riann_onnx(acc: np.ndarray, gyr: np.ndarray, fs: float) -> tuple[np.ndarray, float]:
    """Official RIANN ONNX: returns (T,4) [w,x,y,z].

    The ONNX model has normalization baked in and expects raw physical units.
    Uses a constant dt = 1/fs (same as the official riann Python wrapper).
    """
    sess = get_riann_session()
    T = acc.shape[0]
    dt = np.full((T, 1), 1.0 / fs, dtype=np.float32)
    inp = np.concatenate([acc, gyr, dt], axis=-1)[None]  # (1, T, 7)
    h0 = np.zeros((2, 1, 1, 200), dtype=np.float32)  # (num_layers, 1, 1, hidden)
    t0 = time.perf_counter()
    out = sess.run([], {"input": inp, "h0": h0})[0][0]  # (T, 4) wxyz
    return out.astype(np.float32), time.perf_counter() - t0


def run_vqf(acc: np.ndarray, gyr: np.ndarray, fs: float) -> tuple[np.ndarray, float]:
    """Online VQF-6DoF: returns (T,4) [w,x,y,z].

    VQF's output uses the same +z-up ENU reference as RIANN — no axis
    conversion needed.  The first ~5 s (warm-up) is replaced by the first
    converged estimate to avoid transient errors in the RMSE.
    """
    from vqf import VQF

    t0 = time.perf_counter()
    filt = VQF(1.0 / fs)
    res = filt.updateBatch(gyr.astype(np.float64), acc.astype(np.float64), None)
    q = getattr(res, "quat6D", None)
    if q is None:
        q = res["quat6D"]
    q = np.asarray(q, dtype=np.float32)
    warmup = min(500, len(q) // 4)
    q[:warmup] = q[warmup : warmup + 1]
    return q, time.perf_counter() - t0


@torch.no_grad()
def run_model(
    model: torch.nn.Module,
    seq: dict,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Chunked stateful GRU inference: returns (T,4) [w,x,y,z] normalised.

    Processes in CHUNK_SIZE windows, carrying the GRU hidden state across
    chunk boundaries, so tilt errors don't accumulate from re-initialisation.

    Input to the model is raw physical units; ScaledModel.input_norm handles
    the (x - mean) / std normalisation internally.
    """
    acc, gyr, dt_arr = seq["acc"], seq["gyr"], seq["dt"]
    T = seq["T"]

    x_full = torch.cat(
        [
            torch.as_tensor(acc, device=device),
            torch.as_tensor(gyr, device=device),
            torch.as_tensor(dt_arr, device=device).unsqueeze(-1),
        ],
        dim=-1,
    ).unsqueeze(0)  # [1, T, 7]

    # Normalize once; run the inner SimpleRNN in chunks.
    # StandardScaler has no forward() — must call .normalize() directly.
    x_norm = model.input_norm.normalize(x_full)  # [1, T, 7]
    inner = model.model  # SimpleRNN(return_state=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    h = None
    outputs = []
    for start in range(0, T, CHUNK_SIZE):
        chunk = x_norm[:, start : start + CHUNK_SIZE]
        out, h = inner(chunk, h)
        outputs.append(out)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    q = torch.cat(outputs, dim=1)[0].cpu().numpy()  # [T, 4]
    return qnorm_np(q.astype(np.float32)), elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Quaternion math (all pure-numpy, no external quaternion library needed)
# ─────────────────────────────────────────────────────────────────────────────


def qnorm_np(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-9)
    return q / n


def qmul_np(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw,
        ],
        -1,
    )


def qconj_np(q: np.ndarray) -> np.ndarray:
    c = q.copy()
    c[..., 1:] *= -1
    return c


def riann_tilt_np(q_hat: np.ndarray, q_gt: np.ndarray) -> np.ndarray:
    """e_α = 2·arccos(√(w²+z²)) — RIANN formula, yaw-invariant.

    Extracts the tilt component of q_err = q_hat ⊗ q_gt^{-1}.
    For pure yaw error: w²+z²=1 → tilt=0°.  For pure tilt: z²=0 → tilt=max.
    """
    q_hat = qnorm_np(q_hat.astype(np.float64))
    q_gt = qnorm_np(q_gt.astype(np.float64))
    q_err = qnorm_np(qmul_np(q_hat, qconj_np(q_gt)))
    w, z = q_err[..., 0], q_err[..., 3]
    cos_h = np.sqrt(np.clip(w**2 + z**2, 0.0, 1.0))
    return np.degrees(2.0 * np.arccos(np.clip(cos_h, 0.0, 1.0))).astype(np.float32)


def extract_rp_np(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ZYX Euler roll/pitch [°] from (N,4) [w,x,y,z]."""
    q = qnorm_np(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    return roll, pitch


def ang_rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """RMSE with ±180° wrap-around."""
    d = ((a - b + 180) % 360) - 180
    if mask is not None:
        d = d[mask]
    return float(np.sqrt(np.mean(d**2))) if len(d) > 0 else float("nan")


def tilt_rmse(e: np.ndarray, mask: np.ndarray | None = None) -> float:
    x = e[mask] if mask is not None else e
    return float(np.sqrt(np.mean(x**2))) if len(x) > 0 else float("nan")


def align_heading(
    q_est: np.ndarray,
    q_gt: np.ndarray,
    valid: np.ndarray | None = None,
    n_align: int = N_ALIGN,
) -> np.ndarray:
    """Align q_est's world-frame heading to GT using the first n_align VALID frames.

    Many sequences have NaN GT at the start (optical tracker not yet tracking).
    Those frames are replaced with identity quaternions in validate_sequence, so
    computing the heading offset over the first N frames would align to identity
    instead of the real GT — producing completely wrong roll/pitch.

    Fix: collect the first n_align frames where valid=True, then compute the
    mean yaw offset q_off = mean(q_gt ⊗ q_est^{-1}) over those frames.
    """
    from scipy.spatial.transform import Rotation

    if valid is None:
        valid = np.ones(len(q_est), bool)

    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return qnorm_np(q_est.astype(np.float32))

    idx = valid_idx[:n_align]
    r_est = Rotation.from_quat(q_est[idx][:, [1, 2, 3, 0]])
    r_gt = Rotation.from_quat(q_gt[idx][:, [1, 2, 3, 0]])
    offsets = (r_gt * r_est.inv()).as_quat()
    for i in range(1, len(offsets)):
        if offsets[i] @ offsets[i - 1] < 0:
            offsets[i] *= -1
    q_mean = offsets.mean(0)
    q_mean /= np.linalg.norm(q_mean)
    r_off = Rotation.from_quat(q_mean)
    aligned = (r_off * Rotation.from_quat(q_est[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
    return qnorm_np(aligned.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SeqResult:
    seq_id: str
    dataset: str
    duration_s: float
    n_valid: int
    gate_frac: float
    trial_group: str

    model_tilt_rmse: float
    riann_tilt_rmse: float
    vqf_tilt_rmse: float

    model_roll_rmse: float
    riann_roll_rmse: float
    vqf_roll_rmse: float

    model_pitch_rmse: float
    riann_pitch_rmse: float
    vqf_pitch_rmse: float

    model_tilt_static: float
    riann_tilt_static: float

    time_model_ms: float
    time_riann_ms: float
    time_vqf_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# Per-sequence evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_sequence(
    path: Path,
    dataset: str,
    model: torch.nn.Module | None,
    device: torch.device,
    out_dir: Path,
    plot: bool = True,
    backend_workers: int = BACKEND_WORKERS,
) -> SeqResult | None:
    seq_id = path.stem
    try:
        seq = validate_sequence(load_h5(path), seq_id)
    except Exception as exc:
        print(f"  [skip] {seq_id}: {exc}")
        return None

    acc, gyr, q_gt = seq["acc"], seq["gyr"], seq["quat"]
    valid = seq["valid_mask"]
    T, dt_s = seq["T"], float(np.median(seq["dt"]))
    fs = seq["fs"]

    # Clip to the first valid frame.
    # Many sequences have NaN GT at the start (optical tracker not yet locked).
    # Feeding the GRU those pre-valid frames contaminates its hidden state —
    # in testing: starting from h=0 at valid_start → 1.0° tilt, starting from
    # frame 0 and evaluating from valid_start → 7.0°.  RIANN and VQF are also
    # slightly helped because their initial estimate is anchored to actual motion.
    first_valid = int(np.where(valid)[0][0]) if valid.any() else 0
    if first_valid > 0:
        acc = acc[first_valid:]
        gyr = gyr[first_valid:]
        q_gt = q_gt[first_valid:]
        valid = valid[first_valid:]
        seq = {
            **seq,
            "acc": acc,
            "gyr": gyr,
            "quat": q_gt,
            "dt": seq["dt"][first_valid:],
            "valid_mask": valid,
            "T": len(acc),
        }

    T = seq["T"]
    dur_s = T * dt_s
    t = np.arange(T) * dt_s

    # Static gate for gated metrics (helps isolate algorithmic error from dynamics)
    acc_norm = np.linalg.norm(acc, axis=1)
    is_static = np.abs(acc_norm - GRAVITY_MAG) < STATIC_THR
    gate_frac = float(is_static.mean()) * 100.0
    mask_val = valid if valid.any() else np.ones(T, bool)

    # ── Backends ─────────────────────────────────────────────────────────────
    # RIANN ONNX and VQF run on CPU; our model normally runs on CUDA. Running
    # them concurrently hides CPU baseline cost behind GPU inference without
    # changing metric semantics or sequence order.
    q_model = t_model = None
    if backend_workers > 1:
        max_workers = 2 if model is None else min(max(1, backend_workers), 3)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(run_riann_onnx, acc, gyr, fs): "riann",
                ex.submit(run_vqf, acc, gyr, fs): "vqf",
            }
            if model is not None:
                futures[ex.submit(run_model, model, seq, device)] = "model"

            outputs = {}
            for fut, name in futures.items():
                outputs[name] = fut.result()

        q_riann, t_riann = outputs["riann"]
        q_vqf, t_vqf = outputs["vqf"]
        if model is not None:
            q_model, t_model = outputs["model"]
    else:
        q_riann, t_riann = run_riann_onnx(acc, gyr, fs)
        q_vqf, t_vqf = run_vqf(acc, gyr, fs)
        if model is not None:
            q_model, t_model = run_model(model, seq, device)

    # ── PRIMARY: tilt RMSE (yaw-invariant, no alignment) ────────────────────
    e_riann = riann_tilt_np(q_riann, q_gt)
    e_vqf = riann_tilt_np(q_vqf, q_gt)
    e_model = (
        riann_tilt_np(q_model, q_gt)
        if q_model is not None
        else np.full(T, float("nan"), np.float32)
    )

    riann_tr = tilt_rmse(e_riann, mask_val)
    vqf_tr = tilt_rmse(e_vqf, mask_val)
    model_tr = tilt_rmse(e_model, mask_val) if q_model is not None else float("nan")

    static_gate = mask_val & is_static
    riann_ts = tilt_rmse(e_riann, static_gate) if static_gate.any() else float("nan")
    model_ts = (
        tilt_rmse(e_model, static_gate)
        if (q_model is not None and static_gate.any())
        else float("nan")
    )

    # ── SECONDARY: roll/pitch RMSE (after heading alignment) ─────────────────
    q_riann_a = align_heading(q_riann, q_gt, valid=mask_val)
    q_vqf_a = align_heading(q_vqf, q_gt, valid=mask_val)
    q_model_a = align_heading(q_model, q_gt, valid=mask_val) if q_model is not None else None

    rg, pg = extract_rp_np(q_gt)
    rr, pr = extract_rp_np(q_riann_a)
    rv, pv = extract_rp_np(q_vqf_a)
    rm, pm = (
        extract_rp_np(q_model_a)
        if q_model_a is not None
        else (np.full(T, float("nan")), np.full(T, float("nan")))
    )

    riann_rr = ang_rmse(rr, rg, mask_val)
    vqf_rr = ang_rmse(rv, rg, mask_val)
    model_rr = ang_rmse(rm, rg, mask_val) if q_model is not None else float("nan")

    riann_pr = ang_rmse(pr, pg, mask_val)
    vqf_pr = ang_rmse(pv, pg, mask_val)
    model_pr = ang_rmse(pm, pg, mask_val) if q_model is not None else float("nan")

    def f(v):
        return "  --  " if math.isnan(v) else f"{v:6.2f}"

    print(f"  {dataset:16s}  {seq_id[:35]:<35}  {fs:5.0f}Hz  {dur_s:5.0f}s  gate={gate_frac:.0f}%")
    print(
        f"    Tilt°  ours={f(model_tr)}  riann={f(riann_tr)}  vqf={f(vqf_tr)}"
        f"  (static: ours={f(model_ts)} riann={f(riann_ts)})"
    )
    print(f"    Roll°  ours={f(model_rr)}  riann={f(riann_rr)}  vqf={f(vqf_rr)}")
    print(f"    Pitch° ours={f(model_pr)}  riann={f(riann_pr)}  vqf={f(vqf_pr)}")

    if plot:
        _plot_sequence(
            t=t,
            seq_id=seq_id,
            dataset=dataset,
            dur_s=dur_s,
            rg=rg,
            pg=pg,
            rm=rm,
            pm=pm,
            rr=rr,
            pr=pr,
            rv=rv,
            pv=pv,
            e_model=e_model,
            e_riann=e_riann,
            e_vqf=e_vqf,
            rmse_tilt={"model": model_tr, "riann": riann_tr, "vqf": vqf_tr},
            rmse_roll={"model": model_rr, "riann": riann_rr, "vqf": vqf_rr},
            rmse_pitch={"model": model_pr, "riann": riann_pr, "vqf": vqf_pr},
            time_ms={"model": (t_model or 0) * 1e3, "riann": t_riann * 1e3, "vqf": t_vqf * 1e3},
            out_path=out_dir / f"{dataset}_{seq_id}.png",
        )

    return SeqResult(
        seq_id=seq_id,
        dataset=dataset,
        duration_s=dur_s,
        n_valid=int(mask_val.sum()),
        gate_frac=gate_frac,
        trial_group=infer_trial_group(dataset, seq_id),
        model_tilt_rmse=model_tr,
        riann_tilt_rmse=riann_tr,
        vqf_tilt_rmse=vqf_tr,
        model_roll_rmse=model_rr,
        riann_roll_rmse=riann_rr,
        vqf_roll_rmse=vqf_rr,
        model_pitch_rmse=model_pr,
        riann_pitch_rmse=riann_pr,
        vqf_pitch_rmse=vqf_pr,
        model_tilt_static=model_ts,
        riann_tilt_static=riann_ts,
        time_model_ms=(t_model or 0) * 1e3,
        time_riann_ms=t_riann * 1e3,
        time_vqf_ms=t_vqf * 1e3,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────


def _plot_sequence(
    t,
    seq_id,
    dataset,
    dur_s,
    rg,
    pg,
    rm,
    pm,
    rr,
    pr,
    rv,
    pv,
    e_model,
    e_riann,
    e_vqf,
    rmse_tilt,
    rmse_roll,
    rmse_pitch,
    time_ms,
    out_path: Path,
) -> None:
    configure_matplotlib()

    def academic_axes(ax, grid_axis="y"):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3.5, width=0.7)
        ax.grid(axis=grid_axis, color="#D7D7D7", linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)

    def panel(ax, label):
        ax.text(
            0.008, 0.94, label, transform=ax.transAxes, va="top", fontweight="bold", fontsize=10.0
        )

    fig = plt.figure(figsize=(12.0, 8.0), layout="constrained")
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 0.82, 1.0], width_ratios=[1.25, 1.0])
    ax_roll = fig.add_subplot(gs[0, :])
    ax_pitch = fig.add_subplot(gs[1, :], sharex=ax_roll)
    ax_local = fig.add_subplot(gs[2, :], sharex=ax_roll)
    ax_ecdf = fig.add_subplot(gs[3, 0])
    ax_metric = fig.add_subplot(gs[3, 1])
    for ax in (ax_roll, ax_pitch, ax_local, ax_ecdf, ax_metric):
        academic_axes(ax)

    traces = {
        "model": (rm, pm, e_model, "TG-DCM", "-"),
        "riann": (rr, pr, e_riann, "RIANN", (0, (4, 2))),
        "vqf": (rv, pv, e_vqf, "VQF", (0, (1.5, 2))),
    }
    ax_roll.plot(t, rg, color=COLORS["gt"], linewidth=1.25, label="Ground truth", zorder=4)
    ax_pitch.plot(t, pg, color=COLORS["gt"], linewidth=1.25, zorder=4)
    for method, (roll, pitch, _, label, linestyle) in traces.items():
        if not np.all(np.isnan(roll)):
            ax_roll.plot(
                t,
                roll,
                color=COLORS[method],
                linewidth=0.9,
                alpha=0.9,
                linestyle=linestyle,
                label=label,
            )
            ax_pitch.plot(
                t, pitch, color=COLORS[method], linewidth=0.9, alpha=0.9, linestyle=linestyle
            )
    ax_roll.set_ylabel("Roll angle (deg)")
    ax_pitch.set_ylabel("Pitch angle (deg)")
    ax_roll.legend(frameon=False, ncol=4, loc="upper right", handlelength=2.5)
    panel(ax_roll, "(a)")
    panel(ax_pitch, "(b)")

    dt_step = (t[1] - t[0]) if len(t) > 1 else 1.0
    win = max(1, int(5.0 / dt_step))

    def running_rmse(e):
        e2 = e**2
        cs = np.cumsum(e2)
        cs = np.concatenate([[0.0], cs])
        n = np.minimum(np.arange(1, len(e) + 1), win)
        start_idx = np.maximum(0, np.arange(len(e)) - win + 1)
        return np.sqrt((cs[np.arange(len(e)) + 1] - cs[start_idx]) / n)

    for method, (_, _, error, label, linestyle) in traces.items():
        if not np.all(np.isnan(error)):
            ax_local.plot(
                t,
                running_rmse(error),
                color=COLORS[method],
                linewidth=1.35,
                linestyle=linestyle,
                label=label,
            )
    ax_local.axhline(2.0, color="#8A1C1C", linewidth=0.8, linestyle=(0, (4, 3)), label="2 deg")
    ax_local.set_ylabel("5 s tilt RMSE (deg)")
    ax_local.set_xlabel("Time (s)")
    ax_local.set_ylim(bottom=0)
    ax_local.legend(frameon=False, ncol=4, loc="upper right")
    panel(ax_local, "(c)")

    for method, (_, _, error, label, _) in traces.items():
        values = np.sort(error[np.isfinite(error)])
        if values.size:
            probability = np.arange(1, len(values) + 1) / len(values)
            ax_ecdf.step(
                values, probability, where="post", color=COLORS[method], linewidth=1.5, label=label
            )
    ax_ecdf.axhline(0.95, color="#777777", linewidth=0.7, linestyle=(0, (3, 3)))
    ax_ecdf.axvline(2.0, color="#8A1C1C", linewidth=0.8, linestyle=(0, (4, 3)))
    ax_ecdf.set_xlabel("Instantaneous tilt error (deg)")
    ax_ecdf.set_ylabel("Empirical cumulative probability")
    ax_ecdf.set_ylim(0, 1)
    ax_ecdf.set_title("Tilt-error distribution", loc="left", fontweight="bold")
    ax_ecdf.legend(frameon=False, loc="lower right")
    panel(ax_ecdf, "(d)")

    metrics = [("Tilt", rmse_tilt, "o"), ("Roll", rmse_roll, "s"), ("Pitch", rmse_pitch, "D")]
    methods = [("model", "TG-DCM"), ("riann", "RIANN"), ("vqf", "VQF")]
    ypos = np.arange(3)[::-1]
    offsets = {"Tilt": 0.18, "Roll": 0.0, "Pitch": -0.18}
    for metric, values, marker in metrics:
        for y, (method, _) in zip(ypos, methods):
            value = values[method]
            if not math.isnan(value):
                ax_metric.scatter(
                    value,
                    y + offsets[metric],
                    marker=marker,
                    s=34,
                    color=COLORS[method],
                    edgecolor="white",
                    linewidth=0.6,
                )
        ax_metric.scatter([], [], marker=marker, color="#555555", label=metric)
    ax_metric.set_yticks(ypos, [label for _, label in methods])
    ax_metric.set_xlabel("Sequence RMSE (deg)")
    ax_metric.set_title("Sequence metrics", loc="left", fontweight="bold")
    ax_metric.legend(frameon=False, ncol=3, loc="lower right")
    academic_axes(ax_metric, grid_axis="x")
    panel(ax_metric, "(e)")

    plt.setp(ax_roll.get_xticklabels(), visible=False)
    plt.setp(ax_pitch.get_xticklabels(), visible=False)
    fig.suptitle(f"{dataset} | {seq_id} | {dur_s:.1f} s", fontsize=13.0, fontweight="bold")
    fig.text(
        0.5,
        -0.012,
        "Roll and pitch are heading-aligned. Tilt is yaw-invariant. "
        f"Inference time: TG-DCM {time_ms['model']:.1f} ms, "
        f"RIANN {time_ms['riann']:.1f} ms, VQF {time_ms['vqf']:.1f} ms.",
        ha="center",
        va="top",
        fontsize=7.8,
        color="#666666",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"    → {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────


def print_summary(results: list[SeqResult]) -> None:
    if not results:
        return

    def f(v):
        return "   --  " if (v is None or math.isnan(v)) else f"{v:6.2f}°"

    def pct(ours, baseline):
        if math.isnan(ours) or math.isnan(baseline) or baseline == 0:
            return ""
        return f"  ({(ours - baseline) / baseline * 100:+.1f}%)"

    hdr = (
        f"  {'Dataset':<20}  {'Tilt ours':>10}  {'Tilt RIANN':>11}  "
        f"{'Tilt VQF':>9}  {'Roll ours':>10}  {'Pitch ours':>11}  {'N':>4}"
    )
    sep = "  " + "─" * (len(hdr) - 2)

    print("\n" + "═" * len(hdr))
    print("  EVALUATION SUMMARY")
    print("═" * len(hdr))
    print(hdr)
    print(sep)

    all_mt, all_rt, all_vt = [], [], []
    all_mr, all_mp = [], []

    for ds in sorted({r.dataset for r in results}):
        ds_res = [r for r in results if r.dataset == ds]
        mt = [r.model_tilt_rmse for r in ds_res if not math.isnan(r.model_tilt_rmse)]
        rt = [r.riann_tilt_rmse for r in ds_res if not math.isnan(r.riann_tilt_rmse)]
        vt = [r.vqf_tilt_rmse for r in ds_res if not math.isnan(r.vqf_tilt_rmse)]
        mr = [r.model_roll_rmse for r in ds_res if not math.isnan(r.model_roll_rmse)]
        mp = [r.model_pitch_rmse for r in ds_res if not math.isnan(r.model_pitch_rmse)]
        mt_m = np.mean(mt) if mt else float("nan")
        rt_m = np.mean(rt) if rt else float("nan")
        vt_m = np.mean(vt) if vt else float("nan")
        mr_m = np.mean(mr) if mr else float("nan")
        mp_m = np.mean(mp) if mp else float("nan")
        all_mt.extend(mt)
        all_rt.extend(rt)
        all_vt.extend(vt)
        all_mr.extend(mr)
        all_mp.extend(mp)
        print(
            f"  {ds:<20}  {f(mt_m):>10}{pct(mt_m, rt_m):<12}  {f(rt_m):>11}  "
            f"{f(vt_m):>9}  {f(mr_m):>10}  {f(mp_m):>11}  {len(ds_res):>4}"
        )

    print(sep)
    ov_mt = np.mean(all_mt) if all_mt else float("nan")
    ov_rt = np.mean(all_rt) if all_rt else float("nan")
    ov_vt = np.mean(all_vt) if all_vt else float("nan")
    ov_mr = np.mean(all_mr) if all_mr else float("nan")
    ov_mp = np.mean(all_mp) if all_mp else float("nan")
    print(
        f"  {'OVERALL':<20}  {f(ov_mt):>10}{pct(ov_mt, ov_rt):<12}  {f(ov_rt):>11}  "
        f"{f(ov_vt):>9}  {f(ov_mr):>10}  {f(ov_mp):>11}  {len(results):>4}"
    )
    print()
    if not math.isnan(ov_mt) and not math.isnan(ov_rt):
        diff = ov_mt - ov_rt
        verdict = "BETTER ✓" if diff < 0 else "WORSE ✗"
        print(f"  Ours vs RIANN: {ov_mt:.3f}° vs {ov_rt:.3f}°  Δ={diff:+.3f}°  → {verdict}")
    print()
    print("  tilt = 2·arccos(√(w²+z²))  yaw-invariant — matches the training loss exactly.")
    print("  roll/pitch computed after heading-alignment over first 100 frames.")
    print("═" * len(hdr))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate TG-DCM against RIANN and VQF-6D.")
    p.add_argument(
        "--ckpt",
        default=DEFAULT_CKPT,
        help="Checkpoint path (.pt). Default: checkpoints/tg_dcm.pt.",
    )
    p.add_argument(
        "--split",
        choices=["test", "valid", "all"],
        default="test",
        help="Which data split to evaluate (default: test).",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Filter to specific dataset names (e.g. OxIOD Myon). Default: all in split.",
    )
    p.add_argument("--data-dir", default=None, help="Data directory (default: ./data).")
    p.add_argument(
        "--diodem-root",
        default=ROOT / "diodem",
        help="DIODEM root directory. Accepts either diodem/ or diodem/diodem/.",
    )
    p.add_argument(
        "--include-diodem-questionable",
        action="store_true",
        help="Include DIODEM files under exp*/questionable/. Default: exclude them.",
    )
    p.add_argument(
        "--max-seqs", type=int, default=None, help="Max sequences per dataset for quick smoke-test."
    )
    p.add_argument(
        "--out-dir", default="evaluation", help="Output directory for tables and figures."
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Write tables only; suppress aggregate and sequence figures.",
    )
    p.add_argument(
        "--sequence-plots",
        choices=["none", "all"],
        default="none",
        help="Write one diagnostic PNG per sequence. Aggregate figures always use every sequence.",
    )
    p.add_argument(
        "--no-model", action="store_true", help="Skip our model, run RIANN+VQF baselines only."
    )
    p.add_argument("--raw", action="store_true", help="Use raw model weights instead of EMA.")
    p.add_argument(
        "--backend-workers",
        type=int,
        default=BACKEND_WORKERS,
        help="Parallel backend workers per sequence: RIANN ONNX, VQF, and model. Use 1 for serial.",
    )
    p.add_argument(
        "--sequence-workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 1) // 4)),
        help="Sequences evaluated concurrently. Default: min(8, CPU count / 4).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_arg = Path(args.out_dir)
    out_dir = out_arg if out_arg.is_absolute() else ROOT / out_arg
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  RIANN Evaluation")
    print("=" * 72)
    print(f"  Device  : {device}")
    print(f"  Split   : {args.split}")
    print(f"  Out dir : {out_dir}")
    sequence_workers = max(1, args.sequence_workers)
    sequence_plots = args.sequence_plots == "all" and not args.no_plot
    if sequence_plots and sequence_workers > 1:
        print("  Sequence plotting uses one worker because Matplotlib rendering is process-global.")
        sequence_workers = 1
    backend_workers = max(1, args.backend_workers) if sequence_workers == 1 else 1
    print(f"  Workers : sequences={sequence_workers}, backends={backend_workers}")
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    model = None
    if not args.no_model:
        ckpt_path = Path(args.ckpt) if args.ckpt else find_best_checkpoint(ROOT / "runs")
        if ckpt_path is None:
            print("  No checkpoint found in runs/. Running baselines only.")
        else:
            print(f"  Checkpoint: {ckpt_path}")
            model = load_model(ckpt_path, device, use_ema=not args.raw)

    # ── Build sequence list from splits ──────────────────────────────────────
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    files = get_files(data_dir)

    if args.split == "all":
        all_paths = [(p, "valid") for p in files["valid"]] + [(p, "test") for p in files["test"]]
    elif args.split == "valid":
        all_paths = [(p, "valid") for p in files["valid"]]
    else:  # test
        all_paths = [(p, "test") for p in files["test"]]

    # Group by dataset name (parent directory)
    from collections import defaultdict

    by_dataset: dict[str, list[Path]] = defaultdict(list)
    for p, _ in all_paths:
        by_dataset[p.parent.name].append(p)

    requested_datasets = {d.lower() for d in args.datasets} if args.datasets else set()
    use_diodem = "diodem" in requested_datasets

    if args.datasets:
        keep = requested_datasets - {"diodem"}
        by_dataset = {ds: v for ds, v in by_dataset.items() if ds.lower() in keep}

    if use_diodem:
        diodem_paths = discover_diodem_files(
            Path(args.diodem_root),
            include_questionable=args.include_diodem_questionable,
        )
        if diodem_paths:
            by_dataset.setdefault("diodem", []).extend(diodem_paths)

    if not by_dataset:
        print("ERROR: No sequences found. Check --data-dir or --datasets.")
        sys.exit(1)

    total = sum(len(v) for v in by_dataset.values())
    print(f"  Found {total} sequences")
    for ds, paths in sorted(by_dataset.items()):
        n = len(paths) if args.max_seqs is None else min(len(paths), args.max_seqs)
        print(f"    {ds}: {n} seq(s)")
    print()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"  {'Dataset':<16}  {'Sequence':<35}  {'Hz':>5}  {'Dur':>5}  Gate")
    print("  " + "─" * 72)

    tasks = [
        (path, dataset)
        for dataset, paths in sorted(by_dataset.items())
        for path in sorted(paths if args.max_seqs is None else paths[: args.max_seqs])
    ]

    def evaluate_task(task: tuple[Path, str]) -> SeqResult | None:
        path, dataset = task
        return evaluate_sequence(
            path,
            dataset,
            model,
            device,
            out_dir / "sequences",
            plot=sequence_plots,
            backend_workers=backend_workers,
        )

    results: list[SeqResult] = []
    t_eval0 = time.perf_counter()
    if sequence_workers == 1:
        evaluated = map(evaluate_task, tasks)
        results.extend(result for result in evaluated if result is not None)
    else:
        with ThreadPoolExecutor(max_workers=sequence_workers) as executor:
            evaluated = executor.map(evaluate_task, tasks)
            results.extend(result for result in evaluated if result is not None)
    elapsed = time.perf_counter() - t_eval0

    print_summary(results)
    frame = results_frame(results)
    artifacts = write_evaluation_artifacts(frame, out_dir)
    figures = [] if args.no_plot else generate_evaluation_figures(frame, out_dir)
    print(f"  Evaluation wall time: {elapsed:.1f}s for {len(results)} sequence(s)")
    print(f"  Per-sequence table: {artifacts['per_sequence']}")
    if figures:
        print(f"  Aggregate figures: {out_dir / 'repository'} and {out_dir / 'poster'}")


if __name__ == "__main__":
    main()
