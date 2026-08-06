"""Central configuration constants.

Kept dependency-free so it can be imported by both the server and the
subprocess worker without pulling in the heavy conversion stack.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repo root (project lives at <root>/app/...). worker.py imports this too.
ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT_DIR / "static"

# Where per-job working directories live. Override via env (e.g. a mounted volume).
WORK_BASE_DIR = Path(os.environ.get("MT_WORK_DIR", ROOT_DIR / "work"))

# Upload size caps (bytes). Large YOLO .pt checkpoints can be hundreds of MB.
MAX_MODEL_BYTES = int(os.environ.get("MT_MAX_MODEL_MB", "2048")) * 1024 * 1024
MAX_CALIB_BYTES = int(os.environ.get("MT_MAX_CALIB_MB", "512")) * 1024 * 1024

# How long a finished/failed job's artifacts are kept before the TTL sweeper
# deletes the working directory. Clients that never download would otherwise
# leak gigabytes.
JOB_TTL_SECONDS = int(os.environ.get("MT_JOB_TTL_SECONDS", str(60 * 60)))  # 1h

# RKNN builds load torch + onnxruntime + model + calibration set and routinely
# consume multiple GB. Cap concurrent RKNN jobs to 1 (the rest queue). ONNX-only
# exports are cheap and are not throttled by this semaphore.
RKNN_CONCURRENCY = int(os.environ.get("MT_RKNN_CONCURRENCY", "1"))

# Per-SSE-subscriber queue: keep only the most recent N log lines so a stalled
# client cannot grow memory without bound. Older lines are dropped (truncation
# marker sent once).
SSE_QUEUE_MAXLINES = int(os.environ.get("MT_SSE_QUEUE_MAXLINES", "500"))

# rknn-toolkit2 v2.3.2 supported target platforms. Validating here means an
# invalid platform is rejected at job creation instead of failing an opaque,
# multi-minute build later.
SUPPORTED_TARGETS = (
    "rk3562",
    "rk3566",
    "rk3568",
    "rk3576",
    "rk3588",
    "rk3588s",
    "rv1103",
    "rv1103b",
    "rv1106",
    "rv1106b",
    "rk2118",
)

# Whether rknn-toolkit2 is importable in this process (False on macOS dev box,
# True inside the Linux container). Used to mark RKNN paths as available.
def rknn_available() -> bool:
    try:
        import importlib

        return importlib.util.find_spec("rknn") is not None  # type: ignore[attr-defined]
    except Exception:
        return False


# Can this host run a given pipeline right now? (ONNX needs ultralytics; RKNN
# additionally needs rknn-toolkit2, i.e. the Linux container.)
def can_run(pipeline: str) -> bool:
    try:
        import importlib

        has_ultra = importlib.util.find_spec("ultralytics") is not None  # type: ignore[attr-defined]
    except Exception:
        has_ultra = False
    if pipeline == "pt_to_onnx":
        return has_ultra
    if pipeline in ("pt_to_rknn", "onnx_to_rknn"):
        return has_ultra and rknn_available()
    return False
