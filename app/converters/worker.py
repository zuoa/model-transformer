"""Subprocess entry point: run one conversion job and stream progress.

Invoked by the server as ``python -m app.converters.worker <config.json>``.
Runs in an isolated process so a rknn-toolkit2 segfault cannot take down the
FastAPI server, and so stdout can be streamed line-by-line to SSE clients.

Protocol: every meaningful event is one JSON line on stdout (see progress.py).
Library logging (ultralytics / rknn verbose) also lands on stdout/stderr and is
forwarded by the server as raw log lines.

Exit code 0 on success, non-zero on failure. A terminal ``error`` event is
emitted before exiting non-zero so clients get the reason over SSE.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

# Allow ``python -m app.converters.worker`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.converters import calibration, onnx, progress, rknn, ultralytics_rknn  # noqa: E402


def _run_mock(cfg: dict, work_dir: Path) -> Path:
    """Fake conversion for plumbing tests/CI (MT_MOCK=1). No torch/rknn needed."""
    pipeline = cfg["pipeline"]
    ext = "onnx" if pipeline == "pt_to_onnx" else "rknn"
    stem = Path(cfg.get("model_name", "model")).stem
    out = work_dir / f"{stem}.{ext}"
    for pct in (10, 30, 55, 80):
        progress.progress(pct, f"mock step {pct}%")
        time.sleep(0.3)
    out.write_bytes(f"mock {pipeline} output\n".encode())
    progress.log("(mock mode — no real conversion ran)")
    return out


def _prepare_calibration_if_needed(cfg: dict, work_dir: Path) -> None:
    if not cfg.get("is_int8"):
        return
    zip_path = cfg.get("calib_zip")
    if not zip_path or not Path(zip_path).exists():
        raise ValueError("INT8 quantization requires a calibration zip, but none was provided.")
    calib_dir, dataset_txt = calibration.prepare(Path(zip_path), work_dir)
    cfg["calib_dir"] = str(calib_dir)
    cfg["dataset_txt"] = str(dataset_txt)


def _disable_ultralytics_telemetry() -> None:
    """Pre-write a settings.yaml disabling hub sync/analytics into the per-job
    YOLO_CONFIG_DIR, so airgapped containers don't attempt an analytics call."""
    cfg_dir = os.environ.get("YOLO_CONFIG_DIR")
    if not cfg_dir:
        return
    p = Path(cfg_dir) / "settings.yaml"
    if p.exists():
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("settings_version: 0.0.9\nsync: false\n", encoding="utf-8")
    except Exception:
        pass  # non-fatal


def dispatch(cfg: dict) -> Path:
    work_dir = Path(cfg["work_dir"])
    pipeline = cfg["pipeline"]

    if os.environ.get("MT_MOCK") == "1":
        return _run_mock(cfg, work_dir)

    _prepare_calibration_if_needed(cfg, work_dir)

    if pipeline == "pt_to_onnx":
        return onnx.convert(cfg, work_dir)
    if pipeline == "pt_to_rknn":
        return ultralytics_rknn.convert(cfg, work_dir)
    if pipeline == "onnx_to_rknn":
        return rknn.convert(cfg, work_dir)
    raise ValueError(f"Unknown pipeline: {pipeline}")


def main(argv: list[str]) -> int:
    config_path = Path(argv[1])
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    work_dir = Path(cfg["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)

    progress.start(cfg["pipeline"])
    _disable_ultralytics_telemetry()
    try:
        result = dispatch(cfg)
    except Exception as exc:  # noqa: BLE001 - report any failure to the client
        progress.fail(repr(exc), traceback.format_exc())
        return 1

    # Ensure the artifact lives in the work dir and report it.
    result = Path(result)
    progress.done(result.name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
