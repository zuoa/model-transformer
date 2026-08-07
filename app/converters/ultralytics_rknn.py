"""PT -> RKNN via ultralytics' native ``format='rknn'`` export.

This is the recommended PT->RKNN path: ultralytics configures the YOLO Detect
head for deployment and runs rknn-toolkit2 internally, avoiding the INT8
quantization collapse of the DFL/``dist2bbox`` decode nodes that plagues
stock-ONNX->RKNN conversion.

Two non-obvious behaviors of ``export(format='rknn')`` that this module handles:
  * It returns the **output directory** containing ``<stem>-<target>.rknn`` (plus
    ``metadata.yaml``) — NOT a file. We must extract the ``.rknn`` out of it.
  * INT8 calibration is driven by ``data=<YOLO dataset YAML>`` (ultralytics builds
    its own image list from it), not a directory or a txt of paths.

It exposes ultralytics-level knobs (target platform, quantize mode, imgsz, opset)
rather than the full rknn-toolkit2 parameter set. For full toolkit control use
the ONNX->RKNN pipeline with a head-stripped ONNX instead.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from app.converters import progress


def place_rknn_output(produced: Path, out_path: Path, stem: str, work_dir: Path) -> Path:
    """Resolve ultralytics' rknn export result to a single ``.rknn`` file at
    ``out_path``. ``produced`` is what ``model.export()`` returned — typically the
    output *directory*. Extracts the ``.rknn`` file from it, then cleans up the
    directory and the intermediate ONNX."""
    if produced.is_dir():
        candidates = sorted(produced.glob("*.rknn"))
        if not candidates:
            raise RuntimeError(f"ultralytics rknn export produced no .rknn file in {produced}")
        rknn_file = candidates[0]
    elif produced.is_file():
        rknn_file = produced
    else:
        raise RuntimeError(f"ultralytics rknn export returned a non-existent path: {produced}")

    if out_path.exists() or out_path.is_dir():
        shutil.rmtree(out_path, ignore_errors=True) if out_path.is_dir() else out_path.unlink(missing_ok=True)
    shutil.move(str(rknn_file), str(out_path))

    # Tidy up: the export directory (metadata.yaml, etc.) and intermediate ONNX.
    if produced.is_dir() and produced.exists() and produced.resolve() != work_dir.resolve():
        shutil.rmtree(produced, ignore_errors=True)
    intermediate = work_dir / f"{stem}.onnx"
    if intermediate.exists():
        intermediate.unlink()

    if not out_path.is_file():
        raise RuntimeError(f"rknn output is not a file at {out_path}")
    return out_path


def convert(cfg: dict, work_dir: Path) -> Path:
    from ultralytics import YOLO

    model_path = Path(cfg["model_path"])
    stem = Path(cfg.get("model_name", model_path.name)).stem
    out_path = work_dir / f"{stem}.rknn"

    progress.log(f"Loading {model_path.name} with ultralytics ...")
    model = YOLO(str(model_path))

    kwargs: dict = {
        "format": "rknn",
        "name": cfg["target_platform"],
        "imgsz": cfg["imgsz"],
        "opset": cfg["opset"],
    }
    q = cfg.get("quantize")
    if q:
        kwargs["quantize"] = q
        if q == 8:
            if not cfg.get("calib_yaml"):
                raise ValueError("INT8 quantization requires a calibration set (calib_yaml missing).")
            kwargs["data"] = str(cfg["calib_yaml"])
            progress.log("INT8 quantization: using calibration dataset via ultralytics data yaml")

    progress.progress(30, f"Running ultralytics rknn export (target={cfg['target_platform']}) ...")
    produced = Path(model.export(**kwargs))

    out_path = place_rknn_output(produced, out_path, stem, work_dir)
    progress.progress(100, f"RKNN written: {out_path.name}")
    return out_path
