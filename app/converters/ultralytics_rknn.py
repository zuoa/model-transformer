"""PT -> RKNN via ultralytics' native ``format='rknn'`` export.

This is the recommended PT->RKNN path: ultralytics configures the YOLO Detect
head for deployment (its ``Detect.export=True`` mechanism) and runs
rknn-toolkit2 internally, which avoids the INT8 quantization collapse of the
DFL/``dist2bbox`` decode nodes that plagues stock-ONNX->RKNN conversion.

It exposes ultralytics-level knobs (target platform, quantize mode, imgsz,
opset, half) rather than the full rknn-toolkit2 parameter set. For full toolkit
control (mean/std/quantized_method/algorithm) use the ONNX->RKNN pipeline with a
head-stripped ONNX instead.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from app.converters import progress


def convert(cfg: dict, work_dir: Path) -> Path:
    from ultralytics import YOLO

    model_path = Path(cfg["model_path"])
    stem = Path(cfg.get("model_name", model_path.name)).stem
    out_name = f"{stem}.rknn"

    progress.log(f"Loading {model_path.name} with ultralytics ...")
    model = YOLO(str(model_path))

    kwargs = dict(
        format="rknn",
        name=cfg["target_platform"],
        imgsz=cfg["imgsz"],
        opset=cfg["opset"],
        half=cfg["half"],
    )
    q = cfg.get("quantize")
    if q:
        kwargs["quantize"] = q
        progress.log(f"Quantization requested (mode={q}); needs calibration")
    # ultralytics reads calibration images from `data`. We point it at the
    # directory prepared from the uploaded zip.
    if cfg.get("calib_dir"):
        kwargs["data"] = str(cfg["calib_dir"])

    progress.progress(30, f"Running ultralytics rknn export (target={cfg['target_platform']}) ...")
    produced = model.export(**kwargs)
    produced = Path(produced)

    out_path = work_dir / out_name
    if produced.resolve() != out_path.resolve():
        shutil.move(str(produced), str(out_path))

    progress.progress(100, f"RKNN written: {out_name}")
    return out_path
