"""PT -> ONNX via ultralytics."""
from __future__ import annotations

import shutil
from pathlib import Path

from app.converters import progress


def convert(cfg: dict, work_dir: Path) -> Path:
    from ultralytics import YOLO

    model_path = Path(cfg["model_path"])
    stem = Path(cfg.get("model_name", model_path.name)).stem
    out_name = f"{stem}.onnx"

    progress.log(f"Loading {model_path.name} with ultralytics ...")
    model = YOLO(str(model_path))

    progress.progress(20, "Exporting to ONNX")
    produced = model.export(
        format="onnx",
        opset=cfg["opset"],
        simplify=cfg["simplify"],
        dynamic=cfg["dynamic"],
        half=cfg["half"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
    )
    produced = Path(produced)

    # Normalize to a stable download name inside the work dir.
    out_path = work_dir / out_name
    if produced.resolve() != out_path.resolve():
        shutil.move(str(produced), str(out_path))

    progress.progress(100, f"ONNX written: {out_name}")
    return out_path
