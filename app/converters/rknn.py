"""ONNX -> RKNN via rknn-toolkit2 directly (full parameter control).

WARNING surfaced to the user in the UI: if this ONNX is a stock YOLOv8/v11
export that still contains the Detect head's DFL / dist2bbox decode nodes
(Sub/Add/Mul), INT8 quantization will collapse those values and destroy
detection accuracy. Provide a head-stripped / end2end ONNX, or use the
PT->RKNN pipeline (which handles the head correctly).
"""
from __future__ import annotations

import shutil
from pathlib import Path

from app.converters import progress


def convert(cfg: dict, work_dir: Path) -> Path:
    from rknn.api import RKNN

    model_path = Path(cfg["model_path"])
    stem = Path(cfg.get("model_name", model_path.name)).stem
    out_name = f"{stem}.rknn"

    rknn = RKNN(verbose=True)

    progress.log("Configuring RKNN ...")
    rknn.config(
        mean_values=[list(cfg["mean_values"])],
        std_values=[list(cfg["std_values"])],
        target_platform=cfg["target_platform"],
        quantized_dtype=cfg["quantized_dtype"],
        quantized_method=cfg["quantized_method"],
        quantized_algorithm=cfg["quantized_algorithm"],
    )

    progress.progress(20, f"Loading ONNX {model_path.name} ...")
    ret = rknn.load_onnx(model=str(model_path))
    if ret != 0:
        raise RuntimeError(f"rknn.load_onnx failed (code={ret})")

    do_q = bool(cfg["do_quantization"]) and bool(cfg.get("dataset_txt"))
    if do_q:
        progress.progress(
            40,
            f"Building with INT8 quantization ({cfg['quantized_dtype']}/"
            f"{cfg['quantized_algorithm']}) — this can take minutes ...",
        )
        ret = rknn.build(do_quantization=True, dataset=str(cfg["dataset_txt"]))
    else:
        progress.progress(40, "Building (no quantization, FP) ...")
        ret = rknn.build(do_quantization=False)
    if ret != 0:
        raise RuntimeError(f"rknn.build failed (code={ret})")

    out_path = work_dir / out_name
    progress.progress(90, f"Exporting {out_name} ...")
    ret = rknn.export_rknn(str(out_path))
    if ret != 0:
        raise RuntimeError(f"rknn.export_rknn failed (code={ret})")

    rknn.release()
    progress.progress(100, f"RKNN written: {out_name}")
    return out_path
