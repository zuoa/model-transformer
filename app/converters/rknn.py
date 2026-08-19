"""ONNX -> RKNN via rknn-toolkit2 directly (full parameter control).

WARNING surfaced to the user in the UI: if this ONNX is a stock YOLOv8/v11
export that still contains the Detect head's DFL / dist2bbox decode nodes
(Sub/Add/Mul), INT8 quantization will collapse those values and destroy
detection accuracy — typically "only class 0 is recognized". Provide a
head-stripped ONNX, or use the PT->RKNN pipeline (which strips DFL and
emits the rknn_model_zoo 9-output layout).
"""
from __future__ import annotations

from pathlib import Path

from app.converters import progress


def build_from_onnx(onnx_path: Path, out_path: Path, cfg: dict, *, do_quantization: bool) -> Path:
    """Load ``onnx_path``, build, and write ``out_path`` (.rknn)."""
    from rknn.api import RKNN

    rknn = RKNN(verbose=True)

    mean = list(cfg.get("mean_values") or [0, 0, 0])
    std = list(cfg.get("std_values") or [255, 255, 255])
    kwargs = {
        "mean_values": [mean],
        "std_values": [std],
        "target_platform": cfg["target_platform"],
    }
    if do_quantization:
        kwargs["quantized_dtype"] = cfg.get("quantized_dtype", "w8a8")
        kwargs["quantized_method"] = cfg.get("quantized_method", "channel")
        kwargs["quantized_algorithm"] = cfg.get("quantized_algorithm", "mmse")

    progress.log("Configuring RKNN ...")
    rknn.config(**kwargs)

    progress.progress(20, f"Loading ONNX {onnx_path.name} ...")
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        rknn.release()
        raise RuntimeError(f"rknn.load_onnx failed (code={ret})")

    if do_quantization:
        dataset = cfg.get("dataset_txt")
        if not dataset:
            rknn.release()
            raise ValueError("INT8 quantization requires a calibration set (dataset_txt missing).")
        progress.progress(
            40,
            f"Building with INT8 quantization ({kwargs.get('quantized_dtype', 'w8a8')}/"
            f"{kwargs.get('quantized_algorithm', 'mmse')}) — this can take minutes ...",
        )
        ret = rknn.build(do_quantization=True, dataset=str(dataset))
    else:
        progress.progress(40, "Building (no quantization, FP) ...")
        ret = rknn.build(do_quantization=False)
    if ret != 0:
        rknn.release()
        raise RuntimeError(f"rknn.build failed (code={ret})")

    progress.progress(90, f"Exporting {out_path.name} ...")
    ret = rknn.export_rknn(str(out_path))
    rknn.release()
    if ret != 0:
        raise RuntimeError(f"rknn.export_rknn failed (code={ret})")

    progress.progress(100, f"RKNN written: {out_path.name}")
    return out_path


def convert(cfg: dict, work_dir: Path) -> Path:
    model_path = Path(cfg["model_path"])
    stem = Path(cfg.get("model_name", model_path.name)).stem
    out_path = work_dir / f"{stem}.rknn"
    do_q = bool(cfg["do_quantization"]) and bool(cfg.get("dataset_txt"))
    return build_from_onnx(model_path, out_path, cfg, do_quantization=do_q)
