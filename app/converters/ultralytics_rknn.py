"""PT -> RKNN with a rknn_model_zoo-compatible Detect head.

Official ultralytics ``export(format='rknn')`` keeps DFL / dist2bbox in the
graph and emits a single ``[1, 4+nc, 8400]`` tensor. Board-side demos that
expect 3 raw scales then report every box as class 0. This path:

  1. patches Detect (and Segment / Pose / OBB) to return raw per-scale heads
  2. exports that multi-output ONNX
  3. converts with rknn-toolkit2 (INT8 uses dataset.txt, same as ONNX->RKNN)
"""
from __future__ import annotations

from pathlib import Path

from app.converters import progress, rknn, rknn_head


def _onnx_wrapper(inner):
    """Make a list-returning Detect head a tuple so ONNX keeps every output."""
    import torch

    class _OnnxWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            y = self.model(x)
            if isinstance(y, (list, tuple)):
                return tuple(y)
            return y

    return _OnnxWrapper(inner)


def _prefer_split_c2f(model) -> None:
    """Cleaner ONNX graph (same as ultralytics' exporter)."""
    for m in model.modules():
        if type(m).__name__ == "C2f" and hasattr(m, "forward_split"):
            m.forward = m.forward_split


def _export_stripped_onnx(inner, onnx_path: Path, imgsz: int, opset: int) -> list:
    import torch
    wrapper = _onnx_wrapper(inner)
    wrapper.eval()
    im = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        outs = wrapper(im)
    if not isinstance(outs, tuple):
        outs = (outs,)
    names = [f"output{i}" for i in range(len(outs))]
    progress.log(f"Exporting stripped ONNX ({len(outs)} outputs, opset={opset}, imgsz={imgsz}) ...")
    torch.onnx.export(
        wrapper,
        im,
        str(onnx_path),
        input_names=["images"],
        output_names=names,
        opset_version=opset,
        do_constant_folding=True,
    )
    return list(outs)


def _slim_onnx(onnx_path: Path) -> None:
    try:
        import onnxslim
        import onnx
    except ImportError:
        progress.log("onnxslim not installed; skipping ONNX slim")
        return
    progress.log(f"Slimming ONNX with onnxslim {getattr(onnxslim, '__version__', '')} ...")
    model = onnxslim.slim(onnx.load(str(onnx_path)))
    onnx.save(model, str(onnx_path))


def convert(cfg: dict, work_dir: Path) -> Path:
    from ultralytics import YOLO

    model_path = Path(cfg["model_path"])
    stem = Path(cfg.get("model_name", model_path.name)).stem
    out_path = work_dir / f"{stem}.rknn"
    onnx_path = work_dir / f"{stem}.rknn-head.onnx"
    imgsz = int(cfg["imgsz"])
    opset = int(cfg["opset"])

    progress.log(f"Loading {model_path.name} with ultralytics ...")
    model = YOLO(str(model_path))
    if hasattr(model, "fuse"):
        model.fuse()
    inner = model.model
    inner.eval()
    inner.float()
    for p in inner.parameters():
        p.requires_grad = False
    if hasattr(inner, "end2end"):
        inner.end2end = False

    _prefer_split_c2f(inner)
    head = rknn_head.apply(inner)

    names = getattr(model, "names", None) or {}
    nc = getattr(head, "nc", len(names))
    progress.log(f"Model classes nc={nc} names={names}")

    progress.progress(25, "Exporting rknn_model_zoo Detect head (DFL stripped) ...")
    outs = _export_stripped_onnx(inner, onnx_path, imgsz, opset)
    progress.log(rknn_head.describe(head, outs))
    if cfg.get("simplify", True):
        _slim_onnx(onnx_path)

    do_q = int(cfg.get("quantize") or 0) == 8
    if do_q and not cfg.get("dataset_txt"):
        raise ValueError("INT8 quantization requires a calibration set (dataset_txt missing).")

    progress.progress(35, f"Converting stripped ONNX -> RKNN (target={cfg['target_platform']}) ...")
    rknn.build_from_onnx(onnx_path, out_path, cfg, do_quantization=do_q)

    if onnx_path.exists():
        onnx_path.unlink()
    if not out_path.is_file():
        raise RuntimeError(f"rknn output is not a file at {out_path}")
    return out_path
