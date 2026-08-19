"""Patch YOLO Detect / Segment / Pose / OBB heads for RKNN (rknn_model_zoo).

Official ultralytics ``export(format='rknn')`` traces the *decoded* Detect path
and emits a single tensor ``[1, 4+nc, 8400]`` (xywh + class scores, DFL already
applied). That layout is what causes "only class 0 is recognized" on device:

* rknn_model_zoo post-process expects 3 scales × (box 64-ch DFL, class nc-ch,
  optional score-sum). Feeding it one decoded tensor makes argmax land on
  channel 0.
* A YOLOv5-style decoder (``5+nc`` with objectness) treats YOLOv8 channel 4
  (real class 0) as objectness and shifts every other class.
* INT8 on the concatenated tensor shares one scale between box coords (~0-640)
  and class scores (~0-1), so class channels quantize to 0 and argmax is 0.

This module binds airockchip-compatible ``forward`` methods that return raw
per-scale heads and leave DFL / NMS on the CPU:

    3 scales × (box [1, 64, h, w], cls [1, nc, h, w], cls_sum [1, 1, h, w])
"""
from __future__ import annotations

from types import MethodType
from typing import Any

# Class names that carry a YOLO detect-style head (cv2 box + cv3 cls).
_DETECT_NAMES = frozenset({"Detect", "v10Detect"})
_SEGMENT_NAMES = frozenset({"Segment", "Segment26"})
_POSE_NAMES = frozenset({"Pose", "Pose26"})
_OBB_NAMES = frozenset({"OBB", "OBB26"})
_HEAD_NAMES = _DETECT_NAMES | _SEGMENT_NAMES | _POSE_NAMES | _OBB_NAMES


def _require_cv_heads(module: Any) -> None:
    if not (hasattr(module, "cv2") and hasattr(module, "cv3") and hasattr(module, "nl")):
        raise RuntimeError(
            f"{type(module).__name__} has no cv2/cv3 (YOLO26-style heads are not "
            "supported for the rknn_model_zoo Detect strip). Export ONNX with a "
            "YOLOv8/YOLO11 Detect head, or use airockchip/ultralytics_yolov8."
        )


def detect_outputs(module: Any, x: list) -> list:
    """Raw per-scale box / sigmoid-cls / cls-sum tensors (no DFL)."""
    import torch

    _require_cv_heads(module)
    y: list = []
    for i in range(module.nl):
        y.append(module.cv2[i](x[i]))
        cls = torch.sigmoid(module.cv3[i](x[i]))
        cls_sum = torch.clamp(cls.sum(1, keepdim=True), 0, 1)
        y.append(cls)
        y.append(cls_sum)
    return y


def _forward_detect(self, x, *args, **kwargs):
    return detect_outputs(self, x)


def _forward_segment(self, x, *args, **kwargs):
    p = self.proto(x[0])
    mc = [self.cv4[i](x[i]) for i in range(self.nl)]
    det = detect_outputs(self, x)
    # Per scale: box, cls, cls_sum, mask-coeff; proto last. Matches airockchip.
    out: list = []
    for i in range(self.nl):
        out.extend(det[i * 3 : (i + 1) * 3])
        out.append(mc[i])
    out.append(p)
    return out


def _forward_pose(self, x, *args, **kwargs):
    det = detect_outputs(self, x)
    kpt = [self.cv4[i](x[i]) for i in range(self.nl)]
    return list(det) + kpt


def _forward_obb(self, x, *args, **kwargs):
    import torch

    det = detect_outputs(self, x)
    angle = [torch.sigmoid(self.cv4[i](x[i])) for i in range(self.nl)]
    return list(det) + angle


_FORWARD = {
    **{n: _forward_detect for n in _DETECT_NAMES},
    **{n: _forward_segment for n in _SEGMENT_NAMES},
    **{n: _forward_pose for n in _POSE_NAMES},
    **{n: _forward_obb for n in _OBB_NAMES},
}


def apply(model) -> Any:
    """Bind RKNN raw-head forwards on every Detect-family module in ``model``.

    ``model`` is the inner ``nn.Module`` (``YOLO(...).model``), not the
    ultralytics wrapper. Also forces ``export=True``, ``format='rknn'``, and
    disables the end2end / one-to-one branch so YOLO11/v10 keep cv2/cv3.
    Returns the patched head module (the last matching module).
    """
    head = None
    for m in model.modules():
        name = type(m).__name__
        if name not in _HEAD_NAMES:
            continue
        m.export = True
        m.format = "rknn"
        if hasattr(m, "end2end"):
            m.end2end = False
        if hasattr(model, "end2end"):
            model.end2end = False
        m.forward = MethodType(_FORWARD[name], m)
        head = m
    if head is None:
        raise RuntimeError(
            "No YOLO Detect/Segment/Pose/OBB head found; cannot strip DFL for RKNN."
        )
    _require_cv_heads(head)
    return head


def describe(head, outputs: list) -> str:
    """Human-readable layout for the conversion log."""
    nc = getattr(head, "nc", "?")
    nl = getattr(head, "nl", 3)
    kind = type(head).__name__
    lines = [
        f"RKNN Detect head stripped ({kind}): nc={nc}, scales={nl}, "
        f"{len(outputs)} output tensor(s).",
        "Layout (rknn_model_zoo): per scale → box[1,64,h,w] + cls[1,nc,h,w] + cls_sum[1,1,h,w].",
        "DFL / dist2bbox stay on CPU. Set OBJ_CLASS_NUM in postprocess.h to nc.",
    ]
    for i, t in enumerate(outputs):
        try:
            shape = tuple(int(s) for s in t.shape)
        except Exception:
            shape = getattr(t, "shape", "?")
        lines.append(f"  output{i}: {shape}")
    return "\n".join(lines)
