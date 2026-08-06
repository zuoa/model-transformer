"""Pydantic request/response models and the mutual-exclusion validation rules.

The schema is the single source of truth for "which parameter combinations make
sense". Invalid combos (e.g. dynamic shapes + RKNN, FP16 + INT8) are rejected at
job creation so we never spend a multi-minute RKNN build on a doomed request.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.config import SUPPORTED_TARGETS

# Allowed values for the rknn-toolkit2 quantization knobs (verified against the
# airockchip/rknn-toolkit2 v2.3.2 examples + changelog).
QUANTIZED_DTYPES = ("w8a8", "w16a16", "w4a16", "w4a8", "bf16")
QUANTIZED_METHODS = ("channel", "layer")
QUANTIZED_ALGORITHMS = ("normal", "mmse", "kl")


class Pipeline(str, Enum):
    PT_TO_ONNX = "pt_to_onnx"
    PT_TO_RKNN = "pt_to_rknn"
    ONNX_TO_RKNN = "onnx_to_rknn"


class JobParams(BaseModel):
    """All conversion parameters for all pipelines in one model. Fields that
    don't apply to the chosen pipeline are simply ignored by the converter."""

    pipeline: Pipeline

    # --- common ---
    imgsz: int = Field(default=640, ge=32, le=4096)
    opset: int = Field(default=17, ge=12, le=19)

    # --- pt -> onnx ---
    simplify: bool = True
    dynamic: bool = False
    half: bool = False
    batch: int = Field(default=1, ge=1, le=64)

    # --- rknn (shared) ---
    target_platform: Optional[str] = None

    # --- pt -> rknn (ultralytics native format='rknn') ---
    # 8 = INT8 (needs calibration), 16 = FP16 (w16a16), None/0 = FP32
    quantize: Optional[int] = Field(default=None)

    # --- onnx -> rknn (direct toolkit) ---
    do_quantization: bool = False
    quantized_dtype: str = "w8a8"
    quantized_method: str = "channel"
    quantized_algorithm: str = "mmse"
    mean_values: List[int] = Field(default_factory=lambda: [0, 0, 0])
    std_values: List[int] = Field(default_factory=lambda: [255, 255, 255])

    @field_validator("target_platform")
    @classmethod
    def _check_target(cls, v):
        if v is not None and v not in SUPPORTED_TARGETS:
            raise ValueError(f"target_platform must be one of {SUPPORTED_TARGETS}")
        return v

    @field_validator("quantized_dtype")
    @classmethod
    def _check_dtype(cls, v):
        if v not in QUANTIZED_DTYPES:
            raise ValueError(f"quantized_dtype must be one of {QUANTIZED_DTYPES}")
        return v

    @field_validator("quantized_method")
    @classmethod
    def _check_method(cls, v):
        if v not in QUANTIZED_METHODS:
            raise ValueError(f"quantized_method must be one of {QUANTIZED_METHODS}")
        return v

    @field_validator("quantized_algorithm")
    @classmethod
    def _check_algo(cls, v):
        if v not in QUANTIZED_ALGORITHMS:
            raise ValueError(f"quantized_algorithm must be one of {QUANTIZED_ALGORITHMS}")
        return v

    @field_validator("quantize")
    @classmethod
    def _check_quantize(cls, v):
        if v is not None and v not in (8, 16):
            raise ValueError("quantize must be 8, 16, or null/0")
        return v

    @field_validator("mean_values", "std_values")
    @classmethod
    def _check_triple(cls, v):
        if len(v) != 3:
            raise ValueError("mean_values/std_values must have exactly 3 entries (RGB)")
        return v

    @model_validator(mode="after")
    def _check_pipeline_rules(self):
        p = self.pipeline

        # dynamic shapes are only valid for the pure ONNX path; RKNN needs a
        # fixed input shape.
        if p != Pipeline.PT_TO_ONNX and self.dynamic:
            raise ValueError("dynamic=True is incompatible with RKNN (fixed shape required)")

        if p == Pipeline.PT_TO_ONNX:
            # Nothing else to enforce; target_platform etc. are ignored.
            return self

        # RKNN paths need a target platform.
        if not self.target_platform:
            raise ValueError("target_platform is required for RKNN pipelines")

        if p == Pipeline.PT_TO_RKNN:
            # half (FP16 ONNX) + INT8 quantize is contradictory.
            if self.half and self.quantize == 8:
                raise ValueError("half=True conflicts with INT8 (quantize=8); use quantize=16 for FP16")

        if p == Pipeline.ONNX_TO_RKNN:
            # FP16 ONNX into an INT8 build is contradictory.
            if self.half and self.do_quantization and self.quantized_dtype in ("w8a8", "w4a8"):
                raise ValueError(
                    "half=True (FP16 ONNX) conflicts with INT8 quantization; "
                    "export FP32 ONNX or use quantized_dtype=w16a16"
                )

        return self

    def is_int8(self) -> bool:
        """Whether this job needs a calibration dataset."""
        if self.pipeline == Pipeline.PT_TO_RKNN:
            return self.quantize == 8
        if self.pipeline == Pipeline.ONNX_TO_RKNN:
            return self.do_quantization and self.quantized_dtype in ("w8a8", "w4a8")
        return False


class JobCreated(BaseModel):
    id: str
    status: str
    params: JobParams


class JobStatus(BaseModel):
    id: str
    pipeline: str
    status: str  # queued | running | success | failed
    progress: List[str] = Field(default_factory=list)  # recent log lines
    result_name: Optional[str] = None  # filename of the produced artifact
    error: Optional[str] = None
