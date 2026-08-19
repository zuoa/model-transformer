"""Calibration dataset preparation for RKNN INT8 quantization.

* :func:`prepare` — unzip images and write ``dataset.txt`` (absolute paths,
  one per line) for ``rknn.build(do_quantization=True, dataset=...)``.
  Used by both PT->RKNN and ONNX->RKNN.

* :func:`prepare_with_yaml` — leftover helper that also writes a YOLO
  ``data.yaml``. Not used by the current converters.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from app.converters import progress

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_IMAGES = 5


def _is_decodable_image(path: Path) -> bool:
    if path.suffix.lower() not in IMAGE_EXTS:
        return False
    try:
        import cv2

        return cv2.imread(str(path)) is not None
    except Exception:
        try:
            from PIL import Image  # type: ignore

            with Image.open(path):
                return True
        except Exception:
            return True  # extension-only fallback


def _extract(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    progress.log(f"Extracting calibration archive {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def prepare(zip_path: Path, work_dir: Path) -> tuple[Path, Path]:
    """For ONNX->RKNN: unzip, keep decodable images, write ``dataset.txt``."""
    calib_dir = work_dir / "calib_raw"
    _extract(zip_path, calib_dir)
    images = sorted(p for p in calib_dir.rglob("*") if _is_decodable_image(p))
    if len(images) < MIN_IMAGES:
        raise ValueError(
            f"Calibration set has only {len(images)} usable images; rknn-toolkit2 "
            f"needs ~20-100 for good INT8 scales (minimum {MIN_IMAGES} enforced)."
        )
    dataset_txt = work_dir / "dataset.txt"
    dataset_txt.write_text("\n".join(str(p.resolve()) for p in images) + "\n", encoding="utf-8")
    progress.log(f"Calibration ready: {len(images)} images -> {dataset_txt.name}")
    return calib_dir, dataset_txt


def prepare_with_yaml(zip_path: Path, work_dir: Path) -> tuple[Path, Path, Path]:
    """For PT->RKNN via ultralytics: build a minimal YOLO detection dataset
    (images/ + empty labels/ + data.yaml) and a dataset.txt.

    Returns ``(calib_dir, dataset_txt, data_yaml)``.
    """
    calib_dir = work_dir / "calib"
    images_dir = calib_dir / "images"
    labels_dir = calib_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    raw = work_dir / "calib_raw"
    _extract(zip_path, raw)

    moved: list[Path] = []
    for i, p in enumerate(sorted(q for q in raw.rglob("*") if _is_decodable_image(q))):
        dst = images_dir / f"calib_{i:04d}{p.suffix.lower()}"
        shutil.move(str(p), dst)
        (labels_dir / f"calib_{i:04d}.txt").touch()  # empty label -> valid detection dataset
        moved.append(dst)
    shutil.rmtree(raw, ignore_errors=True)

    if len(moved) < MIN_IMAGES:
        raise ValueError(
            f"Calibration set has only {len(moved)} usable images; need ~20-100 "
            f"(minimum {MIN_IMAGES} enforced)."
        )

    dataset_txt = work_dir / "dataset.txt"
    dataset_txt.write_text("\n".join(str(p.resolve()) for p in moved) + "\n", encoding="utf-8")

    # YOLO dataset YAML ultralytics' loader can resolve for calibration.
    data_yaml = calib_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {calib_dir.resolve()}\n"
        f"train: images\n"
        f"val: images\n"
        f"names:\n  0: object\n",
        encoding="utf-8",
    )
    progress.log(f"Calibration dataset ready: {len(moved)} images (YOLO layout) -> {data_yaml.name}")
    return calib_dir, dataset_txt, data_yaml
