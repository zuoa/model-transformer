"""Calibration dataset preparation for RKNN INT8 quantization.

rknn-toolkit2's ``build(do_quantization=True, dataset=...)`` expects ``dataset``
to be a text file listing one image path per line. We unzip the user's uploaded
zip into the job work dir, keep only decodable images, and write that text file.

ultralytics' native ``format='rknn'`` path takes a directory (``data=``) instead;
we expose both ``calib_dir`` and ``dataset_txt`` from :func:`prepare`.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from app.converters import progress

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jpeg"}


def _is_decodable_image(path: Path) -> bool:
    ext = path.suffix.lower()
    if ext not in IMAGE_EXTS:
        return False
    # Prefer an actual decode check when opencv is available; fall back to the
    # PIL/imread import. If neither imaging lib is present, trust the extension.
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


def prepare(zip_path: Path, work_dir: Path) -> tuple[Path, Path]:
    """Unzip ``zip_path`` under ``work_dir/calib`` and build ``dataset.txt``.

    Returns ``(calib_dir, dataset_txt)``. Raises ValueError if no usable images
    were found, so the worker can fail the job before the expensive build.
    """
    calib_dir = work_dir / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)

    progress.log(f"Extracting calibration archive {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(calib_dir)

    images = sorted(p for p in calib_dir.rglob("*") if _is_decodable_image(p))
    if len(images) < 5:
        raise ValueError(
            f"Calibration set has only {len(images)} usable images; rknn-toolkit2 "
            "needs ~20-100 for good INT8 scales (minimum 5 enforced)."
        )

    dataset_txt = work_dir / "dataset.txt"
    dataset_txt.write_text("\n".join(str(p.resolve()) for p in images) + "\n", encoding="utf-8")
    progress.log(f"Calibration ready: {len(images)} images -> {dataset_txt.name}")
    return calib_dir, dataset_txt
