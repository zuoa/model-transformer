# Full image: PT->ONNX (ultralytics) and PT/ONNX->RKNN (rknn-toolkit2 v2.3.2).
# Python 3.12 (cp312 wheel). Build on an x86_64 Linux host (or via Docker on
# Mac) — rknn-toolkit2 only runs on x86_64 Linux; the ONNX path works everywhere.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Native runtime libs (Debian names) — covers opencv-python's GL/glib/SM needs.
# gcc/g++ for any source builds; the rknn wheel itself is manylinux2014 (portable).
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libprotobuf-dev zlib1g libsm6 libgl1 libglib2.0-0 \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# App + conversion deps. Pins chosen to satisfy rknn-toolkit2 v2.3.2
# (torch<=2.4.0, numpy<=1.26.4, protobuf 4.21.6..4.25.4); all have cp312 wheels.
# opencv-python (not headless) matches the rknn wheel's declared dependency.
COPY requirements.txt requirements-convert.txt ./
RUN pip3 install -r requirements.txt -r requirements-convert.txt

# rknn-toolkit2 is NOT on PyPI. Drop the v2.3.2 cp312 wheel in ./wheels/ (see
# wheels/README.md). If absent, the image still builds but RKNN pipelines are
# disabled at runtime (ONNX path keeps working).
COPY wheels/ ./wheels/
RUN if ls wheels/*.whl >/dev/null 2>&1; then \
        pip3 install wheels/*.whl; \
    else \
        echo "WARN: no rknn wheel in wheels/ — RKNN pipelines will be unavailable"; \
    fi

COPY app ./app
COPY static ./static

# Persist converted artifacts and Ultralytics state under /data.
ENV MT_WORK_DIR=/data/work \
    YOLO_CONFIG_DIR=/data/.ultralytics
RUN mkdir -p /data/work /data/.ultralytics
VOLUME ["/data"]

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
