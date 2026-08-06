# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A web app that converts YOLO models to **ONNX** or **RKNN**, exposing conversion
and quantization parameters (including INT8 calibration). Pure Python (FastAPI)
backend, embedded static HTML/JS frontend (no npm build), deployed as a single
Docker container on **x86_64 Linux**.

## Commands

There is **no test suite or linter configured**. Verification is manual via mock
mode (see below). All commands run from the repo root.

```bash
# --- Plumbing test / CI (no torch, no rknn; lightweight deps only) ---
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
MT_MOCK=1 uvicorn app.main:app --reload          # http://localhost:8000 ; all pipelines return fake output

# --- Local dev, PT→ONNX only (works on macOS; RKNN can't run on Mac) ---
pip install -r requirements.txt -r requirements-convert.txt   # pulls torch (~2GB)
uvicorn app.main:app --reload

# --- Full stack incl. RKNN (Linux x86_64; RKNN runs only there) ---
#   1. drop rknn_toolkit2-2.3.2 cp312 wheel into wheels/  (see wheels/README.md)
docker compose up --build                           # http://localhost:8000
docker compose --profile dev up --build             # live-reload source from host

# --- Deploy the prebuilt image from GHCR (no local build) ---
docker compose -f docker-compose.deploy.yml up -d   # pulls ghcr.io/zuoa/model-transformer:latest

# --- Run the conversion worker directly (debug a single conversion) ---
python3 -m app.converters.worker <work_dir>/job_config.json
```

Key env vars: `MT_MOCK` (fake converter), `MT_WORK_DIR`, `MT_RKNN_CONCURRENCY`
(default 1), `MT_MAX_MODEL_MB`, `MT_JOB_TTL_SECONDS`. See `app/config.py`.

## Architecture (the parts that span files)

**Subprocess isolation + SSE streaming.** The FastAPI process never runs a
conversion itself. `POST /api/jobs` writes a `job_config.json` and spawns
`python -m app.converters.worker` via `asyncio.create_subprocess_exec`
(`app/jobs.py:start_job`). Rationale: `rknn-toolkit2` is a C++ extension with
global state and nonzero segfault risk — a crash in the worker must not kill the
server. Worker stdout/stderr (merged, `PYTHONUNBUFFERED=1`) is read line-by-line
in `app/jobs.py:_pump` and fanned out to every SSE subscriber
(`GET /api/jobs/{id}/events`) via a bounded `asyncio.Queue` per subscriber.

**Worker line protocol.** `app/converters/progress.py` emits one JSON object per
stdout line (`{type: start|log|progress|done|error}`). `jobs._handle_line`
parses each line: JSON with a `type` field → structured event; anything else
(ultralytics / rknn verbose logging) → wrapped as a `log` event. When adding
converter code, emit progress through `progress.log/progress/done`, never
`print`.

**Three pipelines → three converter modules**, dispatched by `pipeline` string
in `worker.dispatch`:
- `pt_to_onnx` → `onnx.py` (ultralytics `format="onnx"`)
- `pt_to_rknn` → `ultralytics_rknn.py` (ultralytics native `format="rknn"` — **the
  recommended RKNN path** because it configures the YOLO Detect head correctly)
- `onnx_to_rknn` → `rknn.py` (direct `rknn-toolkit2`, full param set). User owns
  head correctness — the UI warns that a stock YOLOv8/v11 ONNX INT8-quantized
  this way loses accuracy.

**`app/schemas.py` is the single source of truth for valid parameter
combinations.** Mutual-exclusion rules are enforced with Pydantic
`model_validator` so invalid combos (`dynamic=True` + RKNN, `half=True` + INT8,
unknown `target_platform`, missing target for RKNN) are rejected at job creation
— never after a multi-minute build. `is_int8()` drives whether calibration is
required (checked server-side in `app/main.py`).

**Calibration differs by pipeline.** `calibration.prepare` unzips images and
writes `dataset.txt` (absolute paths, one per line) — used by `rknn.py`'s
`build(dataset=...)`. For the ultralytics RKNN path the *directory* is passed as
`data=` instead. Worker calls `prepare` only for INT8 jobs before dispatch.

**Concurrency & cleanup.** RKNN builds are memory-heavy, so a module-level
`asyncio.Semaphore(RKNN_CONCURRENCY=1)` serializes them (`jobs.py`); ONNX exports
bypass it. A TTL sweeper deletes finished jobs' work dirs.

## Hard constraints

- **`rknn-toolkit2` runs only on x86_64 Linux** (Rockchip officially tests
  Ubuntu 18.04/20.04/22.04; the cp312 wheel is manylinux2014, so it also runs on
  the Debian-based `python:3.12-slim` we use). Not on macOS or Windows natively.
  The ONNX path works everywhere; RKNN works only inside the container.
  `config.can_run()` gates availability at the API (skipped under `MT_MOCK`).
- **The rknn wheel is vendored, not on PyPI.** It goes in `wheels/` and the
  Dockerfile installs it from disk. The active repo is `airockchip/rknn-toolkit2`
  v2.3.2 (the `rockchip-linux` fork is archived).
- **Dependency pins in `requirements-convert.txt` must stay compatible with
  rknn-toolkit2 v2.3.2:** `torch<=2.4.0`, `numpy<=1.26.4`,
  `protobuf 4.21.6..4.25.4`. Bumping these can cause runtime ABI crashes. Uses
  `onnxslim` (not the older `onnx-simplifier`).
- **Target platform whitelist** lives in `config.SUPPORTED_TARGETS`; new RKNN
  chips must be added there (and the schema validator picks it up automatically).
- Target Python is **3.12** in the container (`python:3.12-slim` base, cp312 rknn
  wheel). The code uses only 3.10+ syntax, so it also runs on the macOS dev box.
- **CI image (`.github/workflows/docker.yml`)** fetches the rknn wheel at build
  time via sparse checkout of `airockchip/rknn-toolkit2` (version pinned by the
  `RKNN_VERSION` env in the workflow), so the published `linux/amd64` image has
  full RKNN support. It pushes to `ghcr.io/<repo-owner>/<repo-name>`; the deploy
  compose (`docker-compose.deploy.yml`) pulls `ghcr.io/zuoa/model-transformer:latest`.
