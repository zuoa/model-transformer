# YOLO Model Transformer

A small web app to convert YOLO models to **ONNX** or **RKNN**, with conversion
and quantization options (including INT8 calibration for RKNN).

- **PT → ONNX** — via `ultralytics` (opset, simplify, dynamic, half, imgsz, batch).
- **PT → RKNN** — via `ultralytics` native `format="rknn"` (target platform,
  INT8/FP16/FP, calibration). Recommended path — handles the YOLO Detect head
  correctly so INT8 accuracy doesn't collapse.
- **ONNX → RKNN** — via `rknn-toolkit2` directly, with the full quantization
  parameter set (quantized dtype/method/algorithm, mean/std). Bring a
  **head-stripped** ONNX; see the in-app warning.

## Architecture

Single FastAPI process + an isolated subprocess per conversion job + an embedded
static frontend (no npm build). Job stdout is streamed to the browser over SSE.
Conversion jobs run in a subprocess so a `rknn-toolkit2` crash can't take the
server down, and RKNN builds are serialized (concurrency = 1) to avoid OOM.

```
browser ── multipart upload + params ──▶ FastAPI ──▶ python -m app.converters.worker
   ▲                                            │                 (ultralytics / rknn-toolkit2)
   └──────── SSE progress / download ◀─────────┴─ asyncio.create_subprocess_exec
```

Key files: `app/main.py` (routes, upload, SSE), `app/jobs.py` (job store,
subprocess pump, fan-out, TTL), `app/schemas.py` (params + mutual-exclusion
rules), `app/converters/{onnx,ultralytics_rknn,rknn,calibration,worker}.py`.

## Quickstart

### Full stack (ONNX **and** RKNN) — Docker on x86_64 Linux

`rknn-toolkit2` only runs on x86_64 Linux, so run the whole thing in Docker.

1. Drop the rknn-toolkit2 v2.3.2 cp312 wheel into `wheels/` (see
   `wheels/README.md`).
2. Build & run:
   ```bash
   docker compose up --build
   ```
3. Open <http://localhost:8000>.

Dev with live-reload: `docker compose --profile dev up --build`.

### Local dev (PT → ONNX only) — macOS/Linux without Docker

RKNN is unavailable natively on macOS (no `rknn-toolkit2` wheel). The ONNX path
works:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-convert.txt   # pulls torch (~2GB)
uvicorn app.main:app --reload
```

## CI & deployment

A GitHub Actions workflow (`.github/workflows/docker.yml`) builds the image and
pushes it to the GitHub Container Registry. It fetches the `rknn-toolkit2`
wheel at build time (sparse checkout of `airockchip/rknn-toolkit2`), so the
published image has full RKNN support. It triggers on push to `main`/`master`,
on `v*` tags, and manually via *Run workflow*.

- Image: `ghcr.io/<owner>/<repo>` (e.g. `ghcr.io/zuoa/model-transformer` for a
  repo at `github.com/zuoa/model-transformer`).
- Tags: `latest`, branch name, `v1.2.3`/`1.2` on tags, and `sha-<short>`.
- Build target is `linux/amd64` only (rknn-toolkit2 is x86_64-Linux-only).

Deploy from the registry (no local build) with the deploy compose file:

```bash
# If the package is private, authenticate once:
echo "<GHCR_PAT>" | docker login ghcr.io -u zuoa --password-stdin
docker compose -f docker-compose.deploy.yml up -d
```

Pin a specific build by editing `image:` in `docker-compose.deploy.yml`
(e.g. `ghcr.io/zuoa/model-transformer:1.2.3` or `:sha-abc1234`).

> The workflow uses the built-in `GITHUB_TOKEN`, which can only push to the
> namespace of the repo owner. If you want the image under `zuoa` but the repo
> lives under another owner, build/push with a PAT instead.

## Configuration (env vars)

| Var | Default | Meaning |
|---|---|---|
| `MT_WORK_DIR` | `./work` | Job working directories |
| `MT_MAX_MODEL_MB` | `2048` | Max uploaded model size |
| `MT_MAX_CALIB_MB` | `512` | Max uploaded calibration zip size |
| `MT_JOB_TTL_SECONDS` | `3600` | Time finished jobs' artifacts are kept |
| `MT_RKNN_CONCURRENCY` | `1` | Max concurrent RKNN builds |
| `MT_MOCK` | unset | `=1` runs the worker with a fake converter (plumbing tests/CI) |

## Testing the plumbing without torch/rknn

```bash
pip install -r requirements.txt          # lightweight deps only
MT_MOCK=1 uvicorn app.main:app --reload  # all pipelines return mock output
```

## Caveats

- **Detect-head INT8 accuracy**: a stock YOLOv8/v11 ONNX (with the DFL/decode
  layer) quantized to INT8 via ONNX→RKNN loses accuracy. Use PT→RKNN, or supply
  a head-stripped ONNX. The UI warns about this on the ONNX→RKNN pipeline.
- **`dynamic=True` is rejected for RKNN** (fixed input shape required), and
  **FP16 + INT8 is rejected** (use `quantize=16` / `quantized_dtype=w16a16`).
- **RKNN is x86_64-Linux-only.** On macOS, run it inside the Docker container.
