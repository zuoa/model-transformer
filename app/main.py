"""FastAPI application: routes, static serving, upload handling, SSE."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import config as cfg_mod, jobs
from app.schemas import JobCreated, JobParams, Pipeline

MOCK = os.environ.get("MT_MOCK") == "1"

app = FastAPI(title="YOLO Model Transformer", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(cfg_mod.STATIC_DIR)), name="static")


@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(jobs.ttl_sweeper())


@app.get("/")
async def index():
    return FileResponse(str(cfg_mod.STATIC_DIR / "index.html"))


@app.get("/api/targets")
async def targets():
    """Supported RKNN platforms + which pipelines this host can run right now."""
    return {
        "targets": list(cfg_mod.SUPPORTED_TARGETS),
        "availability": {
            "pt_to_onnx": MOCK or cfg_mod.can_run(Pipeline.PT_TO_ONNX.value),
            "pt_to_rknn": MOCK or cfg_mod.can_run(Pipeline.PT_TO_RKNN.value),
            "onnx_to_rknn": MOCK or cfg_mod.can_run(Pipeline.ONNX_TO_RKNN.value),
        },
        "mock": MOCK,
    }


async def _save_upload(upload: UploadFile, dest: Path, max_bytes: int) -> int:
    total = 0
    with open(dest, "wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"{dest.name} exceeds {max_bytes // (1024*1024)}MB limit",
                )
            f.write(chunk)
    return total


@app.post("/api/jobs", response_model=JobCreated)
async def create_job(
    params: str = Form(...),
    model: UploadFile = File(...),
    calib: UploadFile | None = File(None),
):
    try:
        p = JobParams.model_validate_json(params)
    except Exception as exc:  # pydantic validation / JSON errors
        raise HTTPException(status_code=422, detail=str(exc))

    # Host capability gate (skipped under MT_MOCK so plumbing tests can run).
    if not MOCK and not cfg_mod.can_run(p.pipeline.value):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Pipeline '{p.pipeline.value}' is not available on this host. "
                "ONNX needs ultralytics; RKNN needs rknn-toolkit2 (Linux x86_64 container)."
            ),
        )

    # Input extension sanity.
    ext = Path(model.filename or "").suffix.lower()
    if p.pipeline == Pipeline.ONNX_TO_RKNN and ext != ".onnx":
        raise HTTPException(status_code=400, detail="ONNX->RKNN requires a .onnx upload")
    if p.pipeline in (Pipeline.PT_TO_ONNX, Pipeline.PT_TO_RKNN) and ext not in (".pt", ".pth"):
        raise HTTPException(status_code=400, detail="This pipeline requires a .pt/.pth upload")

    # INT8 needs a calibration set.
    if p.is_int8() and (calib is None or not calib.filename):
        raise HTTPException(
            status_code=400,
            detail="INT8 quantization requires a calibration image zip (calib field).",
        )

    job_id = jobs.new_id()
    work_dir = cfg_mod.WORK_BASE_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(model.filename or "model.pt").name
    model_path = work_dir / model_name
    await _save_upload(model, model_path, cfg_mod.MAX_MODEL_BYTES)

    calib_zip = None
    if calib is not None and calib.filename:
        calib_zip = work_dir / "calibration.zip"
        await _save_upload(calib, calib_zip, cfg_mod.MAX_CALIB_BYTES)

    job = jobs.create_job(job_id, p, work_dir, model_path, model_name, calib_zip)
    asyncio.create_task(jobs.start_job(job))
    return JobCreated(id=job_id, status=job.status, params=p)


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return jobs.status(job)


@app.get("/api/jobs/{job_id}/download")
async def job_download(job_id: str):
    job = jobs.get(job_id)
    if not job or job.status != "success" or not job.result_name:
        raise HTTPException(status_code=404, detail="result not available")
    path = job.work_dir / job.result_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="result file missing")
    return FileResponse(str(path), filename=job.result_name)


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    async def gen():
        sub = jobs.subscribe(job)
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(sub.queue.get(), timeout=15.0)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"  # SSE comment; keeps proxies from idling
                    continue
                try:
                    evt = json.loads(payload)
                    if evt.get("type") == "end":
                        break
                except (ValueError, json.JSONDecodeError):
                    pass
        finally:
            jobs.unsubscribe(job, sub)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
