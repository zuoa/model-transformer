"""In-memory job store + subprocess orchestration + SSE fan-out.

Single FastAPI process => an in-memory dict is sufficient. Each job:

* has a working directory under WORK_BASE_DIR holding the uploaded model, the
  (optional) calibration zip, the generated job_config.json, and the output.
* runs in an isolated subprocess (``python -m app.converters.worker``).
* fans its stdout line-by-line to every SSE subscriber via a bounded
  asyncio.Queue per subscriber.

RKNN builds are memory-heavy, so a module-level semaphore serializes them
(RKNN_CONCURRENCY, default 1); ONNX-only exports bypass it.

A TTL sweeper (started by the app) deletes finished/failed job directories so
clients that never download don't leak disk.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app import config as cfg_mod
from app.schemas import JobParams, JobStatus, Pipeline

# How many recent log lines to keep for the GET /api/jobs/{id} snapshot.
PROGRESS_SNAPSHOT_LINES = 200

JOBS: dict[str, "Job"] = {}
_rknn_sem: Optional[asyncio.Semaphore] = None


def _sem() -> asyncio.Semaphore:
    """Lazy semaphore (created on first use inside the running loop)."""
    global _rknn_sem
    if _rknn_sem is None:
        _rknn_sem = asyncio.Semaphore(cfg_mod.RKNN_CONCURRENCY)
    return _rknn_sem


@dataclass
class Subscriber:
    queue: asyncio.Queue
    truncated: bool = False


@dataclass
class Job:
    id: str
    pipeline: str
    params: JobParams
    work_dir: Path
    config_path: Path
    status: str = "queued"  # queued | running | success | failed
    progress: deque = field(default_factory=lambda: deque(maxlen=PROGRESS_SNAPSHOT_LINES))
    result_name: Optional[str] = None
    error: Optional[str] = None
    subscribers: list[Subscriber] = field(default_factory=list)
    process: Optional[asyncio.subprocess.Process] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def is_rknn(self) -> bool:
        return self.pipeline in (Pipeline.PT_TO_RKNN.value, Pipeline.ONNX_TO_RKNN.value)


def create_job(
    job_id: str,
    params: JobParams,
    work_dir: Path,
    model_path: Path,
    model_name: str,
    calib_zip: Optional[Path],
) -> Job:
    work_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "pipeline": params.pipeline.value,
        "work_dir": str(work_dir),
        "model_path": str(model_path),
        "model_name": model_name,
        "calib_zip": str(calib_zip) if calib_zip else None,
        "is_int8": params.is_int8(),
        "opset": params.opset,
        "imgsz": params.imgsz,
        "simplify": params.simplify,
        "dynamic": params.dynamic,
        "half": params.half,
        "batch": params.batch,
        "target_platform": params.target_platform,
        "quantize": params.quantize,
        "do_quantization": params.do_quantization,
        "quantized_dtype": params.quantized_dtype,
        "quantized_method": params.quantized_method,
        "quantized_algorithm": params.quantized_algorithm,
        "mean_values": params.mean_values,
        "std_values": params.std_values,
    }
    config_path = work_dir / "job_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    job = Job(
        id=job_id,
        pipeline=params.pipeline.value,
        params=params,
        work_dir=work_dir,
        config_path=config_path,
    )
    JOBS[job_id] = job
    return job


async def start_job(job: Job) -> None:
    """Acquire the RKNN semaphore (if needed), spawn the worker, and pump output."""
    sem = _sem() if job.is_rknn() else None
    if sem is not None:
        await sem.acquire()
    try:
        job.status = "running"
        _broadcast(job, {"type": "status", "status": "running"})
        env = {
            **_subprocess_env(),
            "PYTHONUNBUFFERED": "1",  # rknn C++ logging isn't line-buffered by default
            "YOLO_CONFIG_DIR": str(job.work_dir / ".ultralytics"),  # isolate per job
        }
        job.process = await asyncio.create_subprocess_exec(
            sys_python(), "-m", "app.converters.worker", str(job.config_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # merge stderr into the stream
            env=env,
        )
        await _pump(job)
    finally:
        if sem is not None:
            sem.release()


def sys_python() -> str:
    return "python3"


def _subprocess_env(self=None) -> dict:  # noqa: ARG001
    import os
    return dict(os.environ)


async def _pump(job: Job) -> None:
    assert job.process and job.process.stdout
    while True:
        line = await job.process.stdout.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not text:
            continue
        _handle_line(job, text)
    rc = await job.process.wait()
    _finalize(job, rc)


def _handle_line(job: Job, text: str) -> None:
    event = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "type" in parsed:
            event = parsed
    except (ValueError, json.JSONDecodeError):
        event = None

    if event is None:
        # Raw library log line -> wrap as a log event.
        event = {"type": "log", "msg": text}

    etype = event.get("type")
    if etype == "done":
        job.result_name = event.get("result")
    elif etype == "error":
        job.error = event.get("msg", "conversion failed")
    elif etype == "log":
        msg = event.get("msg")
        if msg:
            job.progress.append(msg)
    elif etype == "progress":
        msg = event.get("msg") or f"{event.get('pct', '')}%"
        if msg:
            job.progress.append(msg)
    # start / status / snapshot carry no log line.

    _broadcast(job, event)


def _finalize(job: Job, rc: int) -> None:
    if rc == 0 and job.result_name and (job.work_dir / job.result_name).exists():
        job.status = "success"
    else:
        job.status = "failed"
        if not job.error:
            job.error = f"worker exited with code {rc}"
    job.finished_at = time.time()
    _broadcast(
        job,
        {
            "type": "end",
            "status": job.status,
            "result": job.result_name,
            "error": job.error,
        },
    )


def _broadcast(job: Job, event: dict) -> None:
    """Push an event to every subscriber's queue, dropping oldest when full."""
    payload = json.dumps(event, ensure_ascii=False)
    for sub in list(job.subscribers):
        q = sub.queue
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            try:
                q.get_nowait()  # drop oldest
            except asyncio.QueueEmpty:
                pass
            if not sub.truncated:
                sub.truncated = True
                q.put_nowait(json.dumps({"type": "log", "msg": "…(older log truncated)…"}))
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass  # give up on this one line for this slow subscriber


def subscribe(job: Job) -> Subscriber:
    sub = Subscriber(queue=asyncio.Queue(maxsize=cfg_mod.SSE_QUEUE_MAXLINES))
    # Replay current snapshot so a late subscriber sees prior progress.
    snapshot = {"type": "snapshot", "status": job.status, "progress": list(job.progress)}
    sub.queue.put_nowait(json.dumps(snapshot))
    if job.status in ("success", "failed"):
        sub.queue.put_nowait(
            json.dumps({"type": "end", "status": job.status, "result": job.result_name, "error": job.error})
        )
    job.subscribers.append(sub)
    return sub


def unsubscribe(job: Job, sub: Subscriber) -> None:
    if sub in job.subscribers:
        job.subscribers.remove(sub)


def get(job_id: str) -> Optional[Job]:
    return JOBS.get(job_id)


def status(job: Job) -> JobStatus:
    return JobStatus(
        id=job.id,
        pipeline=job.pipeline,
        status=job.status,
        progress=list(job.progress),
        result_name=job.result_name,
        error=job.error,
    )


def cleanup_expired() -> int:
    """Delete working dirs of jobs past their TTL. Returns count removed."""
    now = time.time()
    removed = 0
    for job_id, job in list(JOBS.items()):
        if job.status in ("success", "failed") and job.finished_at:
            if now - job.finished_at > cfg_mod.JOB_TTL_SECONDS:
                _safe_rmtree(job.work_dir)
                JOBS.pop(job_id, None)
                removed += 1
    return removed


def _safe_rmtree(path: Path) -> None:
    import shutil
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


async def ttl_sweeper() -> None:
    """Background task: periodically purge expired job artifacts."""
    while True:
        await asyncio.sleep(60)
        try:
            cleanup_expired()
        except Exception:
            pass


def new_id() -> str:
    return uuid.uuid4().hex[:16]
