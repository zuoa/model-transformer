"""Structured progress emission for the subprocess worker.

Each call writes exactly one JSON line to stdout (flushed). The server reads
worker stdout line-by-line; lines that parse as JSON with a ``type`` field are
treated as structured events, anything else (e.g. rknn-toolkit2's C++ verbose
logging) is forwarded to clients as a raw log line.

Keeping this in its own module lets both ``worker.py`` and the individual
converter modules emit progress without circular imports.
"""
from __future__ import annotations

import json
import sys


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def log(msg) -> None:
    emit({"type": "log", "msg": str(msg)})


def progress(pct: int, msg=None) -> None:
    emit({"type": "progress", "pct": int(pct), "msg": None if msg is None else str(msg)})


def start(pipeline: str) -> None:
    emit({"type": "start", "pipeline": pipeline})


def done(result_name: str) -> None:
    emit({"type": "done", "result": result_name})


def fail(msg: str, trace: str = "") -> None:
    emit({"type": "error", "msg": msg, "trace": trace})
