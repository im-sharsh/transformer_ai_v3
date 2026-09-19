"""Minimal hardened HTTP boundary for AutoData.

Run: uvicorn api_server:app --host 0.0.0.0 --port 8000
The API intentionally exposes profiling/validation first; model training remains an offline job.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src.api.service import analyze_dataset, prepare_dataset_archive, validate_dataset
from src.utils.hardware import detect_hardware

app = FastAPI(title="AutoData API", version="0.1.0", docs_url="/docs", redoc_url=None)


def _parse_overrides(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"overrides must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="overrides must be a JSON object")
    return {str(k): str(v) for k, v in value.items() if v is not None}


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "upload.csv").suffix.lower()
    if suffix not in {".csv", ".parquet", ".pq", ".json", ".jsonl"}:
        raise HTTPException(status_code=415, detail=f"unsupported file type: {suffix}")
    handle = tempfile.NamedTemporaryFile(prefix="autodata_", suffix=suffix, delete=False)
    try:
        while chunk := upload.file.read(1024 * 1024):
            handle.write(chunk)
    finally:
        handle.close()
    return Path(handle.name)


@app.get("/health")
def health():
    hw = detect_hardware()
    return {"status": "ok", "cuda": bool(hw.get("cuda")), "gpu_name": hw.get("gpu_name")}


@app.get("/metadata")
def metadata():
    return {
        "service": "AutoData",
        "api_version": "0.1.0",
        "supported_formats": ["csv", "parquet", "json", "jsonl"],
        "canonical_roles": ["target", "entity", "time", "amount", "category", "counterparty"],
        "gpu_required": False,
    }


@app.post("/profile")
def profile(file: UploadFile = File(...), target: str | None = Form(None), overrides: str | None = Form(None)):
    path = _save_upload(file)
    try:
        return analyze_dataset(path, target=target, overrides=_parse_overrides(overrides))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/prepare")
def prepare(file: UploadFile = File(...), target: str | None = Form(None), overrides: str | None = Form(None),
            level: str = Form("auto"), rows: int = Form(200000)):
    if rows < 100 or rows > 2_000_000:
        raise HTTPException(status_code=400, detail="rows must be between 100 and 2,000,000 for the synchronous pilot endpoint")
    path = _save_upload(file)
    try:
        payload, manifest = prepare_dataset_archive(
            path, target=target, overrides=_parse_overrides(overrides), level=level, rows=rows
        )
        import io
        headers = {
            "Content-Disposition": "attachment; filename=autodata_prepared.zip",
            "X-AutoData-Level": str(manifest["level"]),
        }
        return StreamingResponse(io.BytesIO(payload), media_type="application/zip", headers=headers)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/validate")
def validate(file: UploadFile = File(...), target: str | None = Form(None), overrides: str | None = Form(None)):
    path = _save_upload(file)
    try:
        return validate_dataset(path, target=target, overrides=_parse_overrides(overrides))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)
