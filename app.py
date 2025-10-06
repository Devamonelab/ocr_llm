#!/usr/bin/env python3
import os
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pipeline import run_pipeline

APP_VERSION = "1.0.0"

app = FastAPI(
    title="Doc Extract API",
    version=APP_VERSION,
    description="OCR + LLM document extraction over S3 (Textract + OpenAI).",
)

class ExtractRequest(BaseModel):
    bucket: str = Field(..., description="S3 bucket name")
    key: str = Field(..., description="S3 object key")
    mode: Literal["ocr+llm", "llm"] = "ocr+llm"

class ExtractResponse(BaseModel):
    doc_type: str
    document_name: Optional[str] = None
    result_path: str
    mode: str
    metrics_elapsed_sec: Optional[int] = None
    structured: dict

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "aws_region": os.getenv("AWS_REGION", "unset"),
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "version": APP_VERSION,
    }

@app.get("/version")
def version():
    return {"version": APP_VERSION}

@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    try:
        out = run_pipeline(req.bucket, req.key, req.mode)
        structured = out.get("structured", {})
        return {
            "doc_type": out.get("doc_type"),
            "document_name": structured.get("document_name"),
            "result_path": out.get("result_path"),
            "mode": out.get("mode"),
            "metrics_elapsed_sec": (structured.get("_metrics") or {}).get("elapsed_total_sec"),
            "structured": structured,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e.__class__.__name__}")
