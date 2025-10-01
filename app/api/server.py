from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any

try:
    from ocr_test import run_pipeline
except Exception as e:
    run_pipeline = None  # type: ignore
    _import_error = e


app = FastAPI(title="LendingWise OCR+LLM API", version="1.0.0")


class ExtractRequest(BaseModel):
    bucket: str = Field(..., description="S3 bucket name")
    key: str = Field(..., description="S3 object key (path to file)")
    mode: Literal["ocr+llm", "llm"] = Field("ocr+llm", description="Extraction mode")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/extract")
def extract(req: ExtractRequest) -> Dict[str, Any]:
    if run_pipeline is None:
        raise HTTPException(status_code=500, detail=f"Pipeline import failed: {_import_error}")
    try:
        result = run_pipeline(req.bucket, req.key, req.mode)
        return {
            "doc_type": result.get("doc_type"),
            "mode": result.get("mode"),
            "result_path": result.get("result_path"),
            "name_no_ext": result.get("name_no_ext"),
            "structured": result.get("structured"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.server:app", host="0.0.0.0", port=8000, reload=True)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any

# Import the pipeline from the script
try:
    from ocr_test import run_pipeline
except Exception as e:
    run_pipeline = None  # type: ignore
    _import_error = e


app = FastAPI(title="LendingWise OCR+LLM API", version="1.0.0")


class ExtractRequest(BaseModel):
    bucket: str = Field(..., description="S3 bucket name")
    key: str = Field(..., description="S3 object key (path to file)")
    mode: Literal["ocr+llm", "llm"] = Field("ocr+llm", description="Extraction mode")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/extract")
def extract(req: ExtractRequest) -> Dict[str, Any]:
    if run_pipeline is None:
        raise HTTPException(status_code=500, detail=f"Pipeline import failed: {_import_error}")
    try:
        result = run_pipeline(req.bucket, req.key, req.mode)
        # Return the structured result and where it was saved
        return {
            "doc_type": result.get("doc_type"),
            "mode": result.get("mode"),
            "result_path": result.get("result_path"),
            "name_no_ext": result.get("name_no_ext"),
            "structured": result.get("structured"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.server:app", host="0.0.0.0", port=8000, reload=True)

