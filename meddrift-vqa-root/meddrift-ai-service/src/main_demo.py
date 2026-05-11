import sys
import os
import io
from datetime import datetime
from typing import List, Optional

# Add monitoring directory to sys.path so we can import from it easily
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "monitoring"))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from image_drift import extract_image_embedding
from text_drift import extract_text_embedding
from drift_agent import (
    add_to_buffer, 
    is_buffer_ready, 
    get_buffer_count,
    flush_buffer,
    run_drift_sentinel_batch, 
    BUFFER_THRESHOLD,
)
from registry import list_available_algorithms

app = FastAPI(title="MedDrift Sentinel API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for drift reports
_drift_reports = []
_report_counter = 1

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "meddrift-ai-service"}

@app.get("/api/drift/algorithms")
async def get_algorithms():
    return {"algorithms": list_available_algorithms()}

@app.get("/api/drift/status")
async def get_drift_status():
    latest_report = _drift_reports[-1] if _drift_reports else None
    alert = latest_report.get("alert", False) if latest_report else False
    
    return {
        "buffer_count": get_buffer_count(),
        "buffer_threshold": BUFFER_THRESHOLD,
        "total_reports": len(_drift_reports),
        "alert": alert,
        "latest_report": latest_report
    }

@app.get("/api/drift/reports")
async def get_drift_reports():
    return {
        "total_reports": len(_drift_reports),
        "reports": _drift_reports
    }

@app.post("/api/drift/collect")
async def collect_drift_data(
    question: str = Form(...),
    image: UploadFile = File(...),
):
    """
    Receive image + question -> extract embedding -> add to buffer.
    If buffer reaches threshold, automatically flush and run drift detection.
    """
    global _report_counter
    
    try:
        # 1. Convert uploaded image to PIL Image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 2. Extract embeddings
        img_embedding = extract_image_embedding(pil_image)
        txt_embedding = extract_text_embedding([question])
        
        # 3. Add to buffer (txt_embedding is typically (1, D), so we squeeze it)
        add_to_buffer(img_embedding, txt_embedding.squeeze())
        
        drift_triggered = False
        report = None
        
        # 4. Check if buffer is full and trigger detection
        if is_buffer_ready():
            drift_triggered = True
            img_batch, txt_batch = flush_buffer()
            
            # Run the agent in batch mode
            raw_report = run_drift_sentinel_batch(img_batch, txt_batch)
            
            # Format and save report
            report = {
                "id": _report_counter,
                "timestamp": datetime.now().isoformat(),
                **raw_report  # Unpack the dict returned by run_drift_sentinel_batch
            }
            _drift_reports.append(report)
            _report_counter += 1
            
        return {
            "status": "drift_check_triggered" if drift_triggered else "buffered",
            "buffer_count": get_buffer_count(),
            "buffer_threshold": BUFFER_THRESHOLD,
            "drift_triggered": drift_triggered,
            "alert": report.get("alert", False) if report else False,
            "report_id": report["id"] if report else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/drift/analyze")
async def force_drift_analysis():
    """
    Manual trigger — force drift detection on the current buffer, 
    even if it hasn't reached the threshold.
    """
    global _report_counter
    
    count = get_buffer_count()
    if count == 0:
        raise HTTPException(status_code=400, detail="Buffer is empty. No data to analyze.")
        
    try:
        img_batch, txt_batch = flush_buffer()
        
        # Run the agent in batch mode
        raw_report = run_drift_sentinel_batch(img_batch, txt_batch)
        
        # Format and save report
        report = {
            "id": _report_counter,
            "timestamp": datetime.now().isoformat(),
            **raw_report
        }
        _drift_reports.append(report)
        _report_counter += 1
        
        return {
            "status": "success",
            "samples_analyzed": count,
            "report": report
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Start the server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
