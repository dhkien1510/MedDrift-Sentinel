"""
main.py — MedDrift Sentinel FastAPI Server
==========================================
REST API exposing the drift monitoring pipeline.

Architecture:
  - VQA service và Drift service tách biệt và chạy độc lập.
  - Mỗi VQA request gửi image + question đến endpoint /api/drift/collect.
  - API extract embedding → add vào buffer.
  - Khi buffer đủ 100 samples → tự động flush → chạy Drift Agent → lưu report.
  - Frontend poll GET /api/drift/status để kiểm tra alert và report mới nhất.

Endpoints:
  POST /api/drift/collect   → buffer embeddings, auto-trigger khi đủ 100
  POST /api/drift/analyze   → force trigger trên buffer hiện tại (dù chưa đủ 100)
  GET  /api/drift/status    → buffer count + latest report + alert flag
  GET  /api/drift/reports   → toàn bộ lịch sử reports
  GET  /api/drift/algorithms → danh sách thuật toán hỗ trợ
  GET  /health              → health check
"""

import sys
import os
import io
from datetime import datetime

# Thêm monitoring/ vào sys.path để import được các module con
# (vì chúng dùng bare import như: from registry import load_config)
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

# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(
    title="MedDrift Sentinel API",
    version="1.0.0",
    description="Drift monitoring service for the MedDrift VQA system."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Mở rộng cho mọi origin (tighten khi deploy production)
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# IN-MEMORY REPORT STORAGE
# Reports được lưu trong list Python — tồn tại đến khi server restart.
# Mỗi report có id, timestamp, severity, alert, chart data, agent_analysis.
# ============================================================
_drift_reports: list = []
_report_counter: int = 1


# ============================================================
# ENDPOINT: Health Check
# ============================================================
@app.get("/health", summary="Health check")
async def health_check():
    return {"status": "healthy", "service": "meddrift-ai-service"}


# ============================================================
# ENDPOINT: List supported algorithms
# ============================================================
@app.get("/api/drift/algorithms", summary="List available drift detection algorithms")
async def get_algorithms():
    return {"algorithms": list_available_algorithms()}


# ============================================================
# ENDPOINT: Get buffer status + latest report
# Frontend poll endpoint này để hiển thị alert khi drift xảy ra.
# ============================================================
@app.get("/api/drift/status", summary="Get buffer count and latest drift report")
async def get_drift_status():
    latest_report = _drift_reports[-1] if _drift_reports else None
    alert = latest_report.get("alert", False) if latest_report else False

    return {
        "buffer_count": get_buffer_count(),
        "buffer_threshold": BUFFER_THRESHOLD,
        "total_reports": len(_drift_reports),
        "alert": alert,
        "latest_report": latest_report,
    }


# ============================================================
# ENDPOINT: Get all reports (history)
# Frontend dùng để hiển thị lịch sử drift detection.
# ============================================================
@app.get("/api/drift/reports", summary="Get all drift detection reports")
async def get_drift_reports():
    return {
        "total_reports": len(_drift_reports),
        "reports": _drift_reports,
    }


# ============================================================
# ENDPOINT: Collect — Buffer embedding mỗi request
# Được gọi sau mỗi VQA request (song song, không block VQA).
# ============================================================
@app.post("/api/drift/collect", summary="Collect image+question embedding into buffer")
async def collect_drift_data(
    question: str = Form(..., description="Doctor's question"),
    image: UploadFile = File(..., description="Medical image file"),
):
    """
    Nhận image + question từ mỗi VQA request.
    1. Convert image → PIL → extract CLIP embedding
    2. Extract BioBERT embedding từ question
    3. Add cả hai vào buffer
    4. Nếu buffer đủ threshold → tự động flush → chạy Drift Agent → lưu report
    """
    global _report_counter

    try:
        # 1. Convert uploaded bytes → PIL Image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 2. Extract embeddings
        img_embedding = extract_image_embedding(pil_image)        # shape (512,)
        txt_embedding = extract_text_embedding([question])        # shape (1, 768)

        # 3. Add to buffer
        # txt_embedding là (1, 768) → squeeze về (768,) để stack dễ hơn
        add_to_buffer(img_embedding, txt_embedding.squeeze())

        drift_triggered = False
        report = None

        # 4. Auto-trigger khi buffer đủ threshold
        if is_buffer_ready():
            drift_triggered = True
            img_batch, txt_batch = flush_buffer()

            # Chạy Drift Agent trên batch embedding
            raw_report = run_drift_sentinel_batch(img_batch, txt_batch)

            # Lưu report có thêm id + timestamp
            report = {
                "id": _report_counter,
                "timestamp": datetime.now().isoformat(),
                **raw_report,
            }
            _drift_reports.append(report)
            _report_counter += 1

        return {
            "status": "drift_check_triggered" if drift_triggered else "buffered",
            "buffer_count": get_buffer_count(),
            "buffer_threshold": BUFFER_THRESHOLD,
            "drift_triggered": drift_triggered,
            "alert": report.get("alert", False) if report else False,
            "report_id": report["id"] if report else None,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ENDPOINT: Force analyze — trigger thủ công (dùng khi test)
# Chạy drift detection trên buffer hiện tại, dù chưa đủ threshold.
# ============================================================
@app.post("/api/drift/analyze", summary="Force drift detection on current buffer")
async def force_drift_analysis():
    """
    Manual trigger — chạy drift detection ngay trên buffer hiện tại
    dù chưa đủ BUFFER_THRESHOLD. Hữu ích khi test với ít samples.
    """
    global _report_counter

    count = get_buffer_count()
    if count == 0:
        raise HTTPException(
            status_code=400,
            detail="Buffer is empty. Send some images via /api/drift/collect first."
        )

    try:
        img_batch, txt_batch = flush_buffer()

        raw_report = run_drift_sentinel_batch(img_batch, txt_batch)

        report = {
            "id": _report_counter,
            "timestamp": datetime.now().isoformat(),
            **raw_report,
        }
        _drift_reports.append(report)
        _report_counter += 1

        return {
            "status": "success",
            "samples_analyzed": count,
            "report": report,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Cấu hình mặc định (In-memory)

# Thêm endpoint để lấy cấu hình hiện tại
@app.get("/api/drift/config")
async def get_config():
    return {
    "image_encoder": "clip-vit-b32",
    "text_encoder": "biobert",
    "drift_algorithm": "mmd",
    "p_value_threshold": 0.05
}


# Thêm endpoint để cập nhật cấu hình
@app.post("/api/drift/config/apply")
async def apply_config(config: dict):
    global current_config
    # Trong thực tế, bạn sẽ gọi hàm load_model() tương ứng từ registry tại đây
    current_config.update(config)
    return {"status": "success", "updated_config": current_config}
# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
