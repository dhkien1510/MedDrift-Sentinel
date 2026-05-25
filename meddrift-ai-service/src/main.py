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
import yaml
import logging
import traceback
from datetime import datetime
from fastapi import BackgroundTasks
from bson import ObjectId
from minio.error import S3Error

_logger = logging.getLogger(__name__)

# Thêm src/ vào sys.path để `from db.xxx import` hoạt động đúng khi
# Docker chạy `uvicorn src.main:app` từ WORKDIR /app.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Thêm monitoring/ vào sys.path để import được các module con
# (vì chúng dùng bare import như: from registry import load_config)
sys.path.insert(0, os.path.join(_SRC_DIR, "monitoring"))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import RedirectResponse
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
from registry import list_available_algorithms, load_config, load_config_meta
from drift_agent import _image_embedding_buffer, _text_embedding_buffer
import numpy as np
try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

try:
    from multimodal_fusion import load_bundle, transform_joint
    _HAS_MULTIMODAL_FUSION = True
except ImportError:
    _HAS_MULTIMODAL_FUSION = False

# Import kết nối MongoDB từ thư mục db
from db.mongo_client import drift_reports, init_db
from db.redis_client import get_cached_status, cache_drift_status, invalidate_status, invalidate_config
from db.minio_client import init_buckets, upload_image, get_image_url

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

# Chạy init_db khi server khởi động để tạo index cho MongoDB
@app.on_event("startup")
async def startup():
    await init_db()
    # Khởi tạo MinIO buckets (tạo nếu chưa có)
    try:
        init_buckets()
    except Exception as exc:
        # Không crash server nếu MinIO chưa sẵn sàng (graceful degradation)
        import logging
        logging.getLogger(__name__).warning("⚠️  MinIO init failed (non-fatal): %s", exc)

# ============================================================
# IN-MEMORY REPORT STORAGE
# Reports được lưu trong list Python — tồn tại đến khi server restart.
# Mỗi report có id, timestamp, severity, alert, chart data, agent_analysis.
# ============================================================

import time
import asyncio
from fastapi import BackgroundTasks
# ============================================================
# HELPER: Multimodal PCA scatter — dùng chung cho cả 2 visualization endpoints
# ============================================================

def _build_multimodal_scatter_2d(
    img_batch: np.ndarray,
    txt_batch: np.ndarray,
    is_scenario: bool,
) -> list:
    """
    Project img_batch + txt_batch vào không gian multimodal chung rồi giảm
    xuống 2D bằng PCA để vẽ scatter plot.

    Pipeline:
      1. Load PCA bundle (pca_img, pca_txt) đã fit từ build_multimodal_reference.py
         → dùng `transform_joint` để chiếu cả batch hiện tại lẫn reference sang
           không gian joint: concat(pca_img.transform(X), pca_txt.transform(X))
      2. Load joint_reference_npy (điểm reference đã được project)
      3. Stack joint_ref + joint_current → fit PCA 2D → giảm về 2 chiều
      4. Trả về list[{x, y, type}] với type = "Reference" hoặc "Simulation"/"Current"

    Nếu PCA bundle chưa được build (multimodal.enabled = false hoặc file không tồn tại)
    → trả về [] để frontend bỏ qua tab multimodal một cách graceful.
    """
    if PCA is None:
        return []
    if not _HAS_MULTIMODAL_FUSION:
        _logger.warning("multimodal_fusion module not available — skipping multimodal PCA")
        return []

    # 1. Đọc paths từ drift_config.yaml
    try:
        config_path = "/configs/drift_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:
        _logger.error("_build_multimodal_scatter_2d: cannot read drift_config.yaml — %s", exc)
        return []

    mm = cfg.get("multimodal") or {}
    if not mm:
        _logger.warning("_build_multimodal_scatter_2d: no `multimodal` section in config")
        return []

    pca_pickle_path   = mm.get("pca_pickle")
    joint_ref_path    = mm.get("joint_reference_npy")

    if not pca_pickle_path or not joint_ref_path:
        _logger.warning("_build_multimodal_scatter_2d: pca_pickle or joint_reference_npy not set")
        return []

    # Resolve absolute path relative to the repo root
    import pathlib
    ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent)
    pca_abs  = pca_pickle_path  if os.path.isabs(pca_pickle_path)  else os.path.join(ROOT_DIR, pca_pickle_path)
    ref_abs  = joint_ref_path   if os.path.isabs(joint_ref_path)   else os.path.join(ROOT_DIR, joint_ref_path)

    if not os.path.isfile(pca_abs):
        _logger.warning("_build_multimodal_scatter_2d: PCA bundle not found at %s", pca_abs)
        return []
    if not os.path.isfile(ref_abs):
        _logger.warning("_build_multimodal_scatter_2d: joint reference not found at %s", ref_abs)
        return []

    try:
        # 2. Load bundle + joint reference
        bundle    = load_bundle(pca_abs)       # {"pca_image": PCA, "pca_text": PCA}
        joint_ref = np.load(ref_abs)           # shape (N_ref, n_components*2)

        img_arr = np.asarray(img_batch)
        txt_arr = np.asarray(txt_batch)

        # 3. Transform batch hiện tại vào joint space (dùng bundle đã fit)
        joint_current = transform_joint(img_arr, txt_arr, bundle)  # (N, n_components*2)

        # 4. Stack ref + current → fit PCA 2D → project
        step     = max(1, len(joint_ref) // 200)   # subsample ref để frontend không lag
        ref_sub  = joint_ref[::step]
        combined = np.vstack([ref_sub, joint_current])

        pca2d   = PCA(n_components=2)
        reduced = pca2d.fit_transform(combined)

        ref_2d  = reduced[:len(ref_sub)]
        curr_2d = reduced[len(ref_sub):]

        # 5. Build output list
        label = "Simulation" if is_scenario else "Current"
        data  = []
        for pt in ref_2d:
            data.append({"x": float(pt[0]), "y": float(pt[1]), "type": "Reference"})
        for pt in curr_2d:
            data.append({"x": float(pt[0]), "y": float(pt[1]), "type": label})

        return data

    except Exception as exc:
        _logger.error("_build_multimodal_scatter_2d failed: %s", exc, exc_info=True)
        return []


# ============================================================
# HELPER: Multimodal live summary cho từng sample đơn lẻ
# Dùng trong /api/drift/collect để trả feedback ngay cho ChatPage
# ============================================================

def _get_multimodal_live_summary(img_vec: np.ndarray, txt_vec: np.ndarray) -> dict:
    """
    Nhận 1 cặp embedding (img_vec shape (D,), txt_vec shape (D,)) →
    chiếu vào joint space bằng PCA bundle đã fit → tính các chỉ số so với reference.

    Trả về dict sẵn để JSON:
      enabled, ready, message,
      joint_dim, joint_l2_norm,
      distance_to_reference_centroid_l2,
      reference_typical_distance_median,
      distance_ratio_vs_typical,
      interpretation_hint, note,
      joint_projection_preview  (8 thành phần đầu)
    """
    base = {"enabled": False, "ready": False, "message": ""}

    if not _HAS_MULTIMODAL_FUSION:
        base["message"] = "multimodal_fusion module không tồn tại."
        return base

    try:
        config_path = "/configs/drift_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:
        base["message"] = f"Không đọc được drift_config.yaml: {exc}"
        return base

    mm = cfg.get("multimodal") or {}
    if not mm.get("enabled", False):
        base["enabled"] = False
        base["message"] = "multimodal.enabled = false trong drift_config.yaml."
        return base

    base["enabled"] = True
    pca_pickle_path = mm.get("pca_pickle")
    joint_ref_path  = mm.get("joint_reference_npy")

    ROOT_DIR = "/"
    pca_abs = pca_pickle_path if os.path.isabs(pca_pickle_path) else os.path.join(ROOT_DIR, pca_pickle_path)
    ref_abs = joint_ref_path  if os.path.isabs(joint_ref_path)  else os.path.join(ROOT_DIR, joint_ref_path)

    if not os.path.isfile(pca_abs) or not os.path.isfile(ref_abs):
        base["message"] = "Chưa có PCA bundle hoặc joint reference — chạy build_multimodal_reference.py trước."
        return base

    try:
        bundle    = load_bundle(pca_abs)
        joint_ref = np.load(ref_abs)                       # (N_ref, joint_dim)

        # Reshape single sample → (1, D) để transform_joint xử lý được
        img_2d = np.asarray(img_vec).reshape(1, -1)
        txt_2d = np.asarray(txt_vec).reshape(1, -1)
        joint_vec = transform_joint(img_2d, txt_2d, bundle)  # (1, joint_dim)
        joint_vec = joint_vec[0]                              # (joint_dim,)

        # Tính centroid của reference
        ref_centroid = joint_ref.mean(axis=0)

        # Khoảng cách sample hiện tại tới centroid
        dist_to_centroid = float(np.linalg.norm(joint_vec - ref_centroid))

        # Phân phối khoảng cách "điển hình" trong reference (mỗi ref point tới centroid)
        ref_dists = np.linalg.norm(joint_ref - ref_centroid, axis=1)
        typical_median = float(np.median(ref_dists))

        ratio = dist_to_centroid / (typical_median + 1e-9)

        if ratio < 1.2:
            hint = "✅ Trong vùng phân phối tham chiếu — dữ liệu bình thường."
        elif ratio < 2.0:
            hint = "⚠️ Hơi lệch so với tham chiếu — cần theo dõi."
        else:
            hint = "🔴 Xa vùng tham chiếu — có dấu hiệu drift đa phương thức."

        return {
            "enabled":   True,
            "ready":     True,
            "message":   "",
            "joint_dim": int(joint_vec.shape[0]),
            "joint_l2_norm": float(np.linalg.norm(joint_vec)),
            "distance_to_reference_centroid_l2":  dist_to_centroid,
            "reference_typical_distance_median":  typical_median,
            "distance_ratio_vs_typical":          float(ratio),
            "interpretation_hint": hint,
            "note": "Đây là ước lượng per-sample, p-value chính thức chỉ tính sau khi flush buffer.",
            "joint_projection_preview": [float(x) for x in joint_vec[:8]],
        }

    except Exception as exc:
        _logger.error("_get_multimodal_live_summary failed: %s", exc, exc_info=True)
        base["message"] = f"Lỗi khi tính multimodal summary: {exc}"
        return base


# ============================================================
# ENDPOINT: Simulate scenario
# ===========================================================
@app.post("/api/drift/simulate_scenario", summary="Run a drift scenario automatically")
async def simulate_scenario(scenario_name: str, background_tasks: BackgroundTasks):
    """
    scenario_name có thể là: 'level1_mild', 'level2_moderate', 'no_drift', v.v.
    """
    async def run_simulation():
        try:
            text_scenario = scenario_name
            if scenario_name == "level2_moderate":
                text_scenario = "level2_moderate_cross"
            elif scenario_name == "level3_severe":
                text_scenario = "level3_severe_offtopic"

            # Resolve encoder subdir từ config hiện tại
            cfg_image = load_config_meta(isImage=True)
            cfg_text  = load_config_meta(isImage=False)
            safe_img_enc = cfg_image["encoder"].replace("/", "--")
            safe_txt_enc = cfg_text["encoder"].replace("/", "--")

            base_data_dir = str(pathlib.Path(__file__).resolve().parent.parent.parent / "data")
            img_path = f"{base_data_dir}/drift_scenarios/image/{safe_img_enc}/{scenario_name}.npy"
            txt_path = f"{base_data_dir}/drift_scenarios/text/{safe_txt_enc}/{text_scenario}.npy"

            img_embs = np.load(img_path)
            txt_embs = np.load(txt_path)
        except FileNotFoundError as e:
            print(f"❌ Scenario file not found: {e}")
            return
            
        total_samples = len(img_embs)
        
        # Load config chuẩn
        config_path = "/configs/drift_config.yaml"
        current_threshold = 100
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                current_yaml = yaml.safe_load(f)
                current_threshold = current_yaml.get("buffer", {}).get("threshold", 100)
        except Exception:
            pass
            
        for i in range(0, total_samples, current_threshold):
            img_batch = img_embs[i : i + current_threshold]
            txt_batch = txt_embs[i : i + current_threshold]
            
            if len(img_batch) == current_threshold:
                print(f"Running simulation batch: {i} to {i+current_threshold}")
                
                # Hàm check AI chạy đồng bộ
                raw_report = run_drift_sentinel_batch(img_batch, txt_batch)
                
                report = {
                    "timestamp": datetime.now().isoformat(),
                    "scenario": scenario_name,
                    "simulated": True,
                    **raw_report
                }
                
                # Do hàm run_simulation đã thành async -> Dùng await lấy DB cực chuẩn
                await drift_reports.insert_one(report)
                try:
                    await invalidate_status()
                except Exception:
                    pass
                
                await asyncio.sleep(2) # Đợi 2s (Không dùng time.sleep làm nghẽn cổ chai)

    background_tasks.add_task(run_simulation)
    return {"message": f"Simulation for scenario {scenario_name} started!"}

# ============================================================
# ENDPOINT: Health Check
# ============================================================
@app.get("/health", summary="Health check")
async def health_check():
    return {"status": "healthy", "service": "meddrift-ai-service"}

# ============================================================
# ENDPOINT: List supported algorithms
# ============================================================
@app.get("/api/drift/algorithms")
async def get_algorithms():
    config_image = load_config_meta(isImage=True)
    config_text  = load_config_meta(isImage=False)
    return {
        "image_algorithms": config_image["allow_method"],
        "text_algorithms":  config_text["allow_method"],
        # Thêm 2 dòng này — filter None do YAML comment
        "image_encoders":    config_image["allow_encoder"],
        "text_encoders":   config_text["allow_encoder"],
    }

# ============================================================
# ENDPOINT: Visualization (PCA Scatter)
# ============================================================
@app.get("/api/drift/visualization", summary="Get 2D PCA projection of reference vs current data")
async def get_drift_visualization(scenario: str = None):
    if PCA is None:
        raise HTTPException(status_code=500, detail="sklearn.decomposition.PCA not available")
    
    def extract_pca(config_is_image, data_list):
        config = load_config(isImage=config_is_image)
        ref_emb = config['ref_embeddings']
        
        if data_list is None or len(data_list) == 0:
            return []
            
        current_emb = np.array(data_list)
        combined_emb = np.vstack([ref_emb, current_emb])
        
        pca = PCA(n_components=2)
        reduced_2d = pca.fit_transform(combined_emb)
        
        ref_2d = reduced_2d[:len(ref_emb)]
        curr_2d = reduced_2d[len(ref_emb):]
        
        data = []
        step = max(1, len(ref_2d) // 200)
        for point in ref_2d[::step]:
            data.append({"x": float(point[0]), "y": float(point[1]), "type": "Reference"})
        for point in curr_2d:
            data.append({"x": float(point[0]), "y": float(point[1]), "type": "Simulation" if scenario else "Current"})
        return data

    if scenario and scenario != "none":
        try:
            text_scenario = scenario
            if scenario == "level2_moderate":
                text_scenario = "level2_moderate_cross"
            elif scenario == "level3_severe":
                text_scenario = "level3_severe_offtopic"

            cfg_image = load_config_meta(isImage=True)
            cfg_text  = load_config_meta(isImage=False)
            safe_img_enc = cfg_image["encoder"].replace("/", "--")
            safe_txt_enc = cfg_text["encoder"].replace("/", "--")

            base_data_dir = str(pathlib.Path(__file__).resolve().parent.parent.parent / "data")
            img_embs = np.load(f"{base_data_dir}/drift_scenarios/image/{safe_img_enc}/{scenario}.npy")[:200]
            txt_embs = np.load(f"{base_data_dir}/drift_scenarios/text/{safe_txt_enc}/{text_scenario}.npy")[:200]
            image_data = extract_pca(True, img_embs)
            text_data  = extract_pca(False, txt_embs)
            multimodal_data = _build_multimodal_scatter_2d(img_embs, txt_embs, True)
        except Exception as e:
            _logger.error(f"Cannot load scenario for PCA: {e}")
            image_data = []
            text_data = []
    else:
        image_data = extract_pca(True, _image_embedding_buffer)
        text_data = extract_pca(False, _text_embedding_buffer)
        multimodal_data = _build_multimodal_scatter_2d(
            np.array(_image_embedding_buffer) if _image_embedding_buffer else np.empty((0,)),
            np.array(_text_embedding_buffer)  if _text_embedding_buffer  else np.empty((0,)),
            is_scenario=False,
        )
        
    return {
        "image_data": image_data,
        "text_data": text_data,
        "multimodal_data": multimodal_data,
    }



# ============================================================
# ENDPOINT: Visualization theo report_id cụ thể
# ============================================================
@app.get(
    "/api/drift/visualization/by_report/{report_id}",
    summary="Get 2D PCA projection for a specific drift report",
)
async def get_visualization_by_report(report_id: str):
    """
    Nhận report_id (MongoDB ObjectId dạng string) → tìm report → trả về PCA data.
    - Nếu report là simulated: load embedding theo scenario từ MinIO.
    - Nếu report là live buffer: trả về mảng rỗng (embedding đã bị flush).
    """
    if PCA is None:
        raise HTTPException(status_code=500, detail="sklearn.decomposition.PCA not available")
 
    # 1. Tìm report trong MongoDB
    try:
        oid = ObjectId(report_id)
    except Exception:
        raise HTTPException(status_code=400, detail=f"report_id không hợp lệ: {report_id}")
 
    report = await drift_reports.find_one({"_id": oid})
    if report is None:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy report: {report_id}")
 
    # 2. Kiểm tra loại report
    is_simulated = bool(report.get("simulated", False))
    scenario = report.get("scenario") if is_simulated else None
 
    image_data, text_data, multimodal_data = [], [], []
 
    def extract_pca(config_is_image, data_list):
        config = load_config(isImage=config_is_image)
        if config is None or config.get("ref_embeddings") is None:
            return []
        ref_emb = config["ref_embeddings"]
        if data_list is None or len(data_list) == 0:
            return []
        current_emb = np.array(data_list)
        combined_emb = np.vstack([ref_emb, current_emb])
        pca = PCA(n_components=2)
        reduced_2d = pca.fit_transform(combined_emb)
        ref_2d = reduced_2d[:len(ref_emb)]
        curr_2d = reduced_2d[len(ref_emb):]
        data = []
        step = max(1, len(ref_2d) // 200)
        for point in ref_2d[::step]:
            data.append({"x": float(point[0]), "y": float(point[1]), "type": "Reference"})
        label = "Simulation" if is_simulated else "Current"
        for point in curr_2d:
            data.append({"x": float(point[0]), "y": float(point[1]), "type": label})
        return data
 
    # 3. Xử lý theo loại report
    if is_simulated and scenario and scenario != "none":
        try:
            text_scenario = scenario
            if scenario == "level2_moderate":
                text_scenario = "level2_moderate_cross"
            elif scenario == "level3_severe":
                text_scenario = "level3_severe_offtopic"
 
            cfg_image = load_config_meta(isImage=True)
            cfg_text  = load_config_meta(isImage=False)
 
            img_obj = scenario_object_name(True, cfg_image["encoder"], scenario)
            txt_obj = scenario_object_name(False, cfg_text["encoder"], text_scenario)
 
            img_embs = download_npy(img_obj)[:200]
            txt_embs = download_npy(txt_obj)[:200]
 
            image_data = extract_pca(True, img_embs)
            text_data  = extract_pca(False, txt_embs)
            multimodal_data = _build_multimodal_scatter_2d(img_embs, txt_embs, True)
        except Exception as e:
            _logger.error(f"❌ PCA by_report (simulated) Error: {e}")
    else:
        # Live report: buffer đã được flush, không còn dữ liệu raw embedding
        # Trả về mảng rỗng — frontend sẽ hiển thị thông báo phù hợp
        image_data, text_data, multimodal_data = [], [], []
 
    return {
        "report_id": report_id,
        "scenario": scenario,
        "simulated": is_simulated,
        "image_data": image_data,
        "text_data": text_data,
        "multimodal_data": multimodal_data,
    } 
# ============================================================
# ENDPOINT: Get buffer status + latest report
# Frontend poll endpoint này để hiển thị alert khi drift xảy ra.
# ============================================================
@app.get("/api/drift/status", summary="Get buffer count and latest drift report")
async def get_drift_status():
    # 1. Kiểm tra Cache Redis trước
    cached_status = await get_cached_status()
    if cached_status:
        return cached_status

    # 2. Nếu không có cache, chạy logic tìm trong MongoDB
    latest_report = await drift_reports.find_one(sort=[("timestamp", -1)])
    
    # MongoDB dùng _id kiểu ObjectId, cần chuyển sang string để gửi cho Frontend (JSON)
    if latest_report:
        latest_report["_id"] = str(latest_report["_id"])
        
    alert = latest_report.get("alert", False) if latest_report else False
    total_reports = await drift_reports.count_documents({})

    response_data = {
        "buffer_count": get_buffer_count(),
        "buffer_threshold": BUFFER_THRESHOLD,
        "total_reports": total_reports,
        "alert": alert,
        "latest_report": latest_report,
    }
    
    # 3. Lưu vào Cache 10 giây để lần sau Frontend gọi sẽ tải rất nhanh
    await cache_drift_status(response_data, ttl=10)

    return response_data


# ============================================================
# ENDPOINT: Get all reports (history)
# Frontend dùng để hiển thị lịch sử drift detection.
# ============================================================
@app.get("/api/drift/reports", summary="Get all drift detection reports")
async def get_drift_reports():
    # Lấy tối đa 100 báo cáo mới nhất từ MongoDB
    cursor = drift_reports.find().sort("timestamp", -1).limit(100)
    reports = await cursor.to_list(length=100)
    
    # Chuyển đổi _id sang string
    for r in reports:
        r["_id"] = str(r["_id"])
        
    total_reports = await drift_reports.count_documents({})
    return {
        "total_reports": total_reports,
        "reports": reports,
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
    try:
        # 1. Convert uploaded bytes → PIL Image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 2. Upload ảnh gốc lên MinIO (lưu key để ghi vào MongoDB report)
        # NOTE: Catch Exception rộng hơn S3Error để xử lý cả ConnectionRefused khi MinIO chưa ready
        image_object_key = None
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = image.filename.replace(" ", "_") if image.filename else "image.jpg"
            image_object_key = f"uploads/{timestamp_str}_{safe_filename}"
            upload_image(
                file_bytes=contents,
                object_name=image_object_key,
                content_type=image.content_type or "image/jpeg",
            )
        except Exception as minio_err:
            # MinIO lỗi (kết nối, bucket chưa tạo, ...) → vẫn tiếp tục drift
            _logger.warning("⚠️  MinIO upload failed (non-fatal): %s", minio_err)
            image_object_key = None

        # 3. Extract embeddings
        img_embedding = extract_image_embedding(pil_image)        # shape (512,)
        txt_embedding = extract_text_embedding([question])        # shape (1, 768)

        # 4. Tính multimodal live summary cho sample này (non-fatal)
        txt_squeezed = txt_embedding.squeeze()
        try:
            mm_summary = _get_multimodal_live_summary(img_embedding, txt_squeezed)
        except Exception:
            mm_summary = {"enabled": False, "ready": False, "message": "Lỗi nội bộ khi tính multimodal summary."}

        # 5. Add to buffer
        add_to_buffer(img_embedding, txt_squeezed)

        drift_triggered = False
        report = None

        # 5. Auto-trigger khi buffer đủ threshold
        if is_buffer_ready():
            drift_triggered = True
            img_batch, txt_batch = flush_buffer()

            # Chạy Drift Agent trên batch embedding
            raw_report = run_drift_sentinel_batch(img_batch, txt_batch)

            # Lưu report (chỉ thêm timestamp, MongoDB tự động sinh _id)
            report = {
                "timestamp": datetime.now().isoformat(),
                "image_path": image_object_key,   # MinIO object key (None nếu upload lỗi)
                **raw_report,
            }
            # Thêm vào collection drift_reports của MongoDB
            await drift_reports.insert_one(report)

        # 6. Xóa Cache Redis (non-fatal nếu Redis chưa ready)
        try:
            await invalidate_status()
        except Exception as redis_err:
            _logger.warning("⚠️  Redis invalidate failed (non-fatal): %s", redis_err)

        return {
            "status": "drift_check_triggered" if drift_triggered else "buffered",
            "buffer_count": get_buffer_count(),
            "buffer_threshold": BUFFER_THRESHOLD,
            "drift_triggered": drift_triggered,
            "alert": report.get("alert", False) if report else False,
            "report_id": str(report["_id"]) if report else None,
            "multimodal": mm_summary,
        }

    except HTTPException:
        raise  # re-raise FastAPI errors as-is
    except Exception as e:
        # Log đầy đủ traceback để debug trong docker logs
        _logger.error("❌ /api/drift/collect failed:\n%s", traceback.format_exc())
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
            "timestamp": datetime.now().isoformat(),
            **raw_report,
        }
        # Thêm vào MongoDB
        await drift_reports.insert_one(report)
        
        # Chuyển ObjectId sang string cho JSON
        report["_id"] = str(report["_id"])

        # Xóa Cache status
        await invalidate_status()

        return {
            "status": "success",
            "samples_analyzed": count,
            "report": report,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# ENDPOINT: Get image — Trả về presigned URL để xem ảnh
# ============================================================
@app.get("/api/images/{image_key:path}", summary="Get presigned URL for a stored medical image")
async def get_image(
    image_key: str,
    expires_hours: int = 1,
):
    """
    Nhận image_key (MinIO object name) → trả về presigned URL có TTL.
    Frontend dùng URL này để hiển thị ảnh (<img src=...>) mà không cần
    expose MinIO credentials.

    Args:
        image_key     : MinIO object key, ví dụ "uploads/20260513_075000_chest.jpg".
        expires_hours : Số giờ URL còn hiệu lực (mặc định 1).
    """
    try:
        presigned_url = get_image_url(image_key, expires_hours=expires_hours)
        # Redirect trực tiếp đến MinIO (307 Temporary Redirect)
        return RedirectResponse(url=presigned_url, status_code=307)
    except S3Error as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy ảnh '{image_key}' trong MinIO: {exc}",
        )


# Thêm endpoint để lấy cấu hình hiện tại
@app.get("/api/drift/config")
async def get_config():
    config_image = load_config_meta(isImage=True)
    config_text = load_config_meta(isImage=False)
    
    # Load raw yaml to get buffer since it's not in meta yet
    config_path = "/configs/drift_config.yaml"
    buffer_threshold = 100
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            current = yaml.safe_load(f)
            buffer_threshold = current.get("buffer", {}).get("threshold", 100)
    except Exception:
        pass

    return {
        "buffer_threshold": buffer_threshold,
        "image_encoder": config_image["encoder"],
        "image_algorithm": config_image["algorithm"],
        "image_p_threshold": config_image["p_threshold"],
        
        "text_encoder": config_text["encoder"],
        "text_algorithm": config_text["algorithm"],
        "text_p_threshold": config_text["p_threshold"],
    }

@app.post("/api/drift/config/apply")
async def apply_config(config: dict):
    try:
        # Đọc file yaml hiện tại
        config_path = "/configs/drift_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            current = yaml.safe_load(f)

        # Cập nhật các field từ frontend
        if "buffer" not in current:
            current["buffer"] = {}
        current["buffer"]["threshold"] = config.get("buffer_threshold", current["buffer"].get("threshold", 100))

        current["image"]["encoder"]     = config.get("image_encoder",     current["image"]["encoder"])
        current["image"]["algorithm"]   = config.get("image_algorithm",   current["image"]["algorithm"])
        current["image"]["p_threshold"] = config.get("image_p_threshold", current["image"]["p_threshold"])

        current["text"]["encoder"]      = config.get("text_encoder",      current["text"]["encoder"])
        current["text"]["algorithm"]    = config.get("text_algorithm",    current["text"]["algorithm"])
        current["text"]["p_threshold"]  = config.get("text_p_threshold",  current["text"]["p_threshold"])

        # Ghi lại vào file yaml
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(current, f, allow_unicode=True)

        # Xóa cache config và status
        await invalidate_config()
        await invalidate_status()

        return {"status": "success", "updated_config": config}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


"""
PATCH cho main.py — 2 thay đổi:

1. simulate_scenario: lưu thêm batch_offset + batch_size vào mỗi report
   → giúp endpoint PCA biết chính xác slice nào của file .npy tương ứng với đợt đó.

2. Endpoint GET /api/drift/visualization/by_report/{report_id} (thay thế version cũ):
   → Dùng batch_offset + batch_size để slice ĐÚNG batch embedding từ MinIO.
   → Hỗ trợ multimodal PCA đầy đủ.
   → Live report (không có offset) → trả rỗng với flag rõ ràng.

CÁCH TÍCH HỢP:
─────────────────────────────────────────────────────────────────
A) Thêm import ở đầu main.py (nếu chưa có):
   from bson import ObjectId

B) Trong hàm simulate_scenario, thay đoạn tạo report (dòng ~171):

   # CŨ:
   report = {
       "timestamp": datetime.now().isoformat(),
       "scenario": scenario_name,
       "simulated": True,
       **raw_report
   }

   # MỚI: thêm batch_offset và batch_size
   report = {
       "timestamp": datetime.now().isoformat(),
       "scenario": scenario_name,
       "simulated": True,
       "batch_offset": i,           # <-- thêm dòng này
       "batch_size": current_threshold,  # <-- thêm dòng này
       **raw_report
   }

C) Thêm endpoint bên dưới vào main.py (thay thế endpoint by_report cũ nếu đã có).
─────────────────────────────────────────────────────────────────
"""

from bson import ObjectId  # thêm ở đầu main.py nếu chưa có


# ============================================================
# ENDPOINT: Visualization theo report_id cụ thể
# ============================================================
@app.get(
    "/api/drift/visualization/by_report/{report_id}",
    summary="Get 2D PCA projection for a specific drift report (correct batch slice)",
)
async def get_visualization_by_report(report_id: str):
    """
    Nhận report_id (MongoDB ObjectId) → tìm report → trả PCA data đúng batch.

    - Simulated report: đọc batch_offset + batch_size từ report,
      slice ĐÚNG phần embedding trên MinIO (không dùng toàn bộ file).
    - Live report: embedding đã flush ra khỏi RAM → trả rỗng.
    - Hỗ trợ image, text và multimodal PCA.
    """
    if PCA is None:
        raise HTTPException(status_code=500, detail="sklearn.decomposition.PCA not available")

    # 1. Tìm report trong MongoDB
    try:
        oid = ObjectId(report_id)
    except Exception:
        raise HTTPException(status_code=400, detail=f"report_id không hợp lệ: {report_id}")

    report = await drift_reports.find_one({"_id": oid})
    if report is None:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy report: {report_id}")

    is_simulated = bool(report.get("simulated", False))
    scenario     = report.get("scenario") if is_simulated else None
    # batch_offset + batch_size được lưu khi simulate (xem hướng dẫn phần B ở trên)
    batch_offset = int(report.get("batch_offset", 0))
    batch_size   = int(report.get("batch_size", 100))

    image_data, text_data, multimodal_data = [], [], []

    # ── helper: fit PCA trên ref + current batch, trả list dict {x,y,type} ──
    def extract_pca(config_is_image: bool, batch: np.ndarray, label: str) -> list:
        config = load_config(isImage=config_is_image)
        if config is None or config.get("ref_embeddings") is None:
            return []
        ref_emb = np.asarray(config["ref_embeddings"])
        if batch is None or len(batch) == 0:
            return []
        current_emb = np.asarray(batch)
        combined    = np.vstack([ref_emb, current_emb])
        pca2d       = PCA(n_components=2)
        reduced     = pca2d.fit_transform(combined)
        ref_2d  = reduced[:len(ref_emb)]
        curr_2d = reduced[len(ref_emb):]
        data = []
        step = max(1, len(ref_2d) // 200)          # subsample ref để frontend không lag
        for pt in ref_2d[::step]:
            data.append({"x": float(pt[0]), "y": float(pt[1]), "type": "Reference"})
        for pt in curr_2d:
            data.append({"x": float(pt[0]), "y": float(pt[1]), "type": label})
        return data

    # ── Simulated report: load từ MinIO, slice đúng batch ──
    if is_simulated and scenario and scenario != "none":
        try:
            # Mapping tên scenario cho text (giống simulate_scenario)
            text_scenario = scenario
            if scenario == "level2_moderate":
                text_scenario = "level2_moderate_cross"
            elif scenario == "level3_severe":
                text_scenario = "level3_severe_offtopic"

            cfg_image = load_config_meta(isImage=True)
            cfg_text  = load_config_meta(isImage=False)

            img_obj = scenario_object_name(True,  cfg_image["encoder"], scenario)
            txt_obj = scenario_object_name(False, cfg_text["encoder"],  text_scenario)

            # Tải toàn bộ file .npy rồi slice ĐÚNG batch tương ứng với report này
            all_img_embs = download_npy(img_obj)
            all_txt_embs = download_npy(txt_obj)

            img_batch = all_img_embs[batch_offset : batch_offset + batch_size]
            txt_batch = all_txt_embs[batch_offset : batch_offset + batch_size]

            if len(img_batch) == 0:
                # batch_offset vượt quá file → fallback lấy batch cuối cùng
                img_batch = all_img_embs[-batch_size:]
                txt_batch = all_txt_embs[-batch_size:]

            label = "Simulation"
            image_data      = extract_pca(True,  img_batch, label)
            text_data       = extract_pca(False, txt_batch, label)
            multimodal_data = _build_multimodal_scatter_2d(img_batch, txt_batch, is_scenario=True)

        except Exception as exc:
            _logger.error("❌ PCA by_report (simulated) Error: %s", exc)

    # ── Live report: embedding đã flush khỏi RAM, không thể tái tạo ──
    # image_data, text_data, multimodal_data đã là [] từ đầu → trả về rỗng

    return {
        "report_id":        report_id,
        "scenario":         scenario,
        "simulated":        is_simulated,
        "batch_offset":     batch_offset,
        "batch_size":       batch_size,
        "image_data":       image_data,
        "text_data":        text_data,
        "multimodal_data":  multimodal_data,
    }