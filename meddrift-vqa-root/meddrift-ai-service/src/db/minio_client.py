"""
db/minio_client.py — MinIO Object Storage Client
==================================================
Buckets:
  - medical-images   : ảnh gốc user                    [giữ nguyên]
  - reference-data   : ref .npy + scenario .npy         [PATCH] thêm scenario
  - drift-scenarios  : [deprecated alias — dùng reference-data/scenarios/]

Layout trong bucket reference-data:
  reference/image/<safe_encoder>/ref_images.npy
  reference/text/<safe_encoder>/ref_questions.npy
  scenarios/image/<safe_encoder>/<scenario_name>.npy
  scenarios/text/<safe_encoder>/<scenario_name>.npy

Lý do dùng MinIO cho .npy:
  - Local disk phụ thuộc Docker volume mount → dễ sai path, không quản lý được qua API.
  - MinIO cho phép upload/replace reference data mà không cần redeploy container.

Local cache:
  - Sau khi download từ MinIO, file được cache tại /tmp/meddrift_cache/<object_name>.
  - Cache hợp lệ cho đến khi bị xóa (bởi invalidate_local_cache) hoặc container restart.
  - Container restart → cache bị xóa → download lại từ MinIO lần đầu truy cập.
"""

import os
import io
import logging
from datetime import timedelta
from pathlib import Path

from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)

# ============================================================
# CẤU HÌNH KẾT NỐI
# ============================================================
MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "meddrift")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_SECURE     = os.getenv("MINIO_SECURE", "false").lower() == "true"

BUCKET_IMAGES    = "medical-images"
BUCKET_REFERENCE = "reference-data"   # chứa cả ref + scenario .npy

# Local cache directory (trong container, mất khi restart — đây là bình thường)
import tempfile
LOCAL_CACHE_DIR = Path(os.getenv("MEDDRIFT_CACHE_DIR", os.path.join(tempfile.gettempdir(), "meddrift_cache")))

# ============================================================
# KHỞI TẠO CLIENT
# ============================================================
minio_client = Minio(
    endpoint=MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE,
)


# ============================================================
# HELPER: Tạo bucket
# ============================================================
def _ensure_bucket(bucket_name: str) -> None:
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            logger.info("✅ MinIO: Đã tạo bucket '%s'", bucket_name)
        else:
            logger.info("ℹ️  MinIO: Bucket '%s' đã tồn tại.", bucket_name)
    except S3Error as exc:
        logger.error("❌ MinIO: Lỗi khi kiểm tra/tạo bucket '%s': %s", bucket_name, exc)
        raise


# ============================================================
# PUBLIC: INIT
# ============================================================
def init_buckets() -> None:
    """
    Khởi tạo tất cả buckets khi server bắt đầu.
    Gọi trong startup event của FastAPI.
    """
    _ensure_bucket(BUCKET_IMAGES)
    _ensure_bucket(BUCKET_REFERENCE)
    LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("✅ MinIO: Tất cả buckets đã được khởi tạo thành công.")


# ============================================================
# PUBLIC: MEDICAL IMAGES (không đổi)
# ============================================================
def upload_image(file_bytes: bytes, object_name: str, content_type: str = "image/jpeg") -> str:
    """Upload ảnh y tế lên bucket 'medical-images'."""
    minio_client.put_object(
        bucket_name=BUCKET_IMAGES,
        object_name=object_name,
        data=io.BytesIO(file_bytes),
        length=len(file_bytes),
        content_type=content_type,
    )
    logger.info("✅ MinIO: Đã upload '%s' lên bucket '%s'.", object_name, BUCKET_IMAGES)
    return object_name


def get_image_url(object_name: str, expires_hours: int = 1) -> str:
    """Tạo presigned URL có thời hạn để frontend hiển thị ảnh."""
    url = minio_client.presigned_get_object(
        bucket_name=BUCKET_IMAGES,
        object_name=object_name,
        expires=timedelta(hours=expires_hours),
    )
    return url


def delete_image(object_name: str) -> None:
    minio_client.remove_object(bucket_name=BUCKET_IMAGES, object_name=object_name)


# ============================================================
# PUBLIC: REFERENCE & SCENARIO .npy FILES
# ============================================================

def _safe_encoder(model_id: str) -> str:
    return model_id.replace("/", "--")


def _cache_path(object_name: str) -> Path:
    """Trả về đường dẫn local cache tương ứng với MinIO object_name."""
    safe = object_name.replace("/", "_")
    return LOCAL_CACHE_DIR / safe


def _download_bytes(object_name: str) -> bytes:
    """Download raw bytes từ bucket reference-data."""
    response = minio_client.get_object(BUCKET_REFERENCE, object_name)
    try:
        data = response.read()
    finally:
        response.close()
        response.release_conn()
    logger.info("📥 MinIO: Downloaded '%s' (%d bytes).", object_name, len(data))
    return data


def download_npy(object_name: str, use_cache: bool = True) -> "np.ndarray":
    """
    Download và load một file .npy từ MinIO reference-data bucket.
    Kết quả được cache tại LOCAL_CACHE_DIR để tránh download lại.

    Args:
        object_name : path trong bucket, ví dụ
                      "reference/image/microsoft--rad-dino/ref_images.npy"
        use_cache   : True (mặc định) — dùng local cache nếu có.

    Returns:
        np.ndarray

    Raises:
        S3Error nếu object không tồn tại trên MinIO.
        FileNotFoundError nếu cả MinIO lẫn cache đều không có.
    """
    import numpy as np

    cache = _cache_path(object_name)

    if use_cache and cache.exists():
        logger.info("💾 Cache hit: '%s'", object_name)
        return np.load(str(cache))

    data = _download_bytes(object_name)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(data)
    return np.load(io.BytesIO(data))


def upload_npy(arr: "np.ndarray", object_name: str) -> str:
    """
    Upload một numpy array (.npy) lên MinIO reference-data bucket.
    Dùng để upload reference embeddings hoặc scenario embeddings qua API.

    Args:
        arr         : numpy array cần upload
        object_name : path đích trong bucket

    Returns:
        object_name
    """
    import numpy as np

    buf = io.BytesIO()
    np.save(buf, arr)
    file_bytes = buf.getvalue()

    minio_client.put_object(
        bucket_name=BUCKET_REFERENCE,
        object_name=object_name,
        data=io.BytesIO(file_bytes),
        length=len(file_bytes),
        content_type="application/octet-stream",
    )
    # Invalidate local cache nếu có
    cache = _cache_path(object_name)
    if cache.exists():
        cache.unlink()
        logger.info("🗑️  Cache invalidated: '%s'", object_name)

    logger.info("✅ MinIO: Uploaded npy '%s'.", object_name)
    return object_name


def reference_object_name(is_image: bool, encoder: str, filename: str) -> str:
    """
    Tạo object name chuẩn cho reference embeddings.
    Ví dụ: "reference/image/microsoft--rad-dino/ref_images.npy"
    """
    sub = "image" if is_image else "text"
    return f"reference/{sub}/{_safe_encoder(encoder)}/{filename}"


def scenario_object_name(is_image: bool, encoder: str, scenario_name: str) -> str:
    """
    Tạo object name chuẩn cho scenario embeddings.
    Ví dụ: "scenarios/image/microsoft--rad-dino/level1_mild.npy"
    """
    sub = "image" if is_image else "text"
    return f"scenarios/{sub}/{_safe_encoder(encoder)}/{scenario_name}.npy"


def object_exists(object_name: str) -> bool:
    """Kiểm tra object có tồn tại trong bucket reference-data không."""
    try:
        minio_client.stat_object(BUCKET_REFERENCE, object_name)
        return True
    except S3Error:
        return False


def invalidate_local_cache(object_name: str) -> None:
    """Xóa local cache của một object (dùng khi upload reference mới)."""
    cache = _cache_path(object_name)
    if cache.exists():
        cache.unlink()
        logger.info("🗑️  Local cache cleared for '%s'.", object_name)


# ── Backward-compat helpers (giữ lại để không break code cũ) ──────────
def upload_reference_file(file_bytes: bytes, object_name: str) -> str:
    """[Deprecated] Dùng upload_npy() cho array hoặc gọi trực tiếp put_object."""
    minio_client.put_object(
        bucket_name=BUCKET_REFERENCE,
        object_name=object_name,
        data=io.BytesIO(file_bytes),
        length=len(file_bytes),
        content_type="application/octet-stream",
    )
    return object_name


def download_reference_file(object_name: str) -> bytes:
    """[Deprecated] Dùng download_npy() để lấy numpy array trực tiếp."""
    return _download_bytes(object_name)