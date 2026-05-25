"""
db/redis_client.py — Redis Client
===================================
Quản lý tất cả tương tác với Redis:
  - Cache drift status (10s TTL)           [PATCH] giữ nguyên
  - Cache active config (no TTL)           [PATCH] giữ nguyên
  - Persistent embedding buffer            [PATCH MỚI] thay thế RAM list trong drift_agent.py
      buffer:image  → RPUSH / LRANGE / DEL
      buffer:text   → RPUSH / LRANGE / DEL
    Lý do: RAM list mất toàn bộ khi container restart/crash.
    Redis survive restart nếu bật AOF/RDB (khuyến nghị bật AOF).
"""

import redis.asyncio as aioredis
import redis as redis_sync   # sync client — dùng cho drift_agent (synchronous context)
import os
import json
import numpy as np
import io

# ============================================================
# CẤU HÌNH
# ============================================================
REDIS_HOST     = os.getenv("REDIS_HOST",     "redis")
REDIS_PORT     = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Keys
KEY_DRIFT_STATUS  = "drift_status"
KEY_DRIFT_CONFIG  = "drift_config"
KEY_BUFFER_IMAGE  = "buffer:image"
KEY_BUFFER_TEXT   = "buffer:text"

# ============================================================
# ASYNC CLIENT (dùng trong FastAPI endpoints)
# ============================================================
redis_pool = aioredis.ConnectionPool(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True,  # JSON keys/values là string
)
redis_client = aioredis.Redis(connection_pool=redis_pool)

# ============================================================
# SYNC CLIENT (dùng trong drift_agent.py — synchronous code)
# Buffer phải dùng binary (decode_responses=False) để lưu numpy bytes
# ============================================================
_sync_pool = redis_sync.ConnectionPool(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=False,  # binary — lưu numpy array bytes
)
redis_sync_client = redis_sync.Redis(connection_pool=_sync_pool)


# ============================================================
# HELPER: Numpy ↔ bytes
# ============================================================
def _np_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()

def _bytes_to_np(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data))


# ============================================================
# STATUS CACHE (async)
# ============================================================
async def cache_drift_status(data: dict, ttl: int = 10):
    """Lưu trạng thái drift vào Redis, tự hết hạn sau ttl giây."""
    await redis_client.setex(KEY_DRIFT_STATUS, ttl, json.dumps(data))

async def get_cached_status() -> dict | None:
    """Lấy status cache (None nếu đã hết hạn)."""
    cached = await redis_client.get(KEY_DRIFT_STATUS)
    return json.loads(cached) if cached else None

async def invalidate_status():
    """Xóa status cache ngay lập tức."""
    await redis_client.delete(KEY_DRIFT_STATUS)


# ============================================================
# CONFIG CACHE (async)
# ============================================================
async def cache_config(data: dict):
    """Lưu active config vào Redis (không TTL — chỉ mất khi invalidate)."""
    await redis_client.set(KEY_DRIFT_CONFIG, json.dumps(data))

async def get_cached_config() -> dict | None:
    """Lấy config cache (None nếu chưa có hoặc đã invalidate)."""
    cached = await redis_client.get(KEY_DRIFT_CONFIG)
    return json.loads(cached) if cached else None

async def invalidate_config():
    """Xóa config cache khi admin thay đổi cấu hình."""
    await redis_client.delete(KEY_DRIFT_CONFIG)


# ============================================================
# EMBEDDING BUFFER — SYNC (dùng trong drift_agent.py)
# ============================================================
# Mỗi embedding được serialize thành bytes (numpy .npy format)
# và RPUSH vào list Redis. Khi flush: LRANGE toàn bộ rồi DEL.

def buffer_push(image_embedding: np.ndarray, text_embedding: np.ndarray) -> int:
    """
    Thêm 1 cặp embedding vào Redis buffer (thread-safe, survive restart).

    Args:
        image_embedding : np.ndarray shape (D_img,)
        text_embedding  : np.ndarray shape (D_txt,)

    Returns:
        Số lượng image embeddings hiện tại trong buffer (dùng để check threshold).
    """
    pipe = redis_sync_client.pipeline()
    pipe.rpush(KEY_BUFFER_IMAGE, _np_to_bytes(image_embedding))
    pipe.rpush(KEY_BUFFER_TEXT,  _np_to_bytes(text_embedding))
    results = pipe.execute()
    return int(results[0])  # length sau push (image list)

def buffer_count() -> int:
    """Trả về số lượng embeddings hiện có trong buffer."""
    return redis_sync_client.llen(KEY_BUFFER_IMAGE)

def buffer_is_ready(threshold: int) -> bool:
    """True nếu buffer đã tích lũy đủ threshold mẫu."""
    return buffer_count() >= threshold

def buffer_flush() -> tuple[np.ndarray, np.ndarray]:
    """
    Lấy toàn bộ embeddings ra, xóa buffer, trả về (img_batch, txt_batch).

    Returns:
        (np.ndarray shape (N, D_img), np.ndarray shape (N, D_txt))

    Raises:
        ValueError nếu buffer rỗng.
    """
    pipe = redis_sync_client.pipeline()
    pipe.lrange(KEY_BUFFER_IMAGE, 0, -1)
    pipe.lrange(KEY_BUFFER_TEXT,  0, -1)
    pipe.delete(KEY_BUFFER_IMAGE)
    pipe.delete(KEY_BUFFER_TEXT)
    img_raw, txt_raw, *_ = pipe.execute()

    if not img_raw:
        raise ValueError("Buffer rỗng — không có gì để flush.")

    img_batch = np.stack([_bytes_to_np(b) for b in img_raw])
    txt_batch = np.stack([_bytes_to_np(b) for b in txt_raw])
    return img_batch, txt_batch

def buffer_clear():
    """Xóa buffer mà không lấy data (dùng khi reset thủ công)."""
    redis_sync_client.delete(KEY_BUFFER_IMAGE, KEY_BUFFER_TEXT)


# ============================================================
# ASYNC WRAPPERS (dùng trong FastAPI khi cần await buffer ops)
# ============================================================
# Dùng aioredis với decode_responses=False riêng cho buffer async

_async_bin_pool = aioredis.ConnectionPool(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=False,
)
_redis_bin = aioredis.Redis(connection_pool=_async_bin_pool)

async def async_buffer_count() -> int:
    return await _redis_bin.llen(KEY_BUFFER_IMAGE)