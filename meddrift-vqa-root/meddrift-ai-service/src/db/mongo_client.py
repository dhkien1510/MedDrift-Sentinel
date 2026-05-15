"""
db/mongo_client.py — MongoDB Client
======================================
Collections:
  - drift_reports   : lịch sử báo cáo drift      [giữ nguyên]
  - chat_sessions   : phiên chat                  [giữ nguyên]
  - drift_config    : lịch sử config + active     [PATCH MỚI]

Lý do thêm drift_config:
  - apply_config() cũ ghi thẳng vào YAML trên disk → không đồng bộ
    giữa nhiều container, không có audit trail, dễ mất khi redeploy.
  - Nay config được persist vào MongoDB (source of truth) và cache
    vào Redis để đọc nhanh. Local YAML chỉ là bootstrap fallback.
"""

from motor.motor_asyncio import AsyncIOMotorClient
import os

MONGO_URI     = os.getenv("MONGO_URI",     "mongodb://meddrift:meddrift123@mongodb:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "meddirft")
client = AsyncIOMotorClient(MONGO_URI)
db     = client[MONGO_DB_NAME]

# ── Collections ─────────────────────────────────────────────
drift_reports = db["drift_reports"]
chat_sessions = db["chat_sessions"]
drift_config  = db["drift_config"]   # [PATCH] config history + active flag


async def init_db():
    """
    Tạo indexes khi FastAPI khởi động.
    drift_config dùng index trên 'active' để get_active_config() O(1).
    """
    await drift_reports.create_index([("timestamp", -1)])
    await drift_config.create_index([("active", -1)])
    await drift_config.create_index([("updated_at", -1)])
    print("✅ Đã kết nối MongoDB và tạo Indexes thành công!")


# ============================================================
# CONFIG HELPERS
# ============================================================

async def save_config(config_dict: dict) -> str:
    """
    Lưu config mới vào MongoDB:
      1. Unset 'active' trên document cũ.
      2. Insert document mới với active=True.
    Trả về inserted _id dạng string.
    """
    from datetime import datetime

    # Bỏ active flag của bản cũ
    await drift_config.update_many({"active": True}, {"$set": {"active": False}})

    doc = {
        **config_dict,
        "active": True,
        "updated_at": datetime.utcnow().isoformat(),
    }
    result = await drift_config.insert_one(doc)
    return str(result.inserted_id)


async def get_active_config() -> dict | None:
    """
    Lấy config đang active (source of truth từ MongoDB).
    Trả về None nếu chưa có config nào được lưu (fallback sang YAML local).
    """
    doc = await drift_config.find_one({"active": True}, sort=[("updated_at", -1)])
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc


async def get_config_history(limit: int = 20) -> list:
    """Lấy lịch sử config (mới nhất trước), dùng cho audit trail."""
    cursor = drift_config.find().sort("updated_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs