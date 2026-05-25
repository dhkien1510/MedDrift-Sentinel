"""
registry.py — Drift Detector Registry
======================================
Quản lý tất cả thuật toán drift detection theo kiến trúc pluggable.

[PATCH] load_config() và load_config_meta() được cập nhật:
  - Source of truth cho config: MongoDB (qua db/mongo_client.py) → Redis cache → fallback YAML local
  - Source of truth cho reference .npy: MinIO bucket 'reference-data' → local cache /tmp/
  - Local YAML và local .npy chỉ còn là bootstrap fallback lần đầu khởi động

Phân loại thuật toán thành 3 nhóm:
  - SIMPLE:   Chỉ cần x_ref + p_val (MMD, KS, CVM, LSDD)
  - KERNEL:   Cần thêm neural network làm kernel (LearnedKernel, ContextMMD)
  - MODEL:    Cần thêm classifier/regressor model (Classifier, SpotTheDiff)
"""

import torch
import torch.nn as nn
import numpy as np
import os
import yaml
import logging

logger = logging.getLogger(__name__)

from alibi_detect.cd import (
    MMDDrift,
    LSDDDrift,
    KSDrift,
    CVMDrift,
    LearnedKernelDrift,
    ContextMMDDrift,
    ClassifierDrift,
    SpotTheDiffDrift,
)
from alibi_detect.utils.pytorch import DeepKernel


SIMPLE_DETECTORS = {
    "kolmogorov-smirnov": KSDrift,
    "cramer-von-mises":   CVMDrift,
    "mmd":                MMDDrift,
    "least-square-density-difference": LSDDDrift,
}

KERNEL_DETECTORS = {
    "learned-kernel-mmd":  LearnedKernelDrift,
    "context-aware-mmd":   ContextMMDDrift,
}

MODEL_DETECTORS = {}

_SIMPLE_PYTORCH_BACKENDS = frozenset({"mmd", "least-square-density-difference"})


# ============================================================
# LOCAL YAML FALLBACK — chỉ dùng khi DB chưa có config
# ============================================================
def _get_root_dir() -> str:
    current_script = os.path.abspath(__file__)
    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(current_script)
            )
        )
    )

def _read_yaml() -> dict:
    """Đọc drift_config.yaml — chỉ dùng làm bootstrap fallback."""
    config_path = os.path.join(_get_root_dir(), "configs/drift_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _safe_encoder_name(model_id: str) -> str:
    return model_id.replace("/", "--")


# ============================================================
# CONFIG: MongoDB → Redis cache → YAML fallback
# ============================================================

def _load_yaml_meta(isImage: bool) -> dict | None:
    """Đọc metadata từ YAML local — chỉ dùng khi DB chưa sẵn sàng."""
    try:
        config = _read_yaml()
        section = config["image"] if isImage else config["text"]
        return {
            "encoder":      section["encoder"],
            "p_threshold":  float(section["p_threshold"]),
            "algorithm":    section["algorithm"],
            "allow_method": list(section["allow_method"]),
            "allow_encoder": list(section["allow_encoder"]),
            "threshold":    float(config["buffer"]["threshold"]),
        }
    except Exception as e:
        logger.error("YAML fallback failed: %s", e)
        return None


def load_config_meta(isImage: bool) -> dict | None:
    """
    Lấy metadata config (không load ref_embeddings).

    Thứ tự ưu tiên:
      1. Redis cache (drift_config key)
      2. MongoDB active config
      3. YAML local fallback

    [PATCH] Trước đây chỉ đọc YAML. Nay đọc từ DB trước.
    Sync wrapper — dùng asyncio.run() hoặc thread executor khi gọi từ sync context.
    """
    # Thử Redis (sync) trước — tránh phải mở event loop
    try:
        import redis as _redis_sync
        import json

        REDIS_HOST     = os.getenv("REDIS_HOST",     "redis")
        REDIS_PORT     = int(os.getenv("REDIS_PORT", 6379))
        REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

        _r = _redis_sync.Redis(
            host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD,
            decode_responses=True, socket_connect_timeout=1,
        )
        cached = _r.get("drift_config")
        if cached:
            full = json.loads(cached)
            # full config chứa cả image + text — chọn đúng section
            section_key = "image" if isImage else "text"
            if section_key in full:
                s = full[section_key]
                return {
                    "encoder":       s["encoder"],
                    "p_threshold":   float(s["p_threshold"]),
                    "algorithm":     s["algorithm"],
                    "allow_method":  list(s.get("allow_method", [])),
                    "allow_encoder": list(s.get("allow_encoder", [])),
                    "threshold":     float(full.get("buffer", {}).get("threshold", 100)),
                }
    except Exception:
        pass  # Redis không sẵn sàng → thử YAML

    # Fallback YAML
    return _load_yaml_meta(isImage)


# ============================================================
# REFERENCE EMBEDDINGS: MinIO → local cache → local disk fallback
# ============================================================

def _load_ref_from_minio(isImage: bool, encoder: str, yaml_cfg: dict) -> np.ndarray | None:
    """
    Download reference .npy từ MinIO.
    [PATCH] Thay thế np.load() trực tiếp từ local disk.
    """
    try:
        from db.minio_client import download_npy, reference_object_name, object_exists
        filename = yaml_cfg["reference"]["image_file"] if isImage else yaml_cfg["reference"]["text_file"]
        obj_name = reference_object_name(isImage, encoder, filename)
        if object_exists(obj_name):
            return download_npy(obj_name)
        logger.warning("MinIO: '%s' không tồn tại — thử local fallback.", obj_name)
    except Exception as e:
        logger.warning("MinIO download failed: %s — thử local fallback.", e)
    return None


def _load_ref_from_local(isImage: bool, encoder: str, yaml_cfg: dict) -> np.ndarray | None:
    """Đọc .npy từ local disk (fallback khi MinIO chưa có file)."""
    root_dir = _get_root_dir()
    ref_dir  = yaml_cfg["reference"]["data_dir"]
    ref_file = yaml_cfg["reference"]["image_file"] if isImage else yaml_cfg["reference"]["text_file"]
    sub      = "image" if isImage else "text"

    nested = os.path.join(root_dir, ref_dir, sub, _safe_encoder_name(encoder), ref_file)
    flat   = os.path.join(root_dir, ref_dir, ref_file)

    for path in [nested, flat]:
        if os.path.isfile(path):
            logger.info("Local fallback: đọc ref từ '%s'", path)
            return np.load(path)
    return None


def load_config(isImage: bool) -> dict | None:
    """
    Load đầy đủ config kể cả ref_embeddings — dùng khi chạy drift detection.

    [PATCH] ref_embeddings giờ được load từ MinIO (có local cache),
    không còn đọc trực tiếp từ local disk nữa.
    """
    try:
        meta = load_config_meta(isImage)
        if meta is None:
            return None

        yaml_cfg = _read_yaml()
        encoder  = meta["encoder"]

        # 1. Thử MinIO trước
        ref_emb = _load_ref_from_minio(isImage, encoder, yaml_cfg)

        # 2. Fallback local disk
        if ref_emb is None:
            ref_emb = _load_ref_from_local(isImage, encoder, yaml_cfg)

        if ref_emb is None:
            label = "image" if isImage else "text"
            logger.error(
                "Không tìm thấy reference embeddings (%s) — "
                "upload lên MinIO hoặc đặt file vào data/reference_data/.", label
            )
            return None

        return {**meta, "ref_embeddings": ref_emb}

    except Exception as e:
        logger.error("load_config failed: %s", e)
        return None


# ============================================================
# NEURAL NETS
# ============================================================
def _build_projection_net(input_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 128), nn.ReLU(),
        nn.Linear(128, 32),        nn.ReLU(),
    )

def _build_classifier_model(input_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 64), nn.ReLU(),
        nn.Linear(64, 2),
    )


# ============================================================
# FACTORY
# ============================================================
def get_detector(name: str, ref_data: np.ndarray, p_val: float, input_dim: int = None):
    if name in SIMPLE_DETECTORS:
        ctor   = SIMPLE_DETECTORS[name]
        kwargs = {"x_ref": ref_data, "p_val": p_val}
        if name in _SIMPLE_PYTORCH_BACKENDS:
            kwargs["backend"] = "pytorch"
        return ctor(**kwargs)

    if name in KERNEL_DETECTORS:
        proj   = _build_projection_net(input_dim)
        kernel = DeepKernel(proj, eps=0.01)
        return KERNEL_DETECTORS[name](
            x_ref=ref_data, kernel=kernel, p_val=p_val, backend="pytorch"
        )

    if name in MODEL_DETECTORS:
        proj = _build_classifier_model(input_dim)
        return MODEL_DETECTORS[name](
            x_ref=ref_data, model=proj, p_val=p_val, backend="pytorch"
        )

    raise ValueError(
        f"Detector '{name}' không tồn tại.\n"
        f"Các thuật toán hỗ trợ: {list_available_algorithms()}"
    )


# ============================================================
# UTILITIES
# ============================================================
def list_available_algorithms() -> list:
    return list({**SIMPLE_DETECTORS, **KERNEL_DETECTORS, **MODEL_DETECTORS})

def get_algorithm_info() -> dict:
    return {
        "simple": {n: {"type": "simple", "requires_training": False} for n in SIMPLE_DETECTORS},
        "kernel": {n: {"type": "kernel", "requires_training": True}  for n in KERNEL_DETECTORS},
        "model":  {n: {"type": "model",  "requires_training": True}  for n in MODEL_DETECTORS},
    }