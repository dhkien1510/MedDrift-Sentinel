"""
registry.py — Drift Detector Registry
======================================
Quản lý tất cả thuật toán drift detection theo kiến trúc pluggable.

Phân loại thuật toán thành 3 nhóm:
  - SIMPLE:   Chỉ cần x_ref + p_val (MMD, KS, CVM, LSDD)
  - KERNEL:   Cần thêm neural network làm kernel (LearnedKernel, ContextMMD)
  - MODEL:    Cần thêm classifier/regressor model (Classifier, SpotTheDiff)

Cách dùng:
  detector = get_detector("mmd", ref_data=ref_embeddings, p_val=0.05, input_dim=512)
  result = detector.predict(test_embeddings)
"""

import torch
import torch.nn as nn
import numpy as np
import os
import yaml

# === Alibi-detect imports ===
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


# ============================================================
# NHÓM 1: Simple Detectors
# Chỉ cần x_ref + p_val → tạo detector ngay
# ============================================================
SIMPLE_DETECTORS = {
    "kolmogorov-smirnov": KSDrift,
    "cramer-von-mises": CVMDrift,
    "mmd": MMDDrift,
    "least-square-density-difference": LSDDDrift,
}


# ============================================================
# NHÓM 2: Kernel-based Detectors
# Cần tạo projection network (nn.Sequential) + DeepKernel
# input_dim khác nhau: image=512, text=768
# ============================================================
KERNEL_DETECTORS = {
    "learned-kernel-mmd": LearnedKernelDrift,
    "context-aware-mmd": ContextMMDDrift,
}


# ============================================================
# NHÓM 3: Model-based Detectors
# Cần tạo classifier model (nn.Sequential) để phân biệt ref vs test
# ============================================================
MODEL_DETECTORS = {
    "classifier-uncertainty": ClassifierDrift,
    "spot-the-diff": SpotTheDiffDrift,
}


def load_config(isImage: bool):
    current_script = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_script)
    src_dir = os.path.dirname(current_dir)
    service_dir = os.path.dirname(src_dir)
    root_dir = os.path.dirname(service_dir)

    config_dir = os.path.join(root_dir, "configs/drift_config.yaml")

    try:
        with open(config_dir, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        try: 
            model_id =  config['image']['encoder'] if isImage else config['text']['encoder']
            p_threshold =  float(config['image']['p_threshold']) if isImage else float(config['text']['p_threshold'])
            algorithm =  config['image']['algorithm'] if isImage else config['text']['algorithm']
            allow_method = list(config['image']['allow_method']) if isImage else list(config['text']['allow_method'])
            ref_image_path = os.path.join(root_dir, config['reference']['data_dir'], config['reference']['image_file'])
            ref_text_path = os.path.join(root_dir, config['reference']['data_dir'], config['reference']['text_file'])
            ref_embeddings = np.load(ref_image_path) if isImage else np.load(ref_text_path)
   

            return {
                "model_id": model_id,
                "ref_embeddings": ref_embeddings,
                "p_threshold": p_threshold,
                "algorithm": algorithm,
                "allow_method": allow_method
            }
        except Exception as e:
            print(f"Error while loading config: {e}")
            return None
    except Exception as e:
        print(f"Error while loading config: {e}")
        return None


# ============================================================
# HELPER: Tạo Projection Network cho Kernel-based detectors
# ============================================================
def _build_projection_net(input_dim: int) -> nn.Sequential:
    """
    Tạo mạng neural nhỏ để project embedding xuống không gian thấp hơn.
    
    Args:
        input_dim: Số chiều đầu vào (512 cho image, 768 cho text)
    
    Returns:
        nn.Sequential: Mạng 2 layers
    """
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
        nn.ReLU()
    )

# ============================================================
# HELPER: Tạo Classifier Model cho Model-based detectors
# ============================================================
def _build_classifier_model(input_dim: int) -> nn.Sequential:
    """
    Tạo mạng classifier nhỏ để phân biệt ref data vs test data.
    
    Args:
        input_dim: Số chiều đầu vào (512 cho image, 768 cho text)
    
    Returns:
        nn.Sequential: Mạng binary classifier

    TODO: Bạn tự code mạng classifier ở đây.
    Gợi ý: input_dim → 64 → ReLU → 2 (binary: ref vs test)
    """
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )


# ============================================================
# FACTORY CHÍNH: Tạo detector theo tên
# ============================================================
def get_detector(name: str, ref_data: np.ndarray, p_val: float, input_dim: int = None):
    """
    Factory function — tạo drift detector theo tên thuật toán.

    Args:
        name:      Tên thuật toán (key trong registry)
        ref_data:  Reference embeddings, shape (N, D)
        p_val:     Ngưỡng p-value (mặc định 0.05)
        input_dim: Số chiều embedding (bắt buộc cho nhóm KERNEL và MODEL)

    Returns:
        Drift detector object (có method .predict())

    Raises:
        ValueError: Nếu tên thuật toán không tồn tại
    """

    # --- Nhóm 1: Simple ---
    if name in SIMPLE_DETECTORS:
        detector = SIMPLE_DETECTORS[name](x_ref=ref_data, p_val=p_val)
    # --- Nhóm 2: Kernel-based ---
    elif name in KERNEL_DETECTORS:
        proj = _build_projection_net(input_dim)
        kernel = DeepKernel(proj, eps=0.01)
        detector = KERNEL_DETECTORS[name](x_ref=ref_data, kernel=kernel, p_val=p_val)

    # --- Nhóm 3: Model-based ---
    elif name in MODEL_DETECTORS:
        proj = _build_classifier_model(input_dim)
        detector = MODEL_DETECTORS[name](x_ref=ref_data, model=proj, p_val=p_val, backend="pytorch")
    else:
        available = list_available_algorithms()
        raise ValueError(
            f"Detector '{name}' không tồn tại.\n"
            f"Các thuật toán hỗ trợ: {available}"
        )
    return detector

# ============================================================
# UTILITY: Liệt kê tất cả thuật toán
# ============================================================
def list_available_algorithms() -> list:
    """Trả về danh sách tất cả thuật toán hỗ trợ (cho API/frontend dropdown)."""
    all_algos = {}
    all_algos.update(SIMPLE_DETECTORS)
    all_algos.update(KERNEL_DETECTORS)
    all_algos.update(MODEL_DETECTORS)
    return list(all_algos.keys())


def get_algorithm_info() -> dict:
    """Trả về thông tin chi tiết từng thuật toán (cho frontend hiển thị)."""
    return {
        "simple": {
            name: {"type": "simple", "requires_training": False}
            for name in SIMPLE_DETECTORS
        },
        "kernel": {
            name: {"type": "kernel", "requires_training": True, "note": "Cần train kernel network"}
            for name in KERNEL_DETECTORS
        },
        "model": {
            name: {"type": "model", "requires_training": True, "note": "Cần train classifier"}
            for name in MODEL_DETECTORS
        },
    }


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    print("=== Available Algorithms ===")
    for algo in list_available_algorithms():
        print(f"  - {algo}")

    print(f"\nTổng: {len(list_available_algorithms())} thuật toán")

    print("\n=== Algorithm Info ===")
    info = get_algorithm_info()
    for group, algos in info.items():
        print(f"\n[{group.upper()}]")
        for name, details in algos.items():
            print(f"  {name}: {details}")