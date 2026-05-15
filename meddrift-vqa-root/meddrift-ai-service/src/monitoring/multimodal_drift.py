"""
Joint multimodal drift: PCA per modality → concat → MMD (or registry algorithm) vs reference joint.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

import registry
from multimodal_fusion import MultimodalPCABundle, load_bundle, transform_joint


def _joint_feature_dim(bundle: MultimodalPCABundle) -> int:
    return bundle.image_pca.n_components_ + bundle.text_pca.n_components_


def load_multimodal_runtime_config() -> dict[str, Any] | None:
    """
    Load YAML multimodal section + artifacts from disk.

    Returns:
        None — multimodal disabled in yaml
        dict with keys:
          ready (bool)
          reason (str, if not ready)
          p_threshold, algorithm
          bundle, ref_joint (if ready)
    """
    try:
        cfg = registry._read_yaml()
    except Exception:
        return None

    mm = cfg.get("multimodal") or {}
    if not mm.get("enabled", False):
        return None

    root = registry._get_root_dir()
    pca_path = os.path.join(root, mm["pca_pickle"])
    joint_path = os.path.join(root, mm["joint_reference_npy"])

    if not os.path.isfile(pca_path):
        return {
            "ready": False,
            "reason": f"Missing PCA pickle (fit reference first): {pca_path}",
            "pca_path": pca_path,
            "joint_path": joint_path,
        }
    if not os.path.isfile(joint_path):
        return {
            "ready": False,
            "reason": f"Missing joint reference embeddings: {joint_path}",
            "pca_path": pca_path,
            "joint_path": joint_path,
        }

    try:
        bundle = load_bundle(pca_path)
        ref_joint = np.load(joint_path)
    except Exception as e:
        return {
            "ready": False,
            "reason": f"Failed to load multimodal artifacts: {e}",
            "pca_path": pca_path,
            "joint_path": joint_path,
        }

    algorithm = mm.get("algorithm", "mmd")
    p_threshold = float(mm.get("p_threshold", 0.05))

    return {
        "ready": True,
        "p_threshold": p_threshold,
        "algorithm": algorithm,
        "bundle": bundle,
        "ref_joint": ref_joint,
        "pca_path": pca_path,
        "joint_path": joint_path,
    }


def check_multimodal_drift_from_embeddings(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """
    Args:
        image_embeddings: (N, D_img)
        text_embeddings: (N, D_txt)
        runtime: output of load_multimodal_runtime_config() with ready=True

    Returns:
        metrics dict including is_drift, p_value, distance, joint_shape, mmd_stat alias
    """
    if not runtime.get("ready"):
        raise ValueError("runtime config not ready")

    bundle: MultimodalPCABundle = runtime["bundle"]
    ref_joint: np.ndarray = runtime["ref_joint"]
    joint_test = transform_joint(image_embeddings, text_embeddings, bundle)
    expected_cols = _joint_feature_dim(bundle)

    if ref_joint.shape[1] != joint_test.shape[1]:
        return {
            "drift_ran": False,
            "is_error": True,
            "message": (
                f"Joint dim mismatch: ref {ref_joint.shape[1]} vs test {joint_test.shape[1]} "
                f"(expected {expected_cols} from current PCA bundle)."
            ),
            "metrics": {},
        }

    input_dim = int(ref_joint.shape[1])
    detector = registry.get_detector(
        name=runtime["algorithm"],
        ref_data=ref_joint,
        p_val=runtime["p_threshold"],
        input_dim=input_dim,
    )
    result = detector.predict(joint_test)
    p_val = result["data"]["p_val"]
    dist = result["data"]["distance"]
    is_drift = result["data"]["is_drift"]

    return {
        "drift_ran": True,
        "is_error": False,
        "is_drift": bool(is_drift),
        "p_value": float(p_val) if np.isscalar(p_val) else float(np.mean(p_val)),
        "distance": float(dist) if np.isscalar(dist) else float(np.mean(dist)),
        "algorithm": runtime["algorithm"],
        "joint_dim": int(joint_test.shape[1]),
        "n_test_samples": int(joint_test.shape[0]),
        "metrics": {
            "mmd_stat": float(dist) if np.isscalar(dist) else float(np.mean(dist)),
            "pval": float(p_val) if np.isscalar(p_val) else float(np.mean(p_val)),
            "is_drift": bool(is_drift),
        },
    }


def multimodal_drift_report(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
) -> dict[str, Any] | None:
    """
    Safe entry: returns None if feature disabled; otherwise a report dict
    suitable for MongoDB / API (always JSON-serializable values).
    """
    runtime = load_multimodal_runtime_config()
    if runtime is None:
        return None

    if not runtime.get("ready"):
        return {
            "enabled": True,
            "drift_ran": False,
            "is_error": True,
            "message": runtime.get("reason", "Multimodal drift not ready"),
            "metrics": {},
        }

    out = check_multimodal_drift_from_embeddings(
        image_embeddings, text_embeddings, runtime
    )
    out["enabled"] = True
    return out


def sample_multimodal_summary(
    image_embedding: np.ndarray,
    text_embedding: np.ndarray,
) -> dict[str, Any]:
    """
    Single-request multimodal output for API / chat: project (image, text) through
    reference-fitted PCA → concat. No batch-level p-value here; reports joint geometry
    vs reference centroid (heuristic OOD-style signal).

    Args:
        image_embedding: shape (D_img,) or (1, D_img)
        text_embedding: shape (D_txt,) or (1, D_txt)
    """
    try:
        img = np.asarray(image_embedding, dtype=np.float32).reshape(1, -1)
        txt = np.asarray(text_embedding, dtype=np.float32).reshape(1, -1)
    except Exception as e:
        return {"enabled": False, "message": f"Invalid embedding input: {e}"}

    runtime = load_multimodal_runtime_config()
    if runtime is None:
        return {
            "enabled": False,
            "message": "Multimodal pipeline is disabled (multimodal.enabled: false).",
        }

    if not runtime.get("ready"):
        return {
            "enabled": True,
            "ready": False,
            "message": runtime.get(
                "reason",
                "PCA bundle or joint reference missing — run scripts/build_multimodal_reference.py.",
            ),
        }

    try:
        bundle: MultimodalPCABundle = runtime["bundle"]
        ref_joint: np.ndarray = runtime["ref_joint"]
        joint = transform_joint(img, txt, bundle).reshape(-1)
        ref_mean = ref_joint.mean(axis=0)
        dist = float(np.linalg.norm(joint - ref_mean))
        ref_dists = np.linalg.norm(ref_joint - ref_mean, axis=1)
        med = float(np.median(ref_dists) + 1e-9)
        ratio = float(dist / med)
        preview_dim = min(8, joint.shape[0])
        preview = [float(x) for x in joint[:preview_dim]]
    except Exception as e:
        return {
            "enabled": True,
            "ready": False,
            "message": f"Projection failed (encoder dims must match PCA training data): {e}",
        }

    return {
        "enabled": True,
        "ready": True,
        "joint_dim": int(joint.shape[0]),
        "joint_projection_preview": preview,
        "joint_l2_norm": float(np.linalg.norm(joint)),
        "distance_to_reference_centroid_l2": dist,
        "reference_typical_distance_median": med,
        "distance_ratio_vs_typical": ratio,
        "interpretation_hint": (
            "Joint embedding is farther from the reference center than a typical in-distribution pair (heuristic)."
            if ratio > 2.0
            else "Joint embedding lies in a typical range vs the reference cloud (heuristic)."
        ),
        "note": (
            "Batch-level p-values (MMD) appear after the buffer flush in Dashboard reports, "
            "not from a single upload."
        ),
    }
