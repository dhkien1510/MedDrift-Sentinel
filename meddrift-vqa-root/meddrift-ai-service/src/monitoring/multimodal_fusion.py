"""
Per-modality PCA on reference data, then concat(img_proj, txt_proj) → joint vector.
Matches the fusion sketch: CLIP/RoBERTa (or configured encoders) → PCA each → concat.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class MultimodalPCABundle:
    image_pca: PCA
    text_pca: PCA
    n_components: int

    def joint_dim(self) -> int:
        return 2 * self.n_components


def fit_multimodal_pca(
    image_ref: np.ndarray,
    text_ref: np.ndarray,
    n_components: int = 128,
) -> MultimodalPCABundle:
    """
    Fit one PCA on image rows and one on text rows (reference only).

    Args:
        image_ref: (N, D_img)
        text_ref: (N, D_txt) — same N as image_ref (paired samples)
        n_components: output dim per modality before concat

    Returns:
        MultimodalPCABundle
    """
    if image_ref.shape[0] != text_ref.shape[0]:
        raise ValueError(
            f"Paired reference required: image N={image_ref.shape[0]} vs text N={text_ref.shape[0]}"
        )
    n_samples = image_ref.shape[0]
    n_img = min(n_components, image_ref.shape[1], n_samples)
    n_txt = min(n_components, text_ref.shape[1], n_samples)
    image_pca = PCA(n_components=n_img, svd_solver="full", random_state=42)
    text_pca = PCA(n_components=n_txt, svd_solver="full", random_state=42)
    image_pca.fit(image_ref.astype(np.float64))
    text_pca.fit(text_ref.astype(np.float64))
    return MultimodalPCABundle(image_pca=image_pca, text_pca=text_pca, n_components=n_components)


def transform_joint(
    image_emb: np.ndarray,
    text_emb: np.ndarray,
    bundle: MultimodalPCABundle,
) -> np.ndarray:
    """
    Project + concat. Shapes:
        image_emb: (N, D_img)
        text_emb: (N, D_txt)
    Returns:
        (N, n_img_actual + n_txt_actual) float32
    """
    if image_emb.shape[0] != text_emb.shape[0]:
        raise ValueError("image_emb and text_emb must have the same number of rows")
    img_p = bundle.image_pca.transform(image_emb.astype(np.float64))
    txt_p = bundle.text_pca.transform(text_emb.astype(np.float64))
    return np.hstack([img_p, txt_p]).astype(np.float32)


def save_bundle(path: str | Path, bundle: MultimodalPCABundle) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "image_pca": bundle.image_pca,
        "text_pca": bundle.text_pca,
        "n_components": bundle.n_components,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_bundle(path: str | Path) -> MultimodalPCABundle:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return MultimodalPCABundle(
        image_pca=payload["image_pca"],
        text_pca=payload["text_pca"],
        n_components=int(payload["n_components"]),
    )
