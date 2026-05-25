"""
Build PCA bundle + joint reference embeddings for the multimodal drift path.

Prerequisites:
  Run `build_reference.py` first so paired embeddings exist under:
    {root}/data/reference_data/image/{safe_image_enc}/{image_file}
    {root}/data/reference_data/text/{safe_text_enc}/{text_file}

Outputs (paths from drift_config.yaml → multimodal):
  - pca_pickle: two sklearn PCA objects fit on reference image/text rows
  - joint_reference_npy: concat(projected_ref_img, projected_ref_txt)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SERVICE_DIR)
MON_DIR = os.path.join(SERVICE_DIR, "src", "monitoring")
sys.path.insert(0, MON_DIR)

from multimodal_fusion import fit_multimodal_pca, save_bundle, transform_joint  # noqa: E402


def _safe(name: str) -> str:
    return name.replace("/", "--")


def load_yaml():
    path = os.path.join(ROOT_DIR, "configs", "drift_config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_yaml()
    mm = cfg.get("multimodal") or {}
    if not mm:
        print("No `multimodal` section in drift_config.yaml — add it first.")
        return 1

    ref = cfg["reference"]
    img_enc = cfg["image"]["encoder"]
    txt_enc = cfg["text"]["encoder"]
    img_rel = os.path.join(
        ref["data_dir"],
        "image",
        _safe(img_enc),
        ref["image_file"],
    )
    txt_rel = os.path.join(
        ref["data_dir"],
        "text",
        _safe(txt_enc),
        ref["text_file"],
    )
    img_path = os.path.join(ROOT_DIR, img_rel)
    txt_path = os.path.join(ROOT_DIR, txt_rel)

    if not os.path.isfile(img_path):
        print(f"Missing image embeddings: {img_path}\nRun scripts/build_reference.py for image first.")
        return 1
    if not os.path.isfile(txt_path):
        print(f"Missing text embeddings: {txt_path}\nRun scripts/build_reference.py for text first.")
        return 1

    image_ref = np.load(img_path)
    text_ref = np.load(txt_path)
    if image_ref.shape[0] != text_ref.shape[0]:
        print(f"Row mismatch: image {image_ref.shape[0]} vs text {text_ref.shape[0]}")
        return 1

    n_comp = int(mm.get("n_components", 128))
    print(f"Fitting PCA (n_components={n_comp}) on N={image_ref.shape[0]} paired samples...")
    bundle = fit_multimodal_pca(image_ref, text_ref, n_components=n_comp)
    joint_ref = transform_joint(image_ref, text_ref, bundle)

    pca_out = os.path.join(ROOT_DIR, mm["pca_pickle"])
    joint_out = os.path.join(ROOT_DIR, mm["joint_reference_npy"])
    save_bundle(pca_out, bundle)
    os.makedirs(os.path.dirname(joint_out), exist_ok=True)
    np.save(joint_out, joint_ref)

    print(f"Saved PCA bundle → {pca_out}")
    print(f"Saved joint reference ({joint_ref.shape}) → {joint_out}")
    print("Set multimodal.enabled: true in configs/drift_config.yaml to use this in batch drift reports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
