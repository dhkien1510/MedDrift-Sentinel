import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoModel
import registry

# ── Image encoders — khớp scripts/build_reference.py ─────────────────
CLIP_MODELS = {"openai/clip-vit-base-patch32"}
BIOMEDCLIP_MODELS = {
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
}
DINOV2_MODELS = {
    "microsoft/rad-dino",
    "microsoft/rad-dino-maira-2",
    "facebook/dinov2-base",
}
VIT_MODELS = {"google/vit-base-patch16-224"}

# ── Runtime stack (một trong các nhánh được gán khi import) ───────────
_IMAGE_BACKEND: str | None = None
_clip_processor = None
_clip_model = None
_biomedclip_model = None
_biomedclip_preprocess = None
_dinov2_processor = None
_dinov2_model = None
_vit_processor = None
_vit_model = None


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _init_image_encoder(model_id: str) -> None:
    """Chọn loader theo model_id — cùng router như build_reference.embed_images."""
    global _IMAGE_BACKEND, _clip_processor, _clip_model
    global _biomedclip_model, _biomedclip_preprocess
    global _dinov2_processor, _dinov2_model, _vit_processor, _vit_model

    device = _device()

    if model_id in BIOMEDCLIP_MODELS:
        import open_clip

        _biomedclip_model, _, _biomedclip_preprocess = open_clip.create_model_from_pretrained(
            f"hf-hub:{model_id}", device=device
        )
        _biomedclip_model.eval()
        _IMAGE_BACKEND = "biomedclip"
        print(f"[Image Drift] Loaded BiomedCLIP (open_clip hf-hub): {model_id}")
    elif model_id in CLIP_MODELS:
        _clip_processor = CLIPProcessor.from_pretrained(model_id)
        _clip_model = CLIPModel.from_pretrained(model_id).to(device)
        _clip_model.eval()
        _IMAGE_BACKEND = "clip"
        print(f"[Image Drift] Loaded CLIPProcessor + CLIPModel: {model_id}")
    elif model_id in DINOV2_MODELS:
        from transformers import Dinov2Model, AutoImageProcessor

        _dinov2_processor = AutoImageProcessor.from_pretrained(model_id)
        _dinov2_model = Dinov2Model.from_pretrained(model_id).to(device)
        _dinov2_model.eval()
        _IMAGE_BACKEND = "dinov2"
        print(f"[Image Drift] Loaded Dinov2Model + AutoImageProcessor: {model_id}")
    elif model_id in VIT_MODELS:
        from transformers import AutoImageProcessor

        _vit_processor = AutoImageProcessor.from_pretrained(model_id)
        _vit_model = AutoModel.from_pretrained(model_id).to(device)
        _vit_model.eval()
        _IMAGE_BACKEND = "vit"
        print(f"[Image Drift] Loaded AutoImageProcessor + AutoModel (ViT): {model_id}")
    else:
        from transformers import AutoImageProcessor

        print(f"[Image Drift] Fallback AutoModel for: {model_id}")
        _vit_processor = AutoImageProcessor.from_pretrained(model_id)
        _vit_model = AutoModel.from_pretrained(model_id).to(device)
        _vit_model.eval()
        _IMAGE_BACKEND = "vit"


config = registry.load_config_meta(isImage=True)
print(config)
if config:
    _init_image_encoder(config["encoder"])
else:
    print("[Image Drift] config is None — encoder not loaded")


def _embedding_dim_from_ref(ref_data: np.ndarray) -> int:
    if ref_data is None or not hasattr(ref_data, "ndim") or ref_data.ndim != 2:
        raise ValueError("ref_data phải là mảng 2D (N, D) để suy ra input_dim detector.")
    return int(ref_data.shape[1])


def extract_image_embedding(image_input: Image.Image) -> np.ndarray:
    """
    Trích xuất embedding từ 1 ảnh PIL — pipeline khớp build_reference.embed_images.
    """
    if _IMAGE_BACKEND is None:
        raise RuntimeError("Image encoder chưa được load (thiếu config hoặc lỗi init).")

    if image_input.mode != "RGB":
        image_input = image_input.convert("RGB")

    device = _device()

    with torch.no_grad():
        if _IMAGE_BACKEND == "clip":
            inputs = _clip_processor(images=image_input, return_tensors="pt").to(device)
            features = _clip_model.get_image_features(**inputs)
        elif _IMAGE_BACKEND == "biomedclip":
            batch = _biomedclip_preprocess(image_input).unsqueeze(0).to(device)
            features = _biomedclip_model.encode_image(batch)
        elif _IMAGE_BACKEND == "dinov2":
            inputs = _dinov2_processor(images=[image_input], return_tensors="pt").to(device)
            out = _dinov2_model(**inputs)
            features = out.last_hidden_state[:, 0, :]
        else:  # vit / fallback
            inputs = _vit_processor(images=[image_input], return_tensors="pt").to(device)
            out = _vit_model(**inputs)
            if out.pooler_output is not None:
                features = out.pooler_output
            else:
                features = out.last_hidden_state[:, 0, :]

    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.squeeze(0).cpu().numpy().astype(np.float32)


def check_image_drift(test_data, p_threshold, algorithm, ref_data):
    embeddings = [extract_image_embedding(i) for i in test_data]
    embeddings = np.array(embeddings)
    input_dim = _embedding_dim_from_ref(ref_data)
    detector = registry.get_detector(
        name=algorithm, ref_data=ref_data, p_val=p_threshold, input_dim=input_dim
    )
    result = detector.predict(embeddings)
    return {
        "is_drift": bool(result["data"]["is_drift"]),
        "p_value": float(result["data"]["p_val"])
        if np.isscalar(result["data"]["p_val"])
        else float(np.mean(result["data"]["p_val"])),
        "distance": float(result["data"]["distance"])
        if np.isscalar(result["data"]["distance"])
        else float(np.mean(result["data"]["distance"])),
    }


def check_image_drift_from_embeddings(embeddings: np.ndarray, p_threshold, algorithm, ref_data):
    input_dim = _embedding_dim_from_ref(ref_data)
    detector = registry.get_detector(
        name=algorithm, ref_data=ref_data, p_val=p_threshold, input_dim=input_dim
    )
    result = detector.predict(embeddings)
    return {
        "is_drift": bool(result["data"]["is_drift"]),
        "p_value": float(result["data"]["p_val"])
        if np.isscalar(result["data"]["p_val"])
        else float(np.mean(result["data"]["p_val"])),
        "distance": float(result["data"]["distance"])
        if np.isscalar(result["data"]["distance"])
        else float(np.mean(result["data"]["distance"])),
    }


if __name__ == "__main__":
    random_images = [
        Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        for _ in range(5)
    ]
    full_cfg = registry.load_config(isImage=True)
    if not full_cfg or "ref_embeddings" not in full_cfg:
        print("Cần drift_config.yaml + reference embeddings (registry.load_config).")
    else:
        result = check_image_drift(
            random_images,
            full_cfg["p_threshold"],
            full_cfg["algorithm"],
            full_cfg["ref_embeddings"],
        )
        print(f"Algorithm: {full_cfg['algorithm']}, p_threshold: {full_cfg['p_threshold']}")
        print(result)
