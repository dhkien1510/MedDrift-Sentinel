import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from alibi_detect.cd import MMDDrift
import registry




config = registry.load_config(isImage=True)

# Module level — load 1 lần khi import (drift_agent.py cần)
print(f"[Image Drift] Loading CLIP: {config['encoder']}...")
clip_processor = CLIPProcessor.from_pretrained(config['encoder'])
clip_model = CLIPModel.from_pretrained(config['encoder'])
clip_model.eval()
print("[Image Drift] CLIP loaded!")

def extract_image_embedding(image_input: Image.Image) -> np.ndarray:
    """
    Trích xuất embedding từ 1 ảnh PIL.

    Args:
        image_input: PIL.Image.Image — ảnh đã mở sẵn

    Returns:
        np.ndarray shape (512,), dtype float32 — vector đã normalize
    """
    # 1. Chuyển sang RGB (X-ray grayscale → RGB để CLIP xử lý được)
    if image_input.mode != "RGB":
        image_input = image_input.convert("RGB")

    # 2. Preprocess: resize, normalize theo chuẩn CLIP
    inputs = clip_processor(images=image_input, return_tensors="pt")

    # 3. Extract features (không tính gradient vì chỉ inference)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)

    # 4. L2 normalize (giống build_reference.py)
    features = features / features.norm(p=2, dim=-1, keepdim=True)

    # 5. Chuyển sang numpy, bỏ batch dimension: (1, 512) → (512,)
    return features.squeeze().cpu().numpy().astype(np.float32)
    

def check_image_drift(test_data, p_threshold, algorithm, ref_data):
    """
    Run drift detection trên raw PIL images.
    Extract embeddings trước, rồi chạy detector.
    """
    embeddings = [extract_image_embedding(i) for i in test_data]
    embeddings = np.array(embeddings)
    detector = registry.get_detector(name=algorithm, ref_data=ref_data, p_val=p_threshold, input_dim=512)

    result = detector.predict(embeddings)

    return {
        'is_drift': bool(result['data']['is_drift']),
        'p_value': float(result['data']['p_val']) if np.isscalar(result['data']['p_val']) else float(np.mean(result['data']['p_val'])),
        'distance': float(result['data']['distance']) if np.isscalar(result['data']['distance']) else float(np.mean(result['data']['distance'])),
    }


def check_image_drift_from_embeddings(embeddings: np.ndarray, p_threshold, algorithm, ref_data):
    """
    Run drift detection trên pre-extracted image embeddings (từ buffer).
    Khác với check_image_drift() — hàm này KHÔNG extract embedding,
    vì embedding đã được extract sẵn khi request đến.
    
    Args:
        embeddings: np.ndarray shape (N, 512) — batch embeddings từ buffer
        p_threshold: Ngưỡng p-value
        algorithm: Tên thuật toán drift detection
        ref_data: Reference embeddings
    
    Returns:
        dict: {is_drift, p_value, distance}
    """
    detector = registry.get_detector(name=algorithm, ref_data=ref_data, p_val=p_threshold, input_dim=512)
    result = detector.predict(embeddings)
    return {
        'is_drift': bool(result['data']['is_drift']),
        'p_value': float(result['data']['p_val']) if np.isscalar(result['data']['p_val']) else float(np.mean(result['data']['p_val'])),
        'distance': float(result['data']['distance']) if np.isscalar(result['data']['distance']) else float(np.mean(result['data']['distance'])),
    }


if __name__ == "__main__":
    random_images = [
        Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        for _ in range(5)
    ]
    result = check_image_drift(random_images, config['p_threshold'], config['algorithm'], config['ref_embeddings'])
    print(f"Algorithm: {config['algorithm']}, p_threshold: {config['p_threshold']}")
    print(result)
