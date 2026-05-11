import registry
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from alibi_detect.cd import MMDDrift, LSDDDrift, KSDrift, CVMDrift, LearnedKernelDrift, ContextMMDDrift, ClassifierDrift, SpotTheDiffDrift
from alibi_detect.utils.pytorch import DeepKernel


config = registry.load_config(isImage=False)

# Load model + tokenizer 1 lần khi import
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
model = AutoModel.from_pretrained(config['model_id']).to(device)
model.eval()


def extract_text_embedding(dataset: list):    
    """
    Trích xuất embedding từ danh sách câu hỏi.
    
    Args:
        dataset: List[str] — danh sách câu hỏi
    
    Returns:
        np.ndarray shape (N, 768) — CLS embeddings
    """
    inputs = tokenizer(
        dataset,
        return_tensors="pt",
        max_length=128,
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
    
    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    return cls_embedding


def check_text_drift(test_data: list, p_threshold: float, algorithm: str, ref_data: np.ndarray):
    """
    Run drift detection trên raw text questions.
    Extract embeddings trước, rồi chạy detector.
    """
    embeddings = []
    batch_size = 32
    for i in range(0, len(test_data), batch_size):
        batch_questions = test_data[i: i + batch_size]
        embedding = extract_text_embedding(batch_questions)
        embeddings.append(embedding)
    embeddings = np.vstack(embeddings)
    detector = registry.get_detector(name=algorithm, ref_data=ref_data, p_val=p_threshold, input_dim=768)
    result = detector.predict(embeddings)
    return {
        'is_drift': bool(result['data']['is_drift']),
        'p_value': float(result['data']['p_val']) if np.isscalar(result['data']['p_val']) else float(np.mean(result['data']['p_val'])),
        'distance': float(result['data']['distance']) if np.isscalar(result['data']['distance']) else float(np.mean(result['data']['distance'])),
    }


def check_text_drift_from_embeddings(embeddings: np.ndarray, p_threshold, algorithm, ref_data):
    """
    Run drift detection trên pre-extracted text embeddings (từ buffer).
    Khác với check_text_drift() — hàm này KHÔNG extract embedding,
    vì embedding đã được extract sẵn khi request đến.
    
    Args:
        embeddings: np.ndarray shape (N, 768) — batch embeddings từ buffer
        p_threshold: Ngưỡng p-value
        algorithm: Tên thuật toán drift detection
        ref_data: Reference embeddings
    
    Returns:
        dict: {is_drift, p_value, distance}
    """
    detector = registry.get_detector(name=algorithm, ref_data=ref_data, p_val=p_threshold, input_dim=768)
    result = detector.predict(embeddings)
    return {
        'is_drift': bool(result['data']['is_drift']),
        'p_value': float(result['data']['p_val']) if np.isscalar(result['data']['p_val']) else float(np.mean(result['data']['p_val'])),
        'distance': float(result['data']['distance']) if np.isscalar(result['data']['distance']) else float(np.mean(result['data']['distance'])),
    }


if __name__ == '__main__':
    print("hello")