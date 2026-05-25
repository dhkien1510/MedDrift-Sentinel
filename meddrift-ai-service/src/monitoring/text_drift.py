import registry
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# ── Text encoders — khớp scripts/build_reference.py ─────────────────
BERT_MODELS = {
    "dmis-lab/biobert-v1.1",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
}
SBERT_MODELS = {
    "NeuML/pubmedbert-base-embeddings",
    "NeuML/pubmedbert-base-embeddings-matryoshka",
    "pritamdeka/S-PubMedBert-MS-MARCO",
}
MODEL2VEC_MODELS = {
    "NeuML/pubmedbert-base-embeddings-8M",
}

_TEXT_BACKEND: str | None = None
_tokenizer = None
_bert_model = None
_sbert_model = None
_m2v_model = None

config = registry.load_config_meta(isImage=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _init_text_encoder(model_id: str) -> None:
    """Chọn loader theo model_id — cùng router như build_reference.embed_texts."""
    global _TEXT_BACKEND, _tokenizer, _bert_model, _sbert_model, _m2v_model

    if model_id in MODEL2VEC_MODELS:
        from model2vec import StaticModel

        _m2v_model = StaticModel.from_pretrained(model_id)
        _TEXT_BACKEND = "model2vec"
        print(f"[Text Drift] Loaded model2vec StaticModel: {model_id}")
    elif model_id in SBERT_MODELS:
        from sentence_transformers import SentenceTransformer

        _sbert_model = SentenceTransformer(model_id)
        _TEXT_BACKEND = "sbert"
        print(f"[Text Drift] Loaded SentenceTransformer: {model_id}")
    else:
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _bert_model = AutoModel.from_pretrained(model_id).to(device)
        _bert_model.eval()
        _TEXT_BACKEND = "bert"
        if model_id in BERT_MODELS:
            print(f"[Text Drift] Loaded AutoTokenizer + AutoModel (BERT mean pool): {model_id}")
        else:
            print(f"[Text Drift] Fallback BERT-style (mean pool): {model_id}")


if config:
    _init_text_encoder(config["encoder"])
else:
    print("[Text Drift] config is None — encoder not loaded")


def _embedding_dim_from_ref(ref_data: np.ndarray) -> int:
    if ref_data is None or not hasattr(ref_data, "ndim") or ref_data.ndim != 2:
        raise ValueError("ref_data phải là mảng 2D (N, D) để suy ra input_dim detector.")
    return int(ref_data.shape[1])


def extract_text_embedding(dataset: list) -> np.ndarray:
    """
    Trích xuất embedding — khớp build_reference (mean pool BERT, SBERT, model2vec).
    """
    if _TEXT_BACKEND is None:
        raise RuntimeError("Text encoder chưa được load (thiếu config hoặc lỗi init).")

    if _TEXT_BACKEND == "model2vec":
        embeddings = _m2v_model.encode(dataset)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-9, None)
        return embeddings.astype(np.float32)

    if _TEXT_BACKEND == "sbert":
        embeddings = _sbert_model.encode(
            dataset,
            batch_size=min(32, len(dataset)) if dataset else 32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    # BERT-style mean pooling (giống build_reference._embed_texts_bert)
    inputs = _tokenizer(
        dataset,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length",
    ).to(device)
    with torch.no_grad():
        outputs = _bert_model(**inputs)
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = sum_embeddings / sum_mask
    embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
    return embeddings.cpu().numpy().astype(np.float32)


def check_text_drift(test_data: list, p_threshold: float, algorithm: str, ref_data: np.ndarray):
    embeddings = []
    batch_size = 32
    for i in range(0, len(test_data), batch_size):
        batch_questions = test_data[i : i + batch_size]
        embedding = extract_text_embedding(batch_questions)
        embeddings.append(embedding)
    embeddings = np.vstack(embeddings)
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


def check_text_drift_from_embeddings(embeddings: np.ndarray, p_threshold, algorithm, ref_data):
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
    print("hello")
