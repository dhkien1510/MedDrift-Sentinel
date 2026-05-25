import numpy as np
import os
import yaml
import json
import torch
from PIL import Image
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
from tqdm import tqdm

# ─────────────────────────────────────────────
# PHÂN LOẠI MODEL — dùng để chọn đúng loader
# ─────────────────────────────────────────────
# Image encoders
CLIP_MODELS    = {"openai/clip-vit-base-patch32"}           # dùng CLIPProcessor + CLIPModel
BIOMEDCLIP_MODELS = {                                        # dùng open_clip (hf-hub:)
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
}
DINOV2_MODELS  = {                                           # dùng Dinov2Model trực tiếp
    "microsoft/rad-dino",
    "microsoft/rad-dino-maira-2",
    "facebook/dinov2-base",
}
VIT_MODELS     = {"google/vit-base-patch16-224"}             # dùng AutoModel thường

# Text encoders
BERT_MODELS    = {                                           # dùng AutoTokenizer + AutoModel (mean pool)
    "dmis-lab/biobert-v1.1",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
}
SBERT_MODELS   = {                                           # dùng SentenceTransformer
    "NeuML/pubmedbert-base-embeddings",
    "NeuML/pubmedbert-base-embeddings-matryoshka",
    "pritamdeka/S-PubMedBert-MS-MARCO",
}
MODEL2VEC_MODELS = {                                         # dùng model2vec StaticModel
    "NeuML/pubmedbert-base-embeddings-8M",
}

# ─────────────────────────────────────────────

def load_config():
    current_script_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_script_path)
    service_dir = os.path.dirname(current_dir)
    root_dir = os.path.dirname(service_dir)
    config_dir = os.path.join(root_dir, 'configs/drift_config.yaml')
    print("Tiến hành đọc tập tin drift_config.yaml...")
    try:
        with open(config_dir, "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config, root_dir
    except Exception as e:
        print(f"Lỗi: {e}")
        return None, None

config, root_dir = load_config()

def get_safe_model_name(model_id: str) -> str:
    return model_id.replace('/', '--')

# ═══════════════════════════════════════════════════════════════
# IMAGE EMBEDDING FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _embed_images_clip(images, model_id, device, batch_size=16):
    """
    Dùng cho: openai/clip-vit-base-patch32
    Load: CLIPProcessor + CLIPModel
    """
    print(f"  → Loader: CLIPProcessor + CLIPModel")
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()

    embedded_list = []
    for i in tqdm(range(0, len(images), batch_size), desc=f"[CLIP] {model_id}"):
        batch = [img.convert("RGB") for img in images[i:i+batch_size]]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        embedded_list.append(features.cpu().numpy())

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return np.vstack(embedded_list)


def _embed_images_biomedclip(images, model_id, device, batch_size=16):
    """
    Dùng cho: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
    Load: open_clip với prefix hf-hub:
    QUAN TRỌNG: không dùng AutoModel — model này không có config model_type chuẩn
    """
    print(f"  → Loader: open_clip (hf-hub:)")
    import open_clip
    model, _, preprocess = open_clip.create_model_from_pretrained(
        f"hf-hub:{model_id}", device=device
    )
    model.eval()

    embedded_list = []
    for i in tqdm(range(0, len(images), batch_size), desc=f"[BiomedCLIP] {model_id}"):
        batch = torch.stack([
            preprocess(img.convert("RGB")) for img in images[i:i+batch_size]
        ]).to(device)
        with torch.no_grad():
            features = model.encode_image(batch)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        embedded_list.append(features.cpu().numpy())

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return np.vstack(embedded_list)


def _embed_images_dinov2(images, model_id, device, batch_size=16):
    """
    Dùng cho: microsoft/rad-dino, microsoft/rad-dino-maira-2, facebook/dinov2-base
    Load: Dinov2Model trực tiếp (KHÔNG dùng AutoModel)
    Lý do: config.json có model_type="dinov2" nhưng AutoModel không tự resolve
            → OSError: Can't load the model
    Output: last_hidden_state[:, 0, :] = CLS token (pooler_output có thể chưa init)
    """
    print(f"  → Loader: Dinov2Model (direct import)")
    from transformers import Dinov2Model, AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(model_id)
    # Dùng Dinov2Model trực tiếp, KHÔNG dùng AutoModel.from_pretrained
    model = Dinov2Model.from_pretrained(model_id).to(device)
    model.eval()

    embedded_list = []
    for i in tqdm(range(0, len(images), batch_size), desc=f"[DINOv2] {model_id}"):
        batch = [img.convert("RGB") for img in images[i:i+batch_size]]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Dùng CLS token thay vì pooler_output
            # (pooler_output của DINOv2 là newly initialized — không đáng tin cậy)
            features = outputs.last_hidden_state[:, 0, :]
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        embedded_list.append(features.cpu().numpy())

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return np.vstack(embedded_list)


def _embed_images_vit(images, model_id, device, batch_size=16):
    """
    Dùng cho: google/vit-base-patch16-224
    Load: AutoModel thông thường (model_type="vit" — AutoModel resolve được)
    """
    print(f"  → Loader: AutoModel (standard ViT)")
    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    embedded_list = []
    for i in tqdm(range(0, len(images), batch_size), desc=f"[ViT] {model_id}"):
        batch = [img.convert("RGB") for img in images[i:i+batch_size]]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # pooler_output cho ViT là linear trên CLS token — đã được init đúng
            if outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                features = outputs.last_hidden_state[:, 0, :]
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        embedded_list.append(features.cpu().numpy())

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return np.vstack(embedded_list)


def embed_images(images, model_id, device, batch_size=16):
    """
    Router: chọn đúng loader theo model_id
    Đây là nguyên nhân chính gây OSError — mỗi model cần loader riêng
    """
    if model_id in BIOMEDCLIP_MODELS:
        return _embed_images_biomedclip(images, model_id, device, batch_size)
    elif model_id in CLIP_MODELS:
        return _embed_images_clip(images, model_id, device, batch_size)
    elif model_id in DINOV2_MODELS:
        print("DINOV2_MODELS")
        return _embed_images_dinov2(images, model_id, device, batch_size)
    elif model_id in VIT_MODELS:
        return _embed_images_vit(images, model_id, device, batch_size)
    else:
        # Fallback: thử AutoModel — có thể lỗi nếu model_type không được map
        print(f"  → Loader: AutoModel (fallback — có thể lỗi nếu model_type lạ)")
        return _embed_images_vit(images, model_id, device, batch_size)


# ═══════════════════════════════════════════════════════════════
# TEXT EMBEDDING FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _embed_texts_bert(texts, model_id, device, batch_size=32):
    """
    Dùng cho: BioBERT, BiomedBERT
    Mean pooling trên last_hidden_state (attention mask weighted)
    """
    print(f"  → Loader: AutoTokenizer + AutoModel (BERT mean pool)")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"[BERT] {model_id}"):
        inputs = tokenizer(
            texts[i:i+batch_size],
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        all_embeddings.append(embeddings.cpu().numpy())

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return np.vstack(all_embeddings)


def _embed_texts_sbert(texts, model_id, batch_size=32):
    """
    Dùng cho: NeuML/pubmedbert-base-embeddings, S-PubMedBert-MS-MARCO
    SentenceTransformer tự xử lý pooling — không cần mean pool thủ công
    KHÔNG truyền device vào SentenceTransformer để tương thích CPU
    """
    print(f"  → Loader: SentenceTransformer")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True  # L2 normalize built-in
    )
    del model
    return embeddings


def _embed_texts_model2vec(texts, model_id):
    """
    Dùng cho: NeuML/pubmedbert-base-embeddings-8M
    model2vec StaticModel — không phải AutoModel/AutoTokenizer
    KHÔNG dùng AutoTokenizer.from_pretrained — sẽ lỗi vì khác architecture
    """
    print(f"  → Loader: model2vec StaticModel")
    from model2vec import StaticModel
    model = StaticModel.from_pretrained(model_id)
    embeddings = model.encode(texts)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-9, None)
    del model
    return embeddings


def embed_texts(texts, model_id, device, batch_size=32):
    """
    Router: chọn đúng loader theo model_id
    """
    if model_id in MODEL2VEC_MODELS:
        return _embed_texts_model2vec(texts, model_id)
    elif model_id in SBERT_MODELS:
        return _embed_texts_sbert(texts, model_id, batch_size)
    elif model_id in BERT_MODELS:
        return _embed_texts_bert(texts, model_id, device, batch_size)
    else:
        # Fallback: thử BERT mean pool
        print(f"  → Loader: AutoModel (fallback BERT-style)")
        return _embed_texts_bert(texts, model_id, device, batch_size)


# ═══════════════════════════════════════════════════════════════
# BUILD FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def build_image_reference(model_id = "", auto = False):
    if not config: return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")

    num_samples = config['reference']['num_samples']
    dataset_path = config['reference']['data']

    # Load dataset
    dataset = load_dataset(dataset_path, split="train")
    dataset = dataset.select(range(num_samples))
    images = dataset['image']

    # Lưu ảnh gốc
    # org_image_dir = os.path.join(root_dir, config['reference']['data_dir'], 'org', 'image')
    # os.makedirs(org_image_dir, exist_ok=True)
    # print("Lưu ảnh gốc vào thư mục org...")
    # for idx, img in enumerate(tqdm(images, desc="Saving org images")):
    #     img.convert("RGB").save(os.path.join(org_image_dir, f"{idx:04d}.jpg"))


    if auto:
        # Danh sách encoder
        encoders = config['image'].get('allow_encoder', [config['image']['encoder']])
        # Lọc bỏ comment (các entry None từ YAML comment)
        encoders = [e for e in encoders if e is not None]
        if config['image']['encoder'] not in encoders:
            encoders.insert(0, config['image']['encoder'])
    else:
        encoders = [model_id]
    
  
    

    # Embed từng encoder
    for model_id in encoders:
        print(f"\n{'='*60}")
        print(f"[IMAGE] Encoder: {model_id}")
        print(f"{'='*60}")
      
        try:
            embeddings = embed_images(images, model_id, device)

            safe_model_id = get_safe_model_name(model_id)
            model_save_dir = os.path.join(
                root_dir, config['reference']['data_dir'], 'image', safe_model_id
            )
            os.makedirs(model_save_dir, exist_ok=True)
            save_path = os.path.join(model_save_dir, config['reference']['image_file'])
            np.save(save_path, embeddings)

            print(f"✓ Đã lưu: {save_path}")
            print(f"  Shape: {embeddings.shape}")
        except Exception as e:
            print(f"✗ LỖI với {model_id}: {e}")
            import traceback
            traceback.print_exc()


def build_questions_reference():
    if not config: return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")

    num_samples = config['reference']['num_samples']
    dataset_path = config['reference']['data']

    # Load dataset
    dataset = load_dataset(dataset_path, split="train").select(range(num_samples))
    questions = list(dataset['question'])

    # Lưu text gốc
    org_text_dir = os.path.join(root_dir, config['reference']['data_dir'], 'org', 'text')
    os.makedirs(org_text_dir, exist_ok=True)
    org_text_path = os.path.join(org_text_dir, 'questions.json')
    print("Lưu text gốc vào thư mục org...")
    with open(org_text_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    # Danh sách encoder
    encoders = config['text'].get('allow_encoder', [config['text']['encoder']])
    encoders = [e for e in encoders if e is not None]
    if config['text']['encoder'] not in encoders:
        encoders.insert(0, config['text']['encoder'])

    # Embed từng encoder
    for model_id in encoders:
        print(f"\n{'='*60}")
        print(f"[TEXT] Encoder: {model_id}")
        print(f"{'='*60}")
        try:
            embeddings = embed_texts(questions, model_id, device)

            safe_model_id = get_safe_model_name(model_id)
            model_save_dir = os.path.join(
                root_dir, config['reference']['data_dir'], 'text', safe_model_id
            )
            os.makedirs(model_save_dir, exist_ok=True)
            save_path = os.path.join(model_save_dir, config['reference']['text_file'])
            np.save(save_path, embeddings)

            print(f"✓ Đã lưu: {save_path}")
            print(f"  Shape: {embeddings.shape}")
        except Exception as e:
            print(f"✗ LỖI với {model_id}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    if config:
        print("\n" + "="*60)
        print("XÂY DỰNG IMAGE REFERENCE")
        print("="*60)
        build_image_reference("microsoft/rad-dino-maira-2")

        # print("\n" + "="*60)
        # print("XÂY DỰNG TEXT REFERENCE")
        # print("="*60)
        # build_questions_reference()