"""
MedVQA Drift Scenario Builder
==============================
Tạo các kịch bản drift có kiểm soát cho image và text embedding,
dùng để benchmark các thuật toán drift detection.

Cấu trúc output:
  drift_scenarios/
    image/
      no_drift.npy          # reference (copy)
      level1_mild.npy
      level2_moderate.npy
      level3_severe.npy
    text/
      no_drift.npy          # reference (copy)
      level1_mild.npy
      level2_moderate.npy
      level3_severe.npy
    metadata.yaml           # ghi lại params từng level để reproduce
"""

import os
import json
import yaml
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
import openai  # openai-compatible client dùng OpenRouter

# Load .env (chứa OPENROUTER_API_KEY)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")


# ─── Seed toàn cục để đảm bảo reproducibility ──────────────────────────────
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN 1: IMAGE AUGMENTATION FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def augment_image_level1(img: Image.Image, seed: int = 0) -> Image.Image:
    """
    Level 1 – Mild drift
    Mô phỏng: scanner calibration thay đổi nhẹ, patient positioning.
    - Brightness ±15%
    - Contrast ±15%
    - Rotation ±10° (fill_color=0 để giữ đúng histogram)
    - Horizontal flip (50% probability)
    """
    rng = np.random.RandomState(seed)

    # Brightness
    b_factor = 1.0 + rng.uniform(-0.15, 0.15)
    img = ImageEnhance.Brightness(img).enhance(b_factor)

    # Contrast
    c_factor = 1.0 + rng.uniform(-0.15, 0.15)
    img = ImageEnhance.Contrast(img).enhance(c_factor)

    # Rotation
    angle = rng.uniform(-10, 10)
    img = img.rotate(angle, expand=False, fillcolor=0)

    # Horizontal flip
    if rng.rand() > 0.5:
        img = ImageOps.mirror(img)

    return img


def augment_image_level2(img: Image.Image, seed: int = 0) -> Image.Image:
    """
    Level 2 – Moderate drift
    Mô phỏng: thiết bị chụp khác (CT vs X-ray noise profile),
               preprocessing pipeline khác nhau.
    - Gaussian blur radius=3
    - Gaussian noise σ=25 (trên pixel [0,255])
    - CLAHE cường mạnh (approximate via contrast cực đại)
    """
    rng = np.random.RandomState(seed)

    # Gaussian blur
    img = img.filter(ImageFilter.GaussianBlur(radius=3))

    # Gaussian noise
    arr = np.array(img).astype(np.float32)
    noise = rng.normal(0, 25, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # CLAHE approximation: aggressive contrast stretch
    img = ImageEnhance.Contrast(img).enhance(3.0)

    return img


def augment_image_level3(img: Image.Image, seed: int = 0, patch_ratio: float = 0.5) -> Image.Image:
    """
    Level 3 – Severe drift
    Mô phỏng: domain shift thực sự (grayscale invert giả lập negative film,
               synthetic noise patch che phủ lớn).
    - Invert grayscale
    - Random noise patch covering patch_ratio of the image area
    """
    rng = np.random.RandomState(seed)

    # Invert
    img = ImageOps.invert(img.convert("L")).convert(img.mode)

    # Noise patch
    arr = np.array(img).astype(np.float32)
    H, W = arr.shape[:2]
    ph = int(H * patch_ratio ** 0.5)
    pw = int(W * patch_ratio ** 0.5)
    y0 = rng.randint(0, H - ph)
    x0 = rng.randint(0, W - pw)
    arr[y0:y0+ph, x0:x0+pw] = rng.randint(0, 256, (ph, pw) + arr.shape[2:] if arr.ndim == 3 else (ph, pw))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    return img


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN 2: TEXT TRANSFORMATION FUNCTIONS (LLM-BASED)
# ════════════════════════════════════════════════════════════════════════════

class LLMTextDriftGenerator:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in .env file.")
        
        # Khởi tạo OpenAI client trỏ đến OpenRouter
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.model = "google/gemini-2.5-flash"

    def _call_llm(self, prompt: str, seed: int = 0) -> str:
        """Hàm gọi API chung có xử lý lỗi cơ bản"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant tasked with generating specific variations of text data. Output ONLY the generated question text, without any quotes or conversational filler."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                seed=seed, # Cố gắng giữ reproducibility cho LLM (tuỳ model có support hay không)
            )
            return response.choices[0].message.content.strip().strip('"').strip("'")
        except Exception as e:
            print(f"LLM API Error: {e}")
            return "error fallback question?"

    def transform_level1(self, question: str, seed: int = 0) -> str:
        """
        Level 1 – Mild drift: radiologist shorthand, informal clinical phrasing, lowercase.
        """
        prompt = (
            f"Rewrite the following medical question using extensive medical abbreviations (like CXR, pt, hx, etc.), "
            f"radiologist shorthand, and informal clinical phrasing. Convert everything to lowercase and remove ending punctuation.\n"
            f"Original: {question}"
        )
        return self._call_llm(prompt, seed)

    def transform_level2(self, question: str, seed: int = 0, mode: str = "cross_domain") -> str:
        """
        Level 2 – Moderate drift: cross_domain, verbose, hoặc multilingual.
        """
        if mode == "cross_domain":
            prompt = "Generate a short question asking about a dermatological, orthopedic, or neurological issue in an X-ray or medical image. Make it sound like a doctor asking a question."
        elif mode == "verbose":
            prompt = (
                f"Rewrite this medical question to be extremely verbose, overly formal, and unnecessarily wordy, "
                f"as if written by someone trying to sound very sophisticated.\n"
                f"Original: {question}"
            )
        elif mode == "multilingual":
            prompt = (
                f"Translate this medical question into either Spanish, French, or German (pick one randomly). "
                f"Keep the translation natural to that language.\n"
                f"Original: {question}"
            )
        else:
            return question

        return self._call_llm(prompt, seed)

    def transform_level3(self, question: str, seed: int = 0, mode: str = "off_topic") -> str:
        """
        Level 3 – Severe drift: completely out of distribution.
        """
        if mode == "off_topic":
            prompt = "Generate a completely random, casual question about weather, sports, food, or pop culture. It must not be related to medicine or healthcare."
        elif mode == "random_tokens":
            # Có thể dùng code python thông thường cho random_tokens cho nhanh vì không cần hiểu ngữ nghĩa
            rng = np.random.RandomState(seed)
            vocab = "abcdefghijklmnopqrstuvwxyz"
            return " ".join("".join(rng.choice(list(vocab), rng.randint(3, 9))) for _ in range(rng.randint(5, 12))) + "?"
        else:
            return question
            
        return self._call_llm(prompt, seed)

# Khởi tạo instance global
try:
    llm_generator = LLMTextDriftGenerator()
except ValueError as e:
    print(f"Warning: {e} - LLM-based text drift will fail if called.")
    llm_generator = None


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN 3: EMBEDDING EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

def embed_images(
    images: list,
    model_id: str,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """Trích xuất image embeddings theo đúng loại model."""
    all_embs = []
    
    if "BiomedCLIP" in model_id:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:' + model_id, device=device)
        model.eval()
        for i in tqdm(range(0, len(images), batch_size), desc="  Image embedding"):
            batch = images[i : i + batch_size]
            inputs = torch.stack([preprocess(img.convert("RGB")) for img in batch]).to(device)
            with torch.no_grad():
                features = model.encode_image(inputs)
                features = features / features.norm(p=2, dim=-1, keepdim=True)
            all_embs.append(features.cpu().numpy())
            
    elif "clip" in model_id.lower() or "biomedclip" in model_id.lower():
        # Generic CLIP models
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).to(device)
        model.eval()
        for i in tqdm(range(0, len(images), batch_size), desc="  Image embedding"):
            batch = images[i : i + batch_size]
            inputs = processor(images=[img.convert("RGB") for img in batch], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                features = model.get_image_features(**inputs)
                features = features / features.norm(p=2, dim=-1, keepdim=True)
            all_embs.append(features.cpu().numpy())
            
    else:
        # Generic Vision Models (DINO, ViT)
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()
        for i in tqdm(range(0, len(images), batch_size), desc="  Image embedding"):
            batch = images[i : i + batch_size]
            inputs = processor(images=[img.convert("RGB") for img in batch], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    features = outputs.pooler_output
                else:
                    features = outputs.last_hidden_state[:, 0, :]
                features = features / features.norm(p=2, dim=-1, keepdim=True)
            all_embs.append(features.cpu().numpy())

    return np.vstack(all_embs)


def embed_texts(
    texts: list,
    model_id: str,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Trích xuất text embeddings theo kiểu mean pooling giúp tương thích với cả SentenceBERT & BioBERT."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  Text embedding"):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length",
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
        all_embs.append(embeddings.cpu().numpy())

    return np.vstack(all_embs)


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN 4: MAIN BUILDER
# ════════════════════════════════════════════════════════════════════════════
def load_config(config_path: str = None) -> dict:
    """
    Đọc drift_config.yaml và trả về dict config đầy đủ.
    Mặc định tìm file tại: <repo_root>/configs/drift_config.yaml
    """
    if config_path is None:
        # Script nằm ở meddrift-ai-service/scripts/ → đi lên 3 cấp tới project root
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(script_dir, "configs", "drift_config.yaml")

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config tại: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"[Config] Loaded from: {config_path}")
    print(f"  image.encoder   : {config['image']['encoder']}")
    print(f"  text.encoder    : {config['text']['encoder']}")
    print(f"  reference.data  : {config['reference']['data']}")
    print(f"  reference.num_samples: {config['reference']['num_samples']}")
    return config

def get_safe_model_name(model_id: str) -> str:
    return model_id.replace('/', '--')

def build_drift_scenarios(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Lấy danh sách encoders
    image_encoders = config["image"].get("allow_encoder", [config["image"]["encoder"]])
    if config["image"]["encoder"] not in image_encoders:
        image_encoders.insert(0, config["image"]["encoder"])
        
    text_encoders = config["text"].get("allow_encoder", [config["text"]["encoder"]])
    if config["text"]["encoder"] not in text_encoders:
        text_encoders.insert(0, config["text"]["encoder"])

    dataset_path    = config["reference"]["data"]
    num_samples     = config["reference"]["num_samples"]

    # Script nằm ở meddrift-ai-service/scripts/ → thư mục gốc project
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_root = Path(script_dir) / "data" / config["reference"].get("drift_output_dir", "drift_scenarios")
    
    org_img_dir = out_root / "org" / "image"
    org_txt_dir = out_root / "org" / "text"
    org_img_dir.mkdir(parents=True, exist_ok=True)
    org_txt_dir.mkdir(parents=True, exist_ok=True)

    # Tạo thư mục emds cho từng encoder
    for enc in image_encoders:
        (out_root / "image" / get_safe_model_name(enc)).mkdir(parents=True, exist_ok=True)
    for enc in text_encoders:
        (out_root / "text" / get_safe_model_name(enc)).mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────
    print("\nLoading dataset...")
    dataset = load_dataset(dataset_path, split="train").select(range(num_samples))
    raw_images    = dataset["image"]
    raw_questions = dataset["question"]

    metadata = {"seed": GLOBAL_SEED, "num_samples": num_samples, "levels": {}}

    # ══════════════════════════════════════════════════════════════════════
    #  IMAGE SCENARIOS
    # ══════════════════════════════════════════════════════════════════════

    image_scenarios = {
        "no_drift": {
            "fn": None,
            "desc": "No augmentation applied – VQA-RAD original images",
        },
        "level1_mild": {
            "fn": lambda img, s: augment_image_level1(img, seed=s),
            "desc": "Brightness ±15%, contrast ±15%, rotation ±10°, hflip 50%",
        },
        "level2_moderate": {
            "fn": lambda img, s: augment_image_level2(img, seed=s),
            "desc": "Gaussian blur r=3, noise σ=25, CLAHE aggressive",
        },
        "level3_severe": {
            "fn": lambda img, s: augment_image_level3(img, seed=s, patch_ratio=0.5),
            "desc": "Grayscale invert + noise patch covering 50% area",
        },
    }

    for scenario_name, cfg in image_scenarios.items():
        print(f"\n[IMAGE] Generating: {scenario_name} ...")
        # 1. Augment images
        if cfg["fn"] is None:
            images_aug = raw_images
        else:
            images_aug = [
                cfg["fn"](img.convert("RGB"), GLOBAL_SEED + i)
                for i, img in enumerate(tqdm(raw_images, desc="  Augmenting"))
            ]
        
        # 2. Save augmented images to org/
        scene_img_dir = org_img_dir / scenario_name
        scene_img_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(tqdm(images_aug, desc="  Saving images to org/")):
            img.convert("RGB").save(scene_img_dir / f"{i:04d}.jpg")

        # 3. Extract embeddings for all allowed encoders
        for enc in image_encoders:
            print(f"  -> Extracting embeddings for model: {enc}")
            embs = embed_images(images_aug, enc, device)
            safe_enc = get_safe_model_name(enc)
            save_path = out_root / "image" / safe_enc / f"{scenario_name}.npy"
            np.save(save_path, embs)
            print(f"    Saved {save_path} | shape: {embs.shape}")
        
        metadata["levels"][f"image_{scenario_name}"] = {"desc": cfg["desc"], "samples_count": len(images_aug)}

    # ══════════════════════════════════════════════════════════════════════
    #  TEXT SCENARIOS
    # ══════════════════════════════════════════════════════════════════════

    text_scenarios = {
        "no_drift": {
            "fn": None,
            "desc": "No transformation – VQA-RAD original questions",
        },
        "level1_mild": {
            "fn": lambda q, s: llm_generator.transform_level1(q, seed=s),
            "desc": "LLM Generated: Medical abbreviation substitution + lowercase",
        },
        "level2_moderate_cross": {
            "fn": lambda q, s: llm_generator.transform_level2(q, seed=s, mode="cross_domain"),
            "desc": "LLM Generated: Cross-domain medical questions (non-radiology)",
        },
        "level2_moderate_verbose": {
            "fn": lambda q, s: llm_generator.transform_level2(q, seed=s, mode="verbose"),
            "desc": "LLM Generated: Verbose clinical phrasing",
        },
        "level2_moderate_multilingual": {
            "fn": lambda q, s: llm_generator.transform_level2(q, seed=s, mode="multilingual"),
            "desc": "LLM Generated: Question translated to foreign languages",
        },
        "level3_severe_offtopic": {
            "fn": lambda q, s: llm_generator.transform_level3(q, seed=s, mode="off_topic"),
            "desc": "LLM Generated: Completely off-topic non-medical questions",
        },
        "level3_severe_random": {
            "fn": lambda q, s: llm_generator.transform_level3(q, seed=s, mode="random_tokens"),
            "desc": "Random token sequences – nonsense strings",
        },
    }

    for scenario_name, cfg in text_scenarios.items():
        print(f"\n[TEXT] Generating: {scenario_name} ...")
        # 1. Transform texts
        if cfg["fn"] is None:
            questions_aug = list(raw_questions)
        else:
            questions_aug = [
                cfg["fn"](q, GLOBAL_SEED + i)
                for i, q in enumerate(tqdm(raw_questions, desc="  Transforming"))
            ]

        # 2. Save augmented texts to org/
        with open(org_txt_dir / f"{scenario_name}.json", "w", encoding="utf-8") as f:
            json.dump(questions_aug, f, ensure_ascii=False, indent=2)
            
        print(f"  Sample[0]: {questions_aug[0]}")
        print(f"  Sample[1]: {questions_aug[1]}")

        # 3. Extract embeddings for all allowed encoders
        for enc in text_encoders:
            print(f"  -> Extracting embeddings for model: {enc}")
            embs = embed_texts(questions_aug, enc, device)
            safe_enc = get_safe_model_name(enc)
            save_path = out_root / "text" / safe_enc / f"{scenario_name}.npy"
            np.save(save_path, embs)
            print(f"    Saved {save_path} | shape: {embs.shape}")
            
        metadata["levels"][f"text_{scenario_name}"] = {"desc": cfg["desc"], "samples_count": len(questions_aug)}

    # ── Lưu metadata ──────────────────────────────────────────────────────
    meta_path = out_root / "metadata.yaml"
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.dump(metadata, f, allow_unicode=True, default_flow_style=False)
    print(f"\nMetadata saved to {meta_path}")
    print("\nDone! All drift scenarios generated.")


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN 5: EMBEDDING-SPACE DRIFT (KHÔNG CẦN DATA GỐC)
#  Dùng khi muốn test nhanh detector mà không cần re-embed toàn bộ dataset
# ════════════════════════════════════════════════════════════════════════════

def build_synthetic_embedding_drift(
    reference_path: str,
    out_dir: str,
    seed: int = GLOBAL_SEED,
):
    """
    Tạo drifted embeddings trực tiếp trong embedding space, không cần dữ liệu gốc.
    Hữu ích để test nhanh detector hoặc khi không có GPU.

    Các phép biến đổi:
      Level 1: Gaussian noise nhỏ (μ=0, σ=0.02)
      Level 2: Gaussian noise trung bình + mean shift (μ=0.5)
      Level 3: Mean shift lớn (μ=3.0) + covariance perturbation
    """
    rng = np.random.RandomState(seed)
    ref = np.load(reference_path)  # shape [N, D]
    N, D = ref.shape
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reference shape: {ref.shape}")

    # No drift
    np.save(out / "no_drift.npy", ref)

    # Level 1 – mild
    noise_l1 = rng.normal(0, 0.02, ref.shape)
    l1 = ref + noise_l1
    np.save(out / "level1_mild.npy", l1)
    print(f"Level 1 mean cosine shift: {np.mean(1 - np.sum(ref * l1, axis=1)):.6f}")

    # Level 2 – moderate
    mean_shift_l2 = rng.normal(0, 0.5, D)
    noise_l2 = rng.normal(0, 0.1, ref.shape)
    l2 = ref + noise_l2 + mean_shift_l2
    np.save(out / "level2_moderate.npy", l2)
    print(f"Level 2 mean shift magnitude: {np.linalg.norm(mean_shift_l2):.4f}")

    # Level 3 – severe (shift + rotation-like perturbation)
    mean_shift_l3 = rng.normal(0, 3.0, D)
    # Random projection để tạo covariance perturbation
    P = rng.normal(0, 1, (D, D))
    P = P / np.linalg.norm(P, axis=0, keepdims=True)
    l3 = ref @ P.T * 0.3 + mean_shift_l3
    np.save(out / "level3_severe.npy", l3)
    print(f"Level 3 mean shift magnitude: {np.linalg.norm(mean_shift_l3):.4f}")

    print(f"\nSynthetic embeddings saved to {out}")


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN 6: UTILITY – QUICK STATS CHO BENCHMARK
# ════════════════════════════════════════════════════════════════════════════

def print_drift_stats(ref_path: str, drifted_paths: dict):
    """
    In thống kê nhanh để verify drift intensity.
    Dùng trước khi chạy detector chính thức.
    """
    ref = np.load(ref_path)

    print(f"\n{'='*60}")
    print(f"{'Scenario':<30} {'MMD↑':>8} {'Mean shift':>12} {'Std ratio':>10}")
    print(f"{'─'*60}")

    ref_mean = ref.mean(axis=0)
    ref_std  = ref.std(axis=0).mean()

    for name, path in drifted_paths.items():
        arr = np.load(path)
        shift = np.linalg.norm(arr.mean(axis=0) - ref_mean)
        std_r = arr.std(axis=0).mean() / ref_std
        # MMD unbiased estimator (quick approximation)
        n = min(500, len(ref), len(arr))
        ix = np.random.choice(len(ref), n, replace=False)
        iy = np.random.choice(len(arr), n, replace=False)
        rx, ry = ref[ix], arr[iy]
        k_xx = np.exp(-np.sum((rx[:, None] - rx[None, :]) ** 2, axis=-1) / (2 * ref.shape[1]))
        k_yy = np.exp(-np.sum((ry[:, None] - ry[None, :]) ** 2, axis=-1) / (2 * ref.shape[1]))
        k_xy = np.exp(-np.sum((rx[:, None] - ry[None, :]) ** 2, axis=-1) / (2 * ref.shape[1]))
        mmd  = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
        print(f"{name:<30} {mmd:>8.5f} {shift:>12.4f} {std_r:>10.4f}")

    print(f"{'='*60}\n")


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── Load config từ drift_config.yaml ─────────────────────────────────
    config = load_config()  # tự động tìm <repo_root>/configs/drift_config.yaml
    # Hoặc truyền đường dẫn tường minh:
    # config = load_config("path/to/drift_config.yaml")

    # ── Option A: Full pipeline (cần GPU + dataset) ───────────────────────
    build_drift_scenarios(config)

    # ── Option B: Synthetic embedding drift (test nhanh, không cần GPU) ──
    # ref_image_dir = config["reference"]["data_dir"]
    # ref_image_path = str(Path(ref_image_dir) / config["reference"]["image_file"])
    # build_synthetic_embedding_drift(
    #     reference_path=ref_image_path,
    #     out_dir="drift_scenarios/image_synthetic",
    # )

    # ── Verify stats (bỏ comment Option B trước khi dùng) ────────────────
    # ref_image_path = str(
    #     Path(config["reference"]["data_dir"]) / config["reference"]["image_file"]
    # )
    # print_drift_stats(
    #     ref_path=ref_image_path,
    #     drifted_paths={
    #         "no_drift":        "drift_scenarios/image_synthetic/no_drift.npy",
    #         "level1_mild":     "drift_scenarios/image_synthetic/level1_mild.npy",
    #         "level2_moderate": "drift_scenarios/image_synthetic/level2_moderate.npy",
    #         "level3_severe":   "drift_scenarios/image_synthetic/level3_severe.npy",
    #     },
    # )