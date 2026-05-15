"""
drift_agent.py — LangChain Drift Sentinel Agent
=================================================
[PATCH] Buffer dùng Redis thay vì RAM list.
  Trước: _image_embedding_buffer = []  → mất khi container restart/crash
  Sau:   buffer_push / buffer_flush từ db/redis_client.py → survive restart

Các thay đổi:
  - add_to_buffer()   → gọi redis_client.buffer_push()
  - is_buffer_ready() → gọi redis_client.buffer_is_ready()
  - get_buffer_count()→ gọi redis_client.buffer_count()
  - flush_buffer()    → gọi redis_client.buffer_flush()
  - _image_embedding_buffer / _text_embedding_buffer giữ lại dưới dạng
    property function để backward-compat với main.py (visualization endpoint)
    nhưng chỉ dùng cho read-only PCA preview, không phải source of truth.

Tất cả tool logic (check_image_drift_tool, check_text_drift_tool,
check_dual_drift_tool, run_drift_sentinel, run_drift_sentinel_batch)
KHÔNG thay đổi — chỉ thay đổi phần buffer management.
"""

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter
from langgraph.prebuilt import create_react_agent
from registry import load_config, load_config_meta
from image_drift import check_image_drift, check_image_drift_from_embeddings
from text_drift import check_text_drift, check_text_drift_from_embeddings
from multimodal_drift import multimodal_drift_report
import numpy as np
import contextvars

# [PATCH] Import Redis buffer helpers
from db.redis_client import (
    buffer_push,
    buffer_count,
    buffer_is_ready,
    buffer_flush,
    buffer_clear,
)

load_dotenv()


# ============================================================
# CONTEXT: Lưu input data trước khi Agent chạy (không đổi)
# ============================================================
_current_images: contextvars.ContextVar    = contextvars.ContextVar('_current_images',    default=None)
_current_questions: contextvars.ContextVar = contextvars.ContextVar('_current_questions', default=None)
_context_mode: contextvars.ContextVar      = contextvars.ContextVar('_context_mode',      default='raw')

def set_context(images, questions, mode='raw'):
    _current_images.set(images)
    _current_questions.set(questions)
    _context_mode.set(mode)


# ============================================================
# BUFFER THRESHOLD
# ============================================================
config = load_config_meta(isImage=True)
BUFFER_THRESHOLD = config.get("threshold", 100) if config else 100


# ============================================================
# BUFFER API — delegate to Redis
# [PATCH] Thay thế hoàn toàn RAM list bằng Redis
# ============================================================

def add_to_buffer(image_embedding: np.ndarray, text_embedding: np.ndarray) -> None:
    """
    Thêm 1 cặp embedding vào Redis buffer (persist qua restart).
    Gọi từ FastAPI /api/drift/collect mỗi request.
    """
    buffer_push(image_embedding, text_embedding)


def is_buffer_ready() -> bool:
    """True nếu buffer đã tích lũy đủ BUFFER_THRESHOLD mẫu."""
    return buffer_is_ready(BUFFER_THRESHOLD)


def get_buffer_count() -> int:
    """Trả về số lượng embeddings hiện có trong Redis buffer."""
    return buffer_count()


def flush_buffer() -> tuple[np.ndarray, np.ndarray]:
    """
    Lấy toàn bộ embeddings ra khỏi Redis buffer và xóa buffer.

    Returns:
        (img_batch np.ndarray (N, D_img), txt_batch np.ndarray (N, D_txt))
    """
    return buffer_flush()


# ============================================================
# BACKWARD COMPAT: _image/text_embedding_buffer cho visualization
# main.py dùng _image_embedding_buffer trong /api/drift/visualization
# để preview PCA. Đây là read-only — không phải source of truth.
# [PATCH] Chuyển thành lazy-load từ Redis khi được truy cập.
# ============================================================

class _RedisBufferProxy:
    """
    Proxy object giả lập list interface để backward-compat với code cũ.
    Đọc dữ liệu từ Redis buffer khi cần, không cache trong RAM.
    """
    def __init__(self, key_fn):
        self._key_fn = key_fn

    def __len__(self):
        return buffer_count()

    def __iter__(self):
        # Chỉ dùng cho visualization — load toàn bộ ra RAM tạm thời
        from db.redis_client import redis_sync_client, KEY_BUFFER_IMAGE, KEY_BUFFER_TEXT, _bytes_to_np
        key = self._key_fn()
        raw = redis_sync_client.lrange(key, 0, -1)
        return iter([_bytes_to_np(b) for b in raw])

    def __bool__(self):
        return buffer_count() > 0


# Import để backward compat (main.py dùng: from drift_agent import _image_embedding_buffer)
from db.redis_client import KEY_BUFFER_IMAGE, KEY_BUFFER_TEXT
_image_embedding_buffer = _RedisBufferProxy(lambda: KEY_BUFFER_IMAGE)
_text_embedding_buffer  = _RedisBufferProxy(lambda: KEY_BUFFER_TEXT)


# ============================================================
# TOOL 1: Image Drift Detection (không đổi)
# ============================================================
@tool
def check_image_drift_tool(image_description: str) -> str:
    """
    Run statistical drift test on medical image embeddings.
    Uses CLIP ViT-B/32 to extract image features, then compares against
    reference chest X-ray distribution using the configured algorithm (MMD/KS/LSDD).
    Call this tool to check if uploaded images are within the expected distribution.
    Returns: drift status, p-value, and statistical distance.
    """
    images = _current_images.get()
    mode   = _context_mode.get()
    if images is None:
        return "No images provided. Please upload images first."

    config = load_config(isImage=True)
    if mode == 'batch':
        result = check_image_drift_from_embeddings(
            images, float(config['p_threshold']), config['algorithm'], config['ref_embeddings']
        )
    else:
        result = check_image_drift(
            images, float(config['p_threshold']), config['algorithm'], config['ref_embeddings']
        )

    return (
        f"Image Drift Result:\n"
        f"  - Algorithm: {config['algorithm']}\n"
        f"  - Distance (score): {result['distance']:.6f}\n"
        f"  - P-value: {result['p_value']:.6f}\n"
        f"  - Is Drifted (p < {config['p_threshold']}): {result['is_drift']}\n"
        f"  - Interpretation: {'Distribution shift detected in images' if result['is_drift'] else 'Images within expected distribution'}"
    )


# ============================================================
# TOOL 2: Text Drift Detection (không đổi)
# ============================================================
@tool
def check_text_drift_tool(text_description: str) -> str:
    """
    Run statistical drift test on medical question embeddings.
    Uses BioBERT to extract text features, then compares against
    reference radiology question distribution using the configured algorithm.
    Call this tool to check if the doctor's question is within medical domain.
    Returns: drift status, p-value, and statistical distance.
    """
    questions = _current_questions.get()
    mode      = _context_mode.get()
    if questions is None:
        return "No questions provided. Please provide a question first."

    config = load_config(isImage=False)
    if mode == 'batch':
        result = check_text_drift_from_embeddings(
            questions, config['p_threshold'], config['algorithm'], config['ref_embeddings']
        )
    else:
        result = check_text_drift(
            questions, config['p_threshold'], config['algorithm'], config['ref_embeddings']
        )

    return (
        f"Text Drift Result:\n"
        f"  - Algorithm: {config['algorithm']}\n"
        f"  - Distance (score): {result['distance']:.6f}\n"
        f"  - P-value: {result['p_value']:.6f}\n"
        f"  - Is Drifted (p < {config['p_threshold']}): {result['is_drift']}\n"
        f"  - Interpretation: {'Question is outside medical radiology domain' if result['is_drift'] else 'Question within expected medical domain'}"
    )


# ============================================================
# TOOL 3: Dual Drift Check + Severity Report (không đổi)
# ============================================================
@tool
def check_dual_drift_tool(analysis_request: str) -> str:
    """
    Run BOTH image and text drift detection simultaneously, then generate
    a severity-classified drift report. This is the comprehensive drift check
    that combines both modalities.
    Use this tool when you need a complete drift analysis with severity level.
    Returns: combined drift status, severity level, and recommended actions.
    """
    images    = _current_images.get()
    questions = _current_questions.get()
    mode      = _context_mode.get()
    if images is None or questions is None:
        return "Error: Missing context. Both images and questions must be provided."

    img_config = load_config(isImage=True)
    txt_config = load_config(isImage=False)

    if mode == 'batch':
        img_result = check_image_drift_from_embeddings(
            images, img_config['p_threshold'], img_config['algorithm'], img_config['ref_embeddings']
        )
        txt_result = check_text_drift_from_embeddings(
            questions, txt_config['p_threshold'], txt_config['algorithm'], txt_config['ref_embeddings']
        )
    else:
        img_result = check_image_drift(
            images, img_config['p_threshold'], img_config['algorithm'], img_config['ref_embeddings']
        )
        txt_result = check_text_drift(
            questions, txt_config['p_threshold'], txt_config['algorithm'], txt_config['ref_embeddings']
        )

    img_drifted = img_result['is_drift']
    txt_drifted = txt_result['is_drift']

    if img_drifted and txt_drifted:
        drift_source = "both"
        severity     = "🔴 CRITICAL"
        action       = "REJECT input — both image and question are outside training distribution. VQA results are unreliable."
    elif img_drifted:
        drift_source = "image"
        severity     = "🟠 HIGH"
        action       = "WARNING — Image may not be a standard chest X-ray. VQA accuracy is compromised."
    elif txt_drifted:
        drift_source = "text"
        severity     = "🟡 MEDIUM"
        action       = "WARNING — Question is outside radiology domain. Consider rephrasing as a medical question."
    else:
        drift_source = "none"
        severity     = "🟢 NORMAL"
        action       = "System operating normally. VQA results are reliable."

    return (
        f"═══ DUAL DRIFT REPORT ═══\n"
        f"Severity: {severity}\n"
        f"Drift Source: {drift_source}\n"
        f"\n"
        f"Image Drift:\n"
        f"  - Drifted: {img_drifted}\n"
        f"  - P-value: {img_result['p_value']:.6f}\n"
        f"  - Distance: {img_result['distance']:.6f}\n"
        f"  - Algorithm: {img_config['algorithm']}\n"
        f"\n"
        f"Text Drift:\n"
        f"  - Drifted: {txt_drifted}\n"
        f"  - P-value: {txt_result['p_value']:.6f}\n"
        f"  - Distance: {txt_result['distance']:.6f}\n"
        f"  - Algorithm: {txt_config['algorithm']}\n"
        f"\n"
        f"Recommended Action: {action}\n"
        f"══════════════════════════"
    )


DRIFT_TOOLS = [
    check_image_drift_tool,
    check_text_drift_tool,
    check_dual_drift_tool,
]

SYSTEM_PROMPT = """You are the MedDrift Sentinel Agent — an AI safety monitor for a medical VQA system.
Your job is to detect and analyze data drift in multimodal inputs (medical images + questions).

## Your Workflow:
1. Use check_dual_drift_tool to run a comprehensive drift analysis on both image and text
2. Based on the results, provide a clear explanation of what happened
3. If drift is detected, explain WHY it might have occurred and what the doctor should do
4. Always respond in a structured format with severity, explanation, and recommended actions

## Important Rules:
- A p-value < 0.05 means statistically significant drift
- Lower p-value = stronger evidence of drift
- Image drift suggests the image modality differs from training distribution (e.g., CT scan instead of chest X-ray)
- Text drift suggests the question is outside the medical radiology domain
- BOTH modalities drifting is CRITICAL — the entire input is out-of-distribution
- Provide your final answer in Vietnamese for the doctor
"""

llm = ChatOpenRouter(model="google/gemini-2.5-flash-lite", temperature=0)
agent_executor = create_react_agent(llm, DRIFT_TOOLS, prompt=SYSTEM_PROMPT)


# ============================================================
# ENTRY POINT 1: Per-request mode (không đổi)
# ============================================================
def run_drift_sentinel(question: str, images: list, image_description: str = "uploaded medical image"):
    set_context(images=images, questions=[question], mode='raw')
    user_message = (
        f"Please analyze drift for the following input:\n"
        f"- Doctor's Question: {question}\n"
        f"- Image: {image_description}\n"
        f"Run the appropriate drift detection tools and provide your analysis."
    )
    try:
        result      = agent_executor.invoke({"messages": [{"role": "user", "content": user_message}]})
        final_msg   = result["messages"][-1]
        return final_msg.content
    except Exception as e:
        return f"Drift Agent Error: {str(e)}\n"


# ============================================================
# ENTRY POINT 2: Batch mode (không đổi)
# ============================================================
def run_drift_sentinel_batch(image_embeddings: np.ndarray, text_embeddings: np.ndarray) -> dict:
    set_context(images=image_embeddings, questions=text_embeddings, mode='batch')
    user_message = (
        f"Analyze drift for a batch of {len(image_embeddings)} samples.\n"
        f"These are pre-extracted embeddings collected over the last "
        f"{len(image_embeddings)} VQA requests.\n"
        f"Run the drift detection tools and provide your analysis."
    )
    try:
        result     = agent_executor.invoke({"messages": [{"role": "user", "content": user_message}]})
        agent_text = result["messages"][-1].content
    except Exception as e:
        agent_text = f"Drift Agent Error: {str(e)}"

    img_config = load_config(isImage=True)
    txt_config = load_config(isImage=False)
    if img_config is None or txt_config is None:
        parts = []
        if img_config is None:
            parts.append("image reference .npy")
        if txt_config is None:
            parts.append("text reference .npy")
        return {
            "severity": "⚪ ERROR", "drift_source": "config", "alert": True, "error": True,
            "message": "Không load được reference embeddings: " + "; ".join(parts),
            "image_drift": {}, "text_drift": {}, "agent_analysis": agent_text,
            "samples_analyzed": len(image_embeddings),
        }

    img_result = check_image_drift_from_embeddings(
        image_embeddings, img_config['p_threshold'], img_config['algorithm'], img_config['ref_embeddings']
    )
    txt_result = check_text_drift_from_embeddings(
        text_embeddings, txt_config['p_threshold'], txt_config['algorithm'], txt_config['ref_embeddings']
    )

    img_drifted = img_result['is_drift']
    txt_drifted = txt_result['is_drift']

    mm_report  = multimodal_drift_report(image_embeddings, text_embeddings)
    mm_drifted = bool(mm_report and mm_report.get("drift_ran") and mm_report.get("is_drift"))

    if img_drifted and txt_drifted:
        severity, drift_source = "🔴 CRITICAL", "both"
    elif img_drifted:
        severity, drift_source = "🟠 HIGH", "image"
    elif txt_drifted:
        severity, drift_source = "🟡 MEDIUM", "text"
    else:
        severity, drift_source = "🟢 NORMAL", "none"

    out = {
        "severity":         severity,
        "drift_source":     drift_source,
        "alert":            img_drifted or txt_drifted or mm_drifted,
        "image_drift":      {**img_result, "algorithm": img_config['algorithm']},
        "text_drift":       {**txt_result, "algorithm": txt_config['algorithm']},
        "agent_analysis":   agent_text,
        "samples_analyzed": len(image_embeddings),
    }
    if mm_report is not None:
        out["multimodal_drift"] = mm_report
    return out