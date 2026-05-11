"""
drift_agent.py — LangChain Drift Sentinel Agent
=================================================
Agent thông minh điều phối toàn bộ drift detection pipeline.
Thay vì trả True/False đơn thuần, Agent:
  1. Chạy cả image + text drift (dual drift)
  2. Phân loại severity (🟢🟡🟠🔴)
  3. Giải thích TẠI SAO drift xảy ra
  4. Đề xuất hành động cho bác sĩ

Tools:
  - check_image_drift_tool:  CLIP + MMD trên ảnh
  - check_text_drift_tool:   BioBERT + MMD trên câu hỏi
  - check_dual_drift_tool:   Kết hợp cả hai → severity + drift_source

Modes:
  - Per-request: run_drift_sentinel() — chạy trực tiếp trên raw images/questions
  - Batch:       run_drift_sentinel_batch() — chạy trên pre-extracted embeddings từ buffer
"""


from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter
from langgraph.prebuilt import create_react_agent
from registry import load_config
from image_drift import check_image_drift, check_image_drift_from_embeddings
from text_drift import check_text_drift, check_text_drift_from_embeddings
import numpy as np
import contextvars


# Load API key từ .env
load_dotenv()


# ============================================================
# CONTEXT: Lưu input data trước khi Agent chạy
# Agent giao tiếp bằng text, không truyền được PIL Image / np.ndarray
# → Lưu data vào ContextVar (thread-safe cho FastAPI)
# ============================================================

_current_images: contextvars.ContextVar = contextvars.ContextVar('_current_images', default=None)
_current_questions: contextvars.ContextVar = contextvars.ContextVar('_current_questions', default=None)
_context_mode: contextvars.ContextVar = contextvars.ContextVar('_context_mode', default='raw')

def set_context(images, questions, mode='raw'):
    """
    Set input data trước khi gọi Agent.
    
    Args:
        images: List[PIL.Image] (mode='raw') hoặc np.ndarray shape (N, 512) (mode='batch')
        questions: List[str] (mode='raw') hoặc np.ndarray shape (N, 768) (mode='batch')
        mode: 'raw' = raw images/text, 'batch' = pre-extracted embeddings
    """
    _current_images.set(images)
    _current_questions.set(questions)
    _context_mode.set(mode)


# ============================================================
# BUFFER: Tích lũy embeddings trước khi chạy drift detection
# Mỗi VQA request → extract embedding → add vào buffer
# Khi đủ BUFFER_THRESHOLD → flush → chạy Agent trên batch
# ============================================================

BUFFER_THRESHOLD = 100
_image_embedding_buffer = []
_text_embedding_buffer = []

def add_to_buffer(image_embedding, text_embedding):
    """Thêm 1 cặp embedding vào buffer. Gọi từ FastAPI mỗi request."""
    _image_embedding_buffer.append(image_embedding)
    _text_embedding_buffer.append(text_embedding)

def is_buffer_ready():
    """Check nếu buffer đã đủ threshold."""
    return len(_image_embedding_buffer) >= BUFFER_THRESHOLD

def get_buffer_count():
    """Trả về số lượng embeddings hiện tại trong buffer."""
    return len(_image_embedding_buffer)

def flush_buffer():
    """
    Lấy toàn bộ embeddings ra khỏi buffer, reset buffer.
    
    Returns:
        (np.ndarray, np.ndarray): image_embeddings (N, 512), text_embeddings (N, 768)
    """
    global _image_embedding_buffer, _text_embedding_buffer
    img = np.array(_image_embedding_buffer)
    txt = np.array(_text_embedding_buffer)
    _image_embedding_buffer = []
    _text_embedding_buffer = []
    return img, txt


# ============================================================
# TOOL 1: Image Drift Detection
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
    mode = _context_mode.get()
    if images is None:
        return "No images provided. Please upload images first."

    config = load_config(isImage=True)
    
    # Chọn hàm phù hợp dựa trên mode
    if mode == 'batch':
        result = check_image_drift_from_embeddings(
            images, config['p_threshold'], config['algorithm'], config['ref_embeddings']
        )
    else:
        result = check_image_drift(
            images, config['p_threshold'], config['algorithm'], config['ref_embeddings']
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
# TOOL 2: Text Drift Detection
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
    mode = _context_mode.get()
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
# TOOL 3: Dual Drift Check + Severity Report
# Kết hợp image + text → phân loại severity
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
    images = _current_images.get()
    questions = _current_questions.get()
    mode = _context_mode.get()
    if images is None or questions is None:
        return "Error: Missing context. Both images and questions must be provided."

    # 1. Load configs
    img_config = load_config(isImage=True)
    txt_config = load_config(isImage=False)

    # 2. Run both drift checks (chọn hàm dựa trên mode)
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

    # 3. Determine drift source
    img_drifted = img_result['is_drift']
    txt_drifted = txt_result['is_drift']

    if img_drifted and txt_drifted:
        drift_source = "both"
        severity = "🔴 CRITICAL"
        action = "REJECT input — both image and question are outside training distribution. VQA results are unreliable."
    elif img_drifted:
        drift_source = "image"
        severity = "🟠 HIGH"
        action = "WARNING — Image may not be a standard chest X-ray. VQA accuracy is compromised."
    elif txt_drifted:
        drift_source = "text"
        severity = "🟡 MEDIUM"
        action = "WARNING — Question is outside radiology domain. Consider rephrasing as a medical question."
    else:
        drift_source = "none"
        severity = "🟢 NORMAL"
        action = "System operating normally. VQA results are reliable."

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


# ============================================================
# EXPORT: Danh sách tools cho Agent sử dụng
# ============================================================
DRIFT_TOOLS = [
    check_image_drift_tool,
    check_text_drift_tool,
    check_dual_drift_tool,
]


# ============================================================
# SYSTEM PROMPT: Hướng dẫn Agent
# ============================================================
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


# ============================================================
# AGENT: Khởi tạo LLM + Agent (tool calling — không cần ReAct format)
# ============================================================
llm = ChatOpenRouter(
    model="google/gemini-2.5-flash-lite",
    temperature=0,
)

# create_react_agent từ langgraph dùng native tool calling
# (khác với langchain.agents.create_react_agent dùng text format)
agent_executor = create_react_agent(llm, DRIFT_TOOLS, prompt=SYSTEM_PROMPT)


# ============================================================
# ENTRY POINT 1: Per-request mode (raw images + questions)
# Dùng cho testing hoặc khi muốn check drift ngay lập tức
# ============================================================
def run_drift_sentinel(question: str, images: list, image_description: str = "uploaded medical image"):
    """
    Entry point cho per-request drift detection.
    
    Args:
        question: Câu hỏi của bác sĩ
        images: List PIL Images
        image_description: Mô tả ngắn về ảnh (optional)
    
    Returns:
        str: Báo cáo drift chi tiết từ Agent (tiếng Việt)
    """
    # 1. Set context (raw mode — tools sẽ extract embeddings)
    set_context(images=images, questions=[question], mode='raw')
    
    # 2. Chạy Agent
    user_message = (
        f"Please analyze drift for the following input:\n"
        f"- Doctor's Question: {question}\n"
        f"- Image: {image_description}\n"
        f"Run the appropriate drift detection tools and provide your analysis."
    )
    try:
        result = agent_executor.invoke(
            {"messages": [{"role": "user", "content": user_message}]}
        )
        final_message = result["messages"][-1]
        return final_message.content
    except Exception as e:
        return f"Drift Agent Error: {str(e)}\n"


# ============================================================
# ENTRY POINT 2: Batch mode (pre-extracted embeddings từ buffer)
# Dùng khi buffer đủ threshold → flush → chạy Agent trên batch
# ============================================================
def run_drift_sentinel_batch(image_embeddings: np.ndarray, text_embeddings: np.ndarray) -> dict:
    """
    Entry point cho batch drift detection.
    Gọi khi buffer đạt threshold.
    
    Args:
        image_embeddings: np.ndarray shape (N, 512) — CLIP embeddings từ buffer
        text_embeddings: np.ndarray shape (N, 768) — BioBERT embeddings từ buffer
    
    Returns:
        dict: Structured report với severity, drift data (cho charts), và agent analysis
    """
    # 1. Set context (batch mode — tools sẽ dùng _from_embeddings)
    set_context(images=image_embeddings, questions=text_embeddings, mode='batch')
    
    # 2. Chạy Agent để lấy phân tích bằng tiếng Việt
    user_message = (
        f"Analyze drift for a batch of {len(image_embeddings)} samples.\n"
        f"These are pre-extracted embeddings collected over the last "
        f"{len(image_embeddings)} VQA requests.\n"
        f"Run the drift detection tools and provide your analysis."
    )
    try:
        result = agent_executor.invoke(
            {"messages": [{"role": "user", "content": user_message}]}
        )
        agent_text = result["messages"][-1].content
    except Exception as e:
        agent_text = f"Drift Agent Error: {str(e)}"

    # 3. Chạy drift detection trực tiếp để lấy structured data (cho charts/alerts)
    img_config = load_config(isImage=True)
    txt_config = load_config(isImage=False)
    
    img_result = check_image_drift_from_embeddings(
        image_embeddings, img_config['p_threshold'], img_config['algorithm'], img_config['ref_embeddings']
    )
    txt_result = check_text_drift_from_embeddings(
        text_embeddings, txt_config['p_threshold'], txt_config['algorithm'], txt_config['ref_embeddings']
    )

    # 4. Phân loại severity
    img_drifted = img_result['is_drift']
    txt_drifted = txt_result['is_drift']
    
    if img_drifted and txt_drifted:
        severity, drift_source = "🔴 CRITICAL", "both"
    elif img_drifted:
        severity, drift_source = "🟠 HIGH", "image"
    elif txt_drifted:
        severity, drift_source = "🟡 MEDIUM", "text"
    else:
        severity, drift_source = "🟢 NORMAL", "none"

    # 5. Return structured report (frontend dùng cho charts + alerts)
    return {
        "severity": severity,
        "drift_source": drift_source,
        "alert": img_drifted or txt_drifted,
        "image_drift": {
            **img_result,
            "algorithm": img_config['algorithm'],
        },
        "text_drift": {
            **txt_result,
            "algorithm": txt_config['algorithm'],
        },
        "agent_analysis": agent_text,
        "samples_analyzed": len(image_embeddings),
    }


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    from PIL import Image
    
    # Tạo 5 ảnh random noise (kỳ vọng: drift!)
    test_images = [
        Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        for _ in range(5)
    ]
    
    # Test 1: Per-request mode (raw images + questions)
    print("=" * 60)
    print("TEST 1: Per-request mode — raw images + non-medical question")
    print("=" * 60)
    output = run_drift_sentinel(
        question="What breed is this dog?",
        images=test_images,
        image_description="random noise images (not X-ray)"
    )
    print("\n=== AGENT OUTPUT ===")
    print(output)