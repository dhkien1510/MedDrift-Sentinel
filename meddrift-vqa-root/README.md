Ran command: `python meddrift-vqa-root/meddrift-ai-service/src/monitoring/text_drift.py`
Edited text_drift.py
Edited text_drift.py
Ran command: `python meddrift-vqa-root/meddrift-ai-service/src/monitoring/text_drift.py`
Ran command: `python meddrift-vqa-root/meddrift-ai-service/src/monitoring/text_drift.py`
Viewed text_drift.py:4-11

Tuyệt vời! Bước tiếp theo: **`drift_service.py`** — kết hợp image + text drift lại.

## Bước tiếp: `drift_service.py`

Đây là file **đơn giản nhất** — nó chỉ gọi 2 module kia rồi tổng hợp kết quả:

```
drift_service.py
├── check_dual_drift(images, questions, algorithm, p_threshold)
│   ├── gọi image_drift.check_image_drift(...)
│   ├── gọi text_drift.check_text_drift(...)
│   └── tổng hợp: overall_drift, drift_source
└── return dict kết quả
```

### Output nên có dạng:
```python
{
    "image_drift": {"is_drift": True, "p_value": 0.001, "distance": 1.29},
    "text_drift": {"is_drift": False, "p_value": 0.42, "distance": 0.03},
    "overall_drift": True,
    "drift_source": "image"    # "none" | "image" | "text" | "both"
}
```

### Logic `drift_source`:
| Image Drift | Text Drift | `drift_source` |
|------------|-----------|----------------|
| ❌ | ❌ | `"none"` |
| ✅ | ❌ | `"image"` |
| ❌ | ✅ | `"text"` |
| ✅ | ✅ | `"both"` |

### Lưu ý:
- Import bằng `import image_drift` và `import text_drift` (cùng folder `monitoring/`)
- `overall_drift = image_result['is_drift'] OR text_result['is_drift']`
- File này **không cần load model** — nó delegate cho image_drift và text_drift

Sau khi xong `drift_service.py` → bạn đã hoàn thành **Phase 1 + Phase 2** (Core Drift + Registry). Tiếp theo sẽ là **Phase 3 (LangChain Agent)** hoặc **Phase 4 (FastAPI)** — tuỳ bạn muốn làm cái nào trước!