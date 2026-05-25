# Tổng quan Kiến trúc Hệ thống và Luồng dữ liệu (MedDrift-Sentinel)

Dựa trên cấu trúc dự án và mô tả luồng hoạt động, dưới đây là sơ đồ Tổng quan Kiến trúc Hệ thống và Luồng dữ liệu Sequence Diagram, được vẽ lại bằng Mermaid.

## 1. Sơ đồ Kiến trúc Hệ thống (System Architecture Diagram)

Sơ đồ này mô tả cách các thành phần trong hệ thống tương tác với nhau, tập trung vào kiến trúc Sentinel giám sát dựa trên LangChain ReAct Agent.

```mermaid
graph TD
    subgraph Client Tier
        UI[React.js / Web UI <br> + Heuristic VQA Classification]
    end

    subgraph Business Logic Tier
        Gateway[Node.js / Express API Gateway]
    end

    subgraph Storage & External Hub Tier
        Redis[(Redis Buffer & Cache)]
        Mongo[(MongoDB - Reports & Logs)]
        MinIO[(MinIO - Image Storage)]
        HF[(Hugging Face Hub <br> Datasets & Models)]
    end

    subgraph MedDrift-Sentinel: AI Monitoring Tier
        FastAPI[Python FastAPI]
        Buffer[Drift Buffer Monitor]
        LCAgent[LangChain ReAct Agent]
        
        subgraph Drift Detection Tools
            ImgTool[Image Drift <br> MMD Test]
            TxtTool[Text Drift <br> MMD Test]
            MultiTool[Multimodal Drift <br> PCA + MMD]
        end
    end

    subgraph Cloud Inference Tier
        Gemini[Gemini 2.5 Flash <br> via OpenRouter API]
    end

    %% Luồng đi của dữ liệu
    UI -- "1. Submit Image + Text" --> Gateway
    Gateway -- "2. Routes Collect Request" --> FastAPI
    HF -. "Download Pre-trained Encoders <br> & Reference Datasets (Init)" .-> FastAPI
    
    FastAPI -- "3. Return PCA Multimodal Embeddings" --> Gateway
    Gateway -- "4. Forward to Client" --> UI
    
    FastAPI -- "5. Store original image" --> MinIO
    FastAPI -- "6. Push vector embeddings" --> Redis
    
    Redis -- "7. Threshold hit (100)" --> Buffer
    Buffer -- "8. Trigger Sentinel" --> LCAgent
    
    LCAgent -- "9. Execute Tools" --> ImgTool
    LCAgent -- "9. Execute Tools" --> TxtTool
    LCAgent -- "9. Execute Tools" --> MultiTool
    
    LCAgent -- "10. Reasoning & formatting" --> Gemini
    Gemini -- "11. Severity, Explain & Suggest" --> LCAgent
    
    LCAgent -- "12. Save Drift Report" --> Mongo
```

### Chi tiết vai trò của các tầng (Tiers):
1. **Client Tier**: Là giao diện tương tác người dùng xây dựng bằng React.js/Vite. Nơi bác sĩ gửi câu hỏi kèm hình ảnh. Tại đây, React sẽ sử dụng các vector chiếu (PCA) trả về để chạy thuật toán Linear Classification đơn giản ra đáp án VQA (Có/Không). Nơi này cũng hiển thị / giám sát biểu đồ PCA và cấu hình các kịch bản test Drift.
2. **Business Logic Tier**: Được vận hành bằng Node.js & Express.js đóng vai trò API Gateway, giúp định tuyến request xuống AI Monitoring và đảm bảo vấn đề CORS, Static path.
3. **Storage & External Hub Tier**: Lớp lưu trữ cơ sở dữ liệu và quản lý model:
   - **Hugging Face Hub**: Đóng vai trò là nguồn tải xuống Data Tham chiếu (Reference VQA-RAD) và các Models Encoders tiền huấn luyện (như DINOv2, BioBERT) được mount vào thư mục cache docker ở lần khởi chạy đầu tiên.
   - **Redis**: Dùng làm đệm (Buffer) để cache vector trích xuất. Khi số lượng kỷ lục đạt Threshold (ngưỡng 100 theo cấu hình), nó sẽ kích hoạt flush_buffer để chạy Agent.
   - **MongoDB**: Hệ quản trị phi quan hệ để lưu lịch sử, các kịch bản và toàn bộ các Drift Reports sau khi agent chạy xong.
   - **MinIO**: Object Storage lưu trữ ảnh nguyên bản do Client gửi lên để truy xuất hiển thị cho báo cáo và lưu dữ liệu offline generation.
4. **AI Monitoring Tier (MedDrift-Sentinel)**: Module AI cốt lõi xây dựng bởi Python FastAPI, chứa engine trích xuất vector embedding và đính kèm **LangChain ReAct Agent**. Tại đây, nó chiếu các vector đa phương thức (PCA) và xử lý giám sát kiểm định cho luồng batch data từ Redis bằng các Tools phân phối.
5. **Cloud Inference Tier**: Khác với các hệ thống phổ thông, ta không tốn chip xử lý cho LLM tại chỗ mà đẩy logic Suy luận qua đám mây:
   - *OpenRouter (Gemini)*: LLM Agent xử lý tự động trong luồng LangChain. Nhận Input kết quả báo cáo MMD thống kê, thực hiện quá trình Suy diễn (Reasoning), chỉ ra lý do hiện trạng Drift và đánh giá cấp độ cảnh báo (Severity).

---

## 2. Biểu đồ Tuần tự (Sequence Diagram)

Biểu đồ này biểu diễn chi tiết luồng dữ liệu (Data Flow) theo thời gian thực từ lúc Bác sĩ upload câu hỏi và hình ảnh, cho đến cách LangChain Sentinel hoạt động giám sát ngầm.

```mermaid
sequenceDiagram
    autonumber
    
    actor Doctor as Bác sĩ (React)
    participant Node as Express API Gateway
    participant Sentinel as Python FastAPI (Sentinel)
    participant Redis as Redis (Buffer)
    participant Agent as LangChain Agent (Gemini)
    participant Mongo as MongoDB (Reports)

    Doctor->>Node: POST /api/vqa (Image + Question)
    Node->>Sentinel: POST /api/drift/collect
    
    Sentinel->>Sentinel: Trích xuất Embeddings (CLIP/BioBERT/... load từ HF)
    Sentinel->>Redis: Thêm vector vào Buffer
    Sentinel-->>Node: Trả về kết quả Dự phóng Multimodal (PCA Projection)
    Node-->>Doctor: Tính toán Heuristic phân loại VQA & Hiển thị
    
    rect rgb(240, 248, 255)
        Note right of Sentinel: Quá trình Monitor chạy ngầm độc lập (Background)
        alt Buffer đạt ngưỡng Threshold (>= 100 samples)
            Sentinel->>Redis: Lấy toàn bộ Embeddings & Xóa Buffer
            Sentinel->>Agent: Kích hoạt run_drift_sentinel_batch()
            
            rect rgb(255, 240, 245)
                Note over Agent: ReAct flow
                Agent->>Agent: Tool: check_image_drift_tool (MMD/Chi-Square)
                Agent->>Agent: Tool: check_text_drift_tool (MMD/Chi-Square)
                Agent->>Agent: Tool: check_dual_drift_tool (Dual MMD/Chi-Square)
                Agent->>Agent: LLM Reasoning (Gemini) phân tích kết quả
            end
            
            Agent-->>Sentinel: Báo cáo Agent Analysis & Severity
            Sentinel->>Mongo: Lưu Drift Report & Cảnh báo Alert
        end
    end
    
    Doctor->>Node: GET /api/drift/status (Polling)
    Node->>Sentinel: Forward
    Sentinel-->>Doctor: Trả về thông tin Drift Alert mới nhất
```

---

## 3. Luồng Quản lý Cấu hình và Kịch bản Test (Configuration & Scenario Testing Flow)

Ngoài luồng chính phục vụ chẩn đoán, hệ thống còn cung cấp giao diện quản trị (Configuration/Testing Flow). Tại đây, người dùng hệ thống hoặc admin có thể điều chỉnh cấu hình mạng nơ-ron, tham số thuật toán (p-value, test type) hay chọn kịch bản (scenario) test.

```mermaid
sequenceDiagram
    autonumber
    
    actor Admin as Admin/Nhà nghiên cứu (React: ConfigPage, TestPage)
    participant Node as Node.js (Express Proxy)
    participant Sentinel as Python FastAPI (Sentinel AI)
    participant YAML as drift_config.yaml (Local File)
    participant Scenario as Drift Scenarios (Data Dir)

    %% Flow 1: Lấy cấu hình và kịch bản
    Admin->>Node: GET /api/drift/config & /api/scenarios
    Node->>Sentinel: Forward Request
    Sentinel->>YAML: Đọc file cấu hình hiện tại
    YAML-->>Sentinel: Trả về dữ liệu cấu hình
    Sentinel->>Scenario: Đọc danh sách metadata.yaml
    Scenario-->>Sentinel: Trả về danh sách kịch bản
    Sentinel-->>Node: Trả về JSON (Config + Scenarios)
    Node-->>Admin: Hiển thị lên UI (Drift Models, p-value, Scenarios)
    
    %% Flow 2: Cập nhật cấu hình
    Admin->>Node: POST /api/drift/config/apply (Params changes)
    Node->>Sentinel: Forward Data
    Sentinel->>YAML: Ghi đè file cấu hình (drift_config.yaml)
    YAML-->>Sentinel: Báo thành công
    Sentinel->>Sentinel: Reload Memory/Models theo cấu hình mới
    Sentinel-->>Node: 200 OK - Config Applied
    Node-->>Admin: Hiển thị thông báo Cập nhật thành công
    
    %% Flow 3: Chạy kịch bản giả lập
    Admin->>Node: POST /api/vqa/batch_test (Chọn Scenario)
    Node->>Sentinel: Forward Dữ liệu Test
    Sentinel->>Scenario: Load ảnh & text từ scenario tương ứng
    Scenario-->>Sentinel: Image & Text Data
    Sentinel->>Sentinel: Chạy suy luận & Drift Guard
    Sentinel-->>Node: Trả về kết quả đánh giá (Drift Alert/Safe)
    Node-->>Admin: Hiển thị báo cáo kết quả kịch bản Test
```

---

## 4. Hệ thống Tiền xử lý và Sinh Dữ liệu Reference & Drift (Data Pipeline)

Để hệ thống có thể đối chiếu và phát hiện Drift, kiến trúc dự án có một luồng chuẩn bị dữ liệu ngoại tuyến (Offline Data Preparation), thực hiện qua các script Python tại `meddrift-ai-service/scripts/`. Quá trình này bao gồm 2 nhiệm vụ chính:

### 4.1. Xây dựng Reference Data (Tập dữ liệu nền/chuẩn)
Reference data được định nghĩa là trạng thái "Safe" (Bình thường), được trích xuất từ tập dữ liệu gốc (ví dụ: tập dữ liệu huấn luyện VQA-RAD) bằng file **`build_reference.py`** và **`build_multimodal_reference.py`**:
- **Trích xuất Đặc trưng Hình ảnh (Image Reference):** Hệ thống load ảnh gốc và chạy qua các Image Encoders (như CLIP, BiomedCLIP, DINOv2 của Microsoft chuyên y tế, hoặc tiêu chuẩn ViT) để tạo ra các mảng vector đặc trưng `.npy`.
- **Trích xuất Đặc trưng Văn bản (Text Reference):** Hệ thống load các câu hỏi (text) và mã hóa qua Text Encoders (như BioBERT, PubMedBERT-SBERT, Model2Vec) thành file `.npy`.
- **Hợp nhất Đa phương thức (Multimodal Fusion):** Sau khi tính ra list file `.npy` dạng đơn (unimodal), hệ thống sử dụng thuật toán PCA (Principal Component Analysis) để giảm chiều và ráp nối không gian Ảnh + Chữ lại, sinh ra file tham chiếu kết hợp `joint_reference.npy`.

### 4.2. Xây dựng Drift Data Scenarios (Kịch bản giả lập độ lệch)

Quá trình này được triển khai thông qua **`build_drift_data.py`** với mục đích tạo ra các kịch bản ngoại chuẩn có kiểm soát để benchmark thuật toán MMD Guard. Data đi từ tập dữ liệu gốc (VQA-RAD) qua các lớp Filter/Augmentation (đối với ảnh) và LLM Prompting (đối với chữ) để sinh ra đa dạng các biến thể bất thường.

#### Biểu đồ Luồng tạo Kịch bản Data Drift (Data Generation Flow)

```mermaid
graph TD
    classDef origin fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef augment fill:#e1f5fe,stroke:#4682b4;
    classDef llm fill:#fff0f5,stroke:#db7093;
    classDef encode fill:#f0fff0,stroke:#8fbc8f;

    subgraph DataNen ["1. VQA-RAD Dataset Hồi Quy"]
        RawImg["Raw Medical Images"]:::origin
        RawTxt["Raw Medical Questions"]:::origin
    end

    subgraph AnhThaoTacPixel ["2A. Image Augmentation"]
        ImgL0["No Drift: Giữ nguyên"]:::augment
        ImgL1["Level 1 - Mild: ±Sáng, Xoay, Lật"]:::augment
        ImgL2["Level 2 - Moderate: Làm mờ, Nhiễu, CLAHE"]:::augment
        ImgL3["Level 3 - Severe: Đảo màu Âm bản, Mảng nhiễu"]:::augment
    end

    subgraph ChuSinhBangLLM ["2B. LLM OpenRouter/Gemini"]
        TxtL0["No Drift: Giữ nguyên"]:::llm
        TxtL1["Level 1 - Mild: Viết tắt, Khẩu ngữ lâm sàng"]:::llm
        TxtL2["Level 2 - Moderate: Lạc chuyên khoa, Dài dòng, Đa ngôn ngữ"]:::llm
        TxtL3["Level 3 - Severe: Nhảm nhí (Off-topic), Ký tự rác"]:::llm
    end

    RawImg --> ImgL0 & ImgL1 & ImgL2 & ImgL3
    RawTxt --> TxtL0 & TxtL1 & TxtL2 & TxtL3

    ImgL0 & ImgL1 & ImgL2 & ImgL3 --> ExtImg{"Đưa qua <br> Image Encoders (DINO/CLIP)"}:::encode
    TxtL0 & TxtL1 & TxtL2 & TxtL3 --> ExtTxt{"Đưa qua <br> Text Encoders (BioBERT/...)"}:::encode

    ExtImg --> NpyImg[("Thư mục drift_scenarios/image/*.npy")]
    ExtTxt --> NpyTxt[("Thư mục drift_scenarios/text/*.npy")]
    
    NpyImg & NpyTxt --> Meta["Tạo bảng ghi metadata.yaml"]
```

#### Phân tích chi tiết & Ví dụ từng Level

**A. Các Cấp độ Kịch bản Hình ảnh (Image Scenarios)**  
Sử dụng thư viện PIL (Pillow) để thao tác pixel dựa trên ma trận RGB.
- **Level 0 (No Drift):** Giữ nguyên không can thiệp. Đây là đối chứng y chang tập Reference.
- **Level 1 (Nhẹ - Mild):** 
  - *Bản chất:* Xoay ảnh (±10 độ), thay đổi sáng tối và độ tương phản ±15%, lật dọc ngẫu nhiên (50%).
  - *Mô phỏng thực tế:* Bệnh nhân cựa quậy khi chụp X-Quang, phim bị xoay nhẹ trên kính, hoặc ánh sáng đèn backlight của màn hình khác biệt.
- **Level 2 (Vừa - Moderate):** 
  - *Bản chất:* Làm mờ (Gaussian blur r=3), nhiễu xạ Gaussian noise (bước nhảy $\sigma$=25) và kéo mức tương phản cực căng như công thức CLAHE.
  - *Mô phỏng thực tế:* Bệnh viện mới đổi dòng máy scanner hoặc cảm biến bị cũ, profile ảnh đầu ra có độ nhiễu hạt hoàn toàn khác so với thông số huấn luyện gốc.
- **Level 3 (Rào cản - Severe):** 
  - *Bản chất:* Đảo ngược không gian màu âm bản (Grayscale Invert - trắng thành đen, đen thành trắng) kết hợp 50% diện tích bị thay bằng các block hộp nhiễu hỏng ngẫu nhiên.
  - *Mô phỏng thực tế:* Tải nhầm ảnh chụp siêu âm sang X-quang, hoặc module thiết bị chập nát dữ liệu raw đẩy nguyên bức ảnh bị lỗi rác vào Model.

**B. Các Cấp độ Kịch bản Văn bản (Text Scenarios)**  
Nhờ LLM API (OpenRouter truyền qua Gemini 2.5) với Zero-shot Prompting để thay đổi mặt văn bản, sinh ra luồng biến thể mới mẻ liên tục.
- **Level 0 (No Drift):** Giữ nguyên câu hỏi tiếng Anh từ VQA-RAD. Ví dụ: *"Is there cardiomegaly in this image?"*
- **Level 1 (Nhẹ - Mild):** 
  - *Bản chất:* Ép LLM dùng viết tắt y tế cực đoan (Radiologist shorthand), loại bỏ in hoa và dấu câu.
  - *Mô phỏng/Ví dụ:* "pt hx cxr cardiomegaly" (Bác sĩ dùng từ lóng, code-word thay vì hỏi toàn văn).
- **Level 2 (Vừa - Moderate):** 
  - *Bản chất:* Ba nhánh chính: Dịch ngôn ngữ (Tây Ban Nha, Đức...), Viết cực kỳ dông dài (Verbose), hoặc lạc khoa (Cross-domain như hỏi về Da liễu, Chấn thương chỉnh hình trên ảnh tim phổi).
  - *Mô phỏng/Ví dụ (Verbose):* "Could you please elaborate in extensive detail whether the patient's cardiac silhouette appears significantly enlarged..."
- **Level 3 (Rào cản - Severe):** 
  - *Bản chất:* Hỏi nhảm (Ví dụ hỏi về công thức nấu ăn, tỷ số bóng đá) hoặc ép sinh các đoạn mã Token vô nghĩa.
  - *Mô phỏng/Ví dụ (Off-topic):* "What is the best recipe for making a strawberry cake and enjoying the football match?" 

*(Lưu ý: Bên cạnh thao tác dữ liệu gốc qua LLM & PIL, script trên còn có hàm `build_synthetic_embedding_drift` để cộng noise/covariance thẳng vào số thực không gian Embedding để kiểm thử thuật toán MMD mà không cần quá trình render đồ họa lại.)*
