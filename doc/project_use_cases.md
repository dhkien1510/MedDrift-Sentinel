# Phân Tích Use Case & Công Cụ (Use Cases & Technology Stack)

Tài liệu này tổng hợp các Use Case cốt lõi nhất (nhìn từ góc độ Bác sĩ/Người dùng) và Bảng công cụ kỹ thuật phục vụ cho việc làm Slide thuyết trình và viết Báo cáo môn học.

---

## 1. Bảng Các "Use Case" Cơ Bản Nhất (Functional Scenarios)

Dưới đây là 4 tình huống sử dụng điển hình nhất chứng minh giá trị của hệ thống MedDrift-Sentinel tại bệnh viện:

| Mã UC | Tên Use Case | Kịch bản Bác sĩ (Hành động) | Phản hồi của Hệ thống MedDrift-Sentinel | Ý nghĩa MLOps |
|:---:|---|---|---|---|
| **UC-01** | **Chẩn đoán X-quang Chuẩn (Happy Path)** | Up ảnh X-quang phổi đẹp, gõ câu hỏi: *"Có dấu hiệu viêm phổi (Pneumonia) không?"* | Xử lý mượt mà. P-values > 0.05. Trả câu trả lời y khoa chuẩn xác, giao diện xanh lá an toàn. | Hệ thống phục vụ tốt nghiệp vụ cốt lõi không bị cản trở. |
| **UC-02** | **Cảnh báo Dụng cụ Chụp Lỗi (Acquisition Drift)** | Up bức chụp X-quang bị dư sáng, nhòe tia X, hoặc chụp bằng dòng máy cũ kỹ. Câu hỏi bình thường. | AI vẫn ráng trả lời nhưng UI hiện thẻ **Màu Cam**: *"Cảnh báo: Phân phối điểm ảnh bị rẽ hướng. Độ tin cậy giảm, vui lòng xem xét chụp lại X-quang".* | Cứu rỗi tính mạng bệnh nhân khỏi các hệ thống AI tự tin thái quá khi data bị mờ. |
| **UC-03** | **Cảnh báo Lạc Đề (Semantic Shift Text)** | Up X-quang chuẩn chỉnh nhưng gõ nhầm câu hỏi da liễu: *"Giai đoạn ung thư hắc tố bào (Melanoma) ở đây là gì?"* | Text Drift Guard kéo còi. Trả kết quả kèm **Badge Đỏ**: *"Mô hình AI chuyên về X-quang, câu hỏi của bác sĩ nằm ngoài vùng hiểu biết của mô hình (Out-of-Distribution)".* | Tránh AI sinh ra hội chứng "Ảo giác" (Hallucinations) nói xằng bậy ngoài chuyên môn. |
| **UC-04** | **Chặn Rác Trực Tiếp (Severe Garbage Input)** | Người nhà cố tình up ảnh thẻ cá nhân, ảnh siêu âm thai hoặc hỏi thăm *"Thời tiết hôm nay thế nào?"* | Gateways song song ngắt ngay lập tức, vứt bỏ toàn bộ, không thèm đưa lên LLaVA-Med. UI báo lỗi lầm lạc (Invalid Medical Data). | Bảo vệ ngân sách API Cloud (giảm cost gọi LLM vô ích) và tăng tính bảo mật Enterprise. |
| **UC-05** | **Xem màn hình Lịch sử Khám (Audit Log)** | Bác sĩ bấm vào tab History trên màn hình, xem lại báo cáo tuần trước. | Trích xuất toàn bộ dữ liệu (câu hỏi, ảnh bệnh nhân, chỉ số Drift Score ngày đó) bằng tốc độ chớp nhoáng. | Tính minh bạch dữ liệu, hỗ trợ truy vết lỗi cho bệnh viện (Pháp lý). |

---

## 2. Bảng Công Cụ, Công Nghệ & Khung MLOps (Technology Stack)

Hệ thống được thiết kế hoàn thiện theo chuẩn **Microservices Doanh nghiệp** chứ không chỉ lùi về việc "code một mô hình AI bằng file Jupyter Notebook".

### 2.1 Mảng AI, Machine Learning & MLOps
| Tên Công Cụ / Model | Vai trò trong Đồ Án | Lý do lựa chọn (Dành cho vấn đáp) |
|---|---|---|
| **ALIBI-DETECT** | Thư viện thống kê chính. Dạy thuật toán chạy **MMD Test** 2 cụm (Two-Sample). | Là chuẩn ngành (Industry Standard) trong MLOps. Nhẹ hơn rất nhiều so với tự viết Pytorch, có chứng minh toán học cực mạnh về kiểm định. |
| **CLIP (ViT-B/32)** | Dịch bức ảnh X-quang thành một dãy số (Vector 512 chiều). | Mạng thần kinh đỉnh cao của OpenAI. Hiểu được cả ảnh và từ vựng, trích xuất Feature tốt gấp vạn lần CNN truyền thống. |
| **BioBERT** | Dịch câu gõ của bác sĩ thành Vector 768 chiều (Word Embeddings). | Tinh chỉnh riêng trên PubMed (Kho y học khổng lồ). Siêu việt hơn BERT thường khi gặp từ khóa như *Atelectasis* (xẹp phổi). |
| **LLaVA-Med** | "Bộ Não" (Vision-Language Model) ngồi ở đám mây để xì ra câu trả lời tiếng Anh cuối cùng. | Giải quyết được bài toán sinh từ có hình ảnh y khoa kết hợp (VQA Miltimodal). |
| **Scikit-Learn & Matplotlib** | Tool "Nén" ma trận 512D -> 2D (PCA) và vẽ Scatter Plot màu mè. | Giúp trực quan hóa dữ liệu đa chiều cho con người và ban giám khảo dễ hiểu khi lên Slide (Slide 10). |

### 2.2 Mảng Backend, Orchestrator & Lưu trữ (Infrastructure) 
| Tên Phần Mềm | Vai trò trong Đồ Án | Lý do lựa chọn (Dành cho vấn đáp) |
|---|---|---|
| **Python FastAPI** | Server đứng bảo kê các mảng AI (Drift Guard). | FastAPI sinh ra để làm Machine Learning. Bất đồng bộ hóa (Async) cực khỏe khi chạy mô hình Torch. |
| **Node.js (Express)** | API Gateway nhận file ảnh trực tiếp từ Frontend và điều phối 2 nhánh song song. | Khả năng quản trị I/O luồng I/O bất đồng bộ bằng `Promise.all` khiến nó thành vị vua gọi API song song. |
| **Multer & Axios** | Phân rã luồng ảnh (Multipart) không cần chạm ổ cứng, đẩy thẳng bằng RAM (Axios) qua Python. | Giữ Backend không bị tràn bộ nhớ khi có bác sĩ đăng file 10MB liên tục. |
| **Redis** | In-memory Cache (Lưu bộ đệm tạm thời cho các câu hỏi trùng lặp). | Bắt buộc phải có để triệt tiêu độ trễ cực mạnh và né việc phải gọi trả phí API LLaVA-Med giá đắt liên tục. |
| **MongoDB** | Lưu lại log của Use Case 05, các thông số điểm Drift. | Hỗ trợ nhồi nhét file JSON các cấu trúc linh hoạt (Drift dict lúc vầy lúc khác) rất mượt. |
| **Amazon S3 / MinIO** | Chứa xác X-quang khổng lồ. | Tách rời file ảnh khỏi cơ sở dữ liệu. Tuân thủ định luật Hybrid Cloud trong Y tế. |
| **Docker Compose** | Gói tất bật cả (React, Mongo, S3, Node, Python) vô 1 file yaml duy nhất. | Trả lời một chữ "Tiện lợi và Chuyển giao cực nhanh (Deployability)". Cứ gõ `docker-compose up` là ra hệ thống ở mọi máy. |
