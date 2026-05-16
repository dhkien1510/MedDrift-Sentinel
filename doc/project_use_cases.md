# Phân Tích Use Case & Công Cụ (Use Cases & Technology Stack)

Tài liệu này tổng hợp các Use Case cốt lõi mà người dùng (Bác sĩ, Quản trị viên/MLOps) có thể thực hiện trực tiếp trên giao diện Frontend của MedDrift-Sentinel, cùng với Bảng công cụ kỹ thuật phục vụ cho việc phát triển hệ thống.

---

## 1. Các Use Case Trên Giao Diện Người Dùng (Frontend Scenarios)

Hệ thống MedDrift-Sentinel cung cấp 4 màn hình chính tương ứng với các chức năng chuyên biệt:

### 1.1 Trò chuyện & Chẩn đoán hình ảnh (ChatPage)
**Người dùng mục tiêu:** Bác sĩ, Chuyên viên chẩn đoán hình ảnh.
- **Hành động:** Người dùng tải lên một hình ảnh X-quang/y tế, nhập câu hỏi chẩn đoán (ví dụ: "Bệnh nhân có dấu hiệu viêm phổi không?") và gửi yêu cầu.
- **Phản hồi hệ thống:** 
  - Trả về câu trả lời phân tích từ mô hình Medical VQA.
  - Đồng thời, hiển thị ngay các **Cảnh báo Drift (Drift Alerts)** dưới dạng thẻ màu sắc nếu hệ thống ngầm phát hiện bất thường:
    - *Màu Xanh (Safe):* Dữ liệu khớp với phân phối chuẩn y khoa.
    - *Màu Cam/Đỏ (Warning):* Cảnh báo Image Drift (ảnh mờ, sai thiết bị chụp), Text Drift (câu hỏi lạc đề, sai chuyên môn y khoa), hoặc Multimodal Drift. Cảnh báo này giúp bác sĩ cẩn trọng, không phụ thuộc mù quáng vào câu trả lời của AI.

### 1.2 Giám sát thống kê tổng quan (DashBoardPage)
**Người dùng mục tiêu:** Kỹ sư MLOps, Quản trị hệ thống bệnh viện.
- **Hành động:** Truy cập màn hình Dashboard để xem tình trạng "sức khỏe" và sự ổn định của toàn bộ hệ thống AI.
- **Phản hồi hệ thống:**
  - Hiển thị các **Biểu đồ Real-time** về tỷ lệ các truy vấn bị Drift (Image vs Text), biểu đồ Scatter Plot (PCA) so sánh phân phối hiện tại và phân phối gốc.
  - Theo dõi lịch sử khám (Audit Log): Xem lại các request cũ, hình ảnh bệnh nhân, câu hỏi đã đặt và điểm số Drift Score (p-value, distance) để phát hiện và ngăn chặn hiện tượng suy thoái mô hình (Model Decay).

### 1.3 Cấu hình hệ thống (ConfigPage)
**Người dùng mục tiêu:** Quản trị viên, Kỹ sư MLOps.
- **Hành động:** Điều chỉnh các thông số cấu hình cốt lõi của các thuật toán Drift Guard (như `p_threshold`, lựa chọn thuật toán kiểm định).
- **Phản hồi hệ thống:**
  - Áp dụng cấu hình mới theo thời gian thực mà không cần khởi động lại toàn bộ dịch vụ.
  - Cập nhật các Encoders hoặc bật/tắt luồng giám sát Multimodal Drift theo nhu cầu thực tế của bệnh viện.

### 1.4 Kiểm thử các mức độ Drift (TestPage)
**Người dùng mục tiêu:** Kỹ sư QA, Kỹ sư MLOps.
- **Hành động:** Đẩy vào hệ thống các gói dữ liệu giả lập (được sinh ra từ các kịch bản nhiễu Mild, Moderate, Severe).
- **Phản hồi hệ thống:**
  - Tự động chạy hàng loạt (Batch Processing) và hiển thị báo cáo log xem hệ thống Sentinel có "bắt" đúng các trường hợp bị lệch phân phối hay không. Đây là công cụ hữu hiệu để benchmark độ nhạy của thuật toán trước khi cập nhật lên môi trường Production.

---

## 2. Bảng Công Cụ, Công Nghệ & Khung MLOps (Technology Stack)

Hệ thống được thiết kế hoàn thiện theo chuẩn **Microservices Doanh nghiệp** để có thể triển khai thực tế vào các hạ tầng Y tế.

### 2.1 Mảng AI, Machine Learning & MLOps
| Tên Công Cụ / Model | Vai trò trong Đồ Án | Lý do lựa chọn (Dành cho vấn đáp) |
|---|---|---|
| **Scipy / Scikit-Learn** | Chạy các thuật toán kiểm định thống kê (Chi-Square, Kolmogorov-Smirnov, MMD, PCA). | Thư viện chuẩn ngành, nhẹ, xử lý ma trận và vector cực nhanh cho bài toán giám sát MLOps. |
| **HuggingFace Encoders** | Dịch Ảnh và Text thành Vector (Embeddings) để so sánh khoảng cách (VD: `rad-dino-maira-2`, `pubmedbert-base`). | Các model này được Fine-tune chuyên biệt trên dữ liệu Y khoa (PubMed, X-quang) nên hiểu và trích xuất đặc trưng y tế xuất sắc. |
| **LangGraph / OpenAI (Gemini)** | AI Agent đóng vai trò đánh giá ngữ nghĩa (LLM-as-a-judge) nếu phát hiện Drift toán học. | Cung cấp luồng suy luận phức tạp (Agentic Workflow) để phân tích chất lượng trả lời thay vì chỉ dùng các phép toán khô khan. |

### 2.2 Mảng Backend, Orchestrator & Lưu trữ (Infrastructure) 
| Tên Phần Mềm | Vai trò trong Đồ Án | Lý do lựa chọn (Dành cho vấn đáp) |
|---|---|---|
| **React + Vite** | Xây dựng 4 màn hình giao diện (Chat, Dashboard, Config, Test). | Render cực nhanh, kiến trúc Component dễ mở rộng cho dự án MLOps Dashboard phức tạp. |
| **Python (FastAPI)** | Server AI chính chạy các pipeline giám sát Drift. | Bất đồng bộ (Async) mạnh mẽ, là tiêu chuẩn số 1 hiện tại để deploy Machine Learning API. |
| **Node.js (Express)** | API Gateway nhận yêu cầu từ Frontend và điều phối tới AI Server. | Xử lý luồng dữ liệu (I/O) tốt, không bị nghẽn khi có nhiều yêu cầu gửi ảnh lớn cùng lúc. |
| **MongoDB** | Lưu trữ lịch sử báo cáo (Audit Logs) và thống kê Drift. | Cơ sở dữ liệu NoSQL linh hoạt, phù hợp với format JSON chứa kết quả phân tích nhiều tầng. |
| **Redis** | In-memory Cache, lưu trạng thái báo cáo Drift trung gian. | Tốc độ phản hồi cực cao, giúp UI dashboard cập nhật các chỉ số Real-time. |
| **MinIO** | Object Storage chứa ảnh y tế và các file numpy Reference data. | Hoạt động như AWS S3 nội bộ, dễ dàng quản lý khối lượng lớn file ảnh tĩnh và dữ liệu vector nhúng mà không ảnh hưởng tới Database chính. |
| **Docker Compose** | Đóng gói toàn bộ Frontend, Backend, AI Service và các DB. | Đảm bảo tính nhất quán của môi trường, triển khai (Deploy) mọi nơi dễ dàng chỉ với 1 lệnh `docker-compose up -d`. |
