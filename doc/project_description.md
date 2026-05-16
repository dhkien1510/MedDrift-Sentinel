# Problem Definition Document: MedDrift-Sentinel

## 1. Business Context and Motivation 
Tại Việt Nam và trên thế giới, tình trạng quá tải y tế đang là một thách thức hệ thống nghiêm trọng. Trong kỷ nguyên chuyển đổi số, Trí tuệ nhân tạo Đa phương thức (Multimodal AI) - tiêu biểu là các hệ thống Medical Visual Question Answering (VQA) dựa trên các Large Vision-Language Models (như **LLaVA-Med**) - nổi lên như một giải pháp đột phá. Bác sĩ có thể tải lên một bức ảnh X-quang và đặt câu hỏi bằng ngôn ngữ tự nhiên để nhận được chẩn đoán tức thì.

Tuy nhiên, rào cản lớn nhất ngăn cản việc đưa các mô hình VQA này vào môi trường lâm sàng thực tế chính là hiện tượng **Data Drift** (Sự dịch chuyển phân phối dữ liệu đầu vào) và **Model Decay** (Sự suy thoái mô hình dẫn đến ảo giác - Hallucination). 
Dữ liệu y tế trong thực tế luôn biến động: máy chụp X-quang được nâng cấp khiến độ tương phản ảnh thay đổi (Image Acquisition Shift), hoặc bác sĩ bắt đầu sử dụng các từ lóng/viết tắt mới trong câu hỏi so với thời điểm model được huấn luyện (Semantic/Text Shift). Khi đối mặt với những dữ liệu bị "drift" này, các mô hình VQA thường không báo lỗi mà tự tin đưa ra những câu trả lời sai lệch, đe dọa trực tiếp đến tính mạng bệnh nhân.

Dự án **MedDrift-Sentinel** được phát triển để giải quyết triệt để rủi ro này. Đây là một hệ thống **Dual-Modality MLOps**, đóng vai trò là "người lính gác" đứng trước mô hình VQA để liên tục giám sát, phát hiện sự dịch chuyển dữ liệu đầu vào (cả Ảnh và Câu hỏi) và đánh giá chất lượng câu trả lời sinh ra.

## 2. Target Users & Stakeholders 
MedDrift-Sentinel đóng vai trò là hệ thống "Copilot cho Copilot" trong các bệnh viện, phục vụ 2 nhóm đối tượng chính:
*   **Bác sĩ & Chuyên viên chẩn đoán hình ảnh (End-users):** Những người trực tiếp sử dụng hệ thống VQA. MedDrift sẽ hiển thị cảnh báo (Alert) ngay trên màn hình nếu phát hiện tấm ảnh X-quang họ vừa tải lên hoặc câu hỏi họ vừa đặt bị lệch khỏi chuẩn phân phối y khoa an toàn, giúp họ không bị phụ thuộc một cách mù quáng vào AI.
*   **Kỹ sư AI / Đội ngũ quản trị CNTT Bệnh viện (MLOps / Admins):** Cung cấp các biểu đồ giám sát Real-time về sức khỏe của AI. Giúp họ quyết định chính xác thời điểm nào cần thu thập dữ liệu mới và Fine-tune (huấn luyện lại) mô hình LLaVA.

## 3. Project Description

MedDrift-Sentinel là một hệ thống **Dual-Modality MLOps** đóng vai trò liên tục giám sát và phát hiện sự dịch chuyển dữ liệu đầu vào (Data Drift) cho các mô hình AI y tế (Medical VQA). Khi triển khai thực tế, độ chính xác của mô hình VQA sẽ giảm dần do sự thay đổi của môi trường. Hệ thống xây dựng một đường ống (Pipeline) giám sát bằng các kiểm định thống kê đa dạng (như MMD, Kolmogorov-Smirnov) để theo dõi song song: Hình ảnh (Image Drift) và Văn bản (Text Drift). Đặc biệt, hệ thống hỗ trợ Unsupervised Drift Evaluation (LLM-as-a-judge) để đánh giá câu trả lời mà không cần Ground Truth.

**Cách thức tạo ra Data Drift (Drift Generation):**
Để đánh giá hiệu năng của các thuật toán phát hiện, dự án cung cấp bộ công cụ giả lập các mức độ drift (Mild, Moderate, Severe):
- **`scripts/build_drift_data.py`**: Chịu trách nhiệm sinh ra các tập dữ liệu bị nhiễu (drift) theo kịch bản:
  - **Image Drift**: Sử dụng các phép biến đổi ảnh (Augmentation) deterministic thông qua thư viện PIL. Ví dụ: *Level 1* (thay đổi độ sáng, tương phản), *Level 2* (Gaussian blur, noise), *Level 3* (đảo ngược pixel, thêm các noise patch lớn che khuất chi tiết).
  - **Text Drift**: Sử dụng LLM (thông qua API OpenRouter) để biến đổi các câu hỏi gốc thành các dạng khác nhau. Ví dụ: *Level 1* (sử dụng từ viết tắt, tiếng lóng y khoa), *Level 2* (chuyển sang câu hỏi đa ngôn ngữ hoặc từ ngữ rườm rà), *Level 3* (câu hỏi hoàn toàn lạc đề hoặc vô nghĩa).
- **`scripts/build_multimodal_reference.py`**: Quản lý việc tạo ra các không gian đa phương thức (Multimodal Drift). Script này kết hợp đặc trưng (embeddings) của ảnh và văn bản gốc (đã được tạo bởi `build_reference.py`), sau đó áp dụng thuật toán giảm chiều dữ liệu PCA để học một ma trận chiếu chung (Joint PCA Bundle). Dữ liệu này dùng làm phân phối chuẩn để tính toán khoảng cách (distance) trên cả hai chiều Image-Text đồng thời.

## 4. Explanation of Why NLP & CV (Computer Vision) are Required
MedDrift-Sentinel bắt buộc phải có sự kết hợp sâu sắc giữa xử lý ngôn ngữ tự nhiên (NLP) và thị giác máy tính (CV) vì:
*   **Đặc thù đầu vào đa phương thức (Multimodal VQA):** Mô hình AI sinh câu trả lời dựa trên sự tương tác giữa Ảnh và Câu hỏi. Nếu chỉ kiểm tra một chiều sẽ bỏ lọt rủi ro.
*   **Sự phức tạp của ngôn ngữ Y khoa (NLP):** Cần NLP (BioBERT) để hiểu và trích xuất đặc trưng (Embeddings) của các thuật ngữ chuyên ngành phức tạp, nhận diện ra sự thay đổi ngữ nghĩa, chứ không chỉ so sánh từ khóa đơn thuần.
*   **Đánh giá Output không cần nhãn (Unsupervised NLP):** Cần ứng dụng các tham số của LLM (Perplexity, Cosine Similarity qua Sentence-Transformers) để định lượng mức độ "tự tin" và văn phong của văn bản chẩn đoán được sinh ra.

## 5. Success Metrics

#### A. Business Metrics (Chỉ số nghiệp vụ & Giá trị)
*   **Tỷ lệ ngăn chặn can thiệp sai (Safe Intervention Rate):** Số lượng request rủi ro cao (Drift nặng) bị hệ thống cảnh báo kịp thời trước khi gửi đến bác sĩ.
*   **Chi phí tối ưu hóa (Cost/Time Efficiency):** Tiết kiệm thời gian rà soát dữ liệu thủ công của đội ngũ Kỹ sư IT.

#### B. Technical Metrics (Chỉ số kỹ thuật)
*   **Độ trễ giám sát (Monitoring Latency):** Thời gian trích xuất Feature và chạy MMD test phải mất `< 200ms` để không làm chậm luồng trải nghiệm real-time.
*   **Sự tương quan giữa P-value và Accuracy (Drift vs Performance Correlation):** Chứng minh được bằng số liệu phân tích: Khi P-value của thuật toán MMD có dấu hiệu giảm dần (Drift tăng), F1-Score/Accuracy của LLAVA-Med tương ứng có dấu hiệu suy thoái.
*   **Unsupervised Alert Rate:** Tính ổn định trong đánh giá Self-Consistency (Cosine Similarity > 0.85 trên các dữ liệu Safe).

## 6. Project Use Cases

1. **Giám sát chất lượng dữ liệu thời gian thực (Real-time Data Monitoring):** 
   - **Tình huống:** Bác sĩ tại một bệnh viện sử dụng ứng dụng Medical VQA. Họ tải lên ảnh X-quang chụp từ thiết bị đời mới (độ phân giải cao hơn, tương phản khác biệt) và đặt câu hỏi chứa các thuật ngữ viết tắt địa phương.
   - **Hành động của Sentinel:** Hệ thống phân tích feature embeddings (Image + Text) của yêu cầu này, đối chiếu với phân phối chuẩn ban đầu.
   - **Kết quả:** Nếu khoảng cách (drift score) vượt ngưỡng an toàn, hệ thống lập tức hiển thị cảnh báo ngay trên màn hình để bác sĩ cân nhắc trước khi hoàn toàn tin tưởng vào kết quả chẩn đoán của AI.

2. **Tự động kích hoạt chu trình huấn luyện lại (Automated Retraining Trigger):**
   - **Tình huống:** Sau một tháng triển khai, bệnh viện tiếp nhận nhiều bệnh nhân quốc tế, dẫn đến ngôn ngữ truy vấn đa dạng hơn (Text Drift tăng mạnh).
   - **Hành động của Sentinel:** Khi tổng số lượng requests bị lệch phân phối vượt qua một tỷ lệ nhất định (ví dụ 100 batch liên tục cảnh báo drift), hệ thống sẽ gửi báo cáo chi tiết về mức độ suy thoái mô hình cho đội ngũ MLOps.
   - **Kết quả:** Đội ngũ MLOps sử dụng ngay các báo cáo này cùng tập dữ liệu lưu trữ trong MinIO để tiến hành fine-tune mô hình LLaVA cho phù hợp với phân phối dữ liệu mới.
