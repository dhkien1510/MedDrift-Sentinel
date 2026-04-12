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

## 3. Description of the Problem Being Solved
Vấn đề cốt lõi: Khi đem một mô hình Medical VQA triển khai thực tế, độ chính xác của nó sẽ giảm dần một cách tĩnh lặng do sự thay đổi của môi trường.

*   **Chi tiết kỹ thuật:** Bài toán yêu cầu phát hiện sự lệch miền dữ liệu (OOD - Out of Distribution) trên cả 2 chiều dữ liệu không đồng nhất: Hình ảnh (Pixel metrics) và Văn bản (Text Semantics) mà không làm chậm trải nghiệm của User. 
*   **Giải pháp (MedDrift-Sentinel):** Xây dựng một đường ống (Pipeline) giám sát bằng cơ chế kiểm định thống kê MMD (Maximum Mean Discrepancy). 
    *   Sử dụng **CLIP** để giám sát sự phân phối của Ảnh (Image Drift).
    *   Sử dụng **BioBERT** để giám sát sự dịch chuyển ngữ nghĩa của Câu hỏi (Text Drift).
    *   Cung cấp cơ chế **Unsupervised Drift Evaluation (LLM-as-a-judge & Self-Consistency)** để đo lường mức độ Hallucination của câu trả lời mà không cần Ground Truth (Bác sĩ chấm điểm).

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
