# Problem Definition Document
## Business context and motivation 
Tại Việt Nam, tình trạng quá tải y tế đang là một thách thức hệ thống nghiêm trọng. Với tỷ lệ bác sĩ trên dân số chỉ đạt khoảng 15/10.000 (thống kê năm 2025), áp lực lên đội ngũ y tế là cực kỳ lớn. Tại các bệnh viện tuyến Trung ương, tình trạng bệnh nhân phải chờ đợi kéo dài và ghép giường đã trở thành rào cản cho chất lượng chăm sóc sức khỏe.

Phân tích sâu về nguyên nhân, có tới 80% người dân tìm đến bệnh viện khi chỉ mắc bệnh nhẹ hoặc cần chăm sóc sức khỏe ban đầu, trong khi thực tế chỉ có 5% thực sự cần đến dịch vụ chuyên sâu tại tuyến cuối. Thực trạng này tạo ra một lượng "nhu cầu ảo" khổng lồ do hành vi tìm kiếm và sử dụng dịch vụ y tế không hợp lý, gây lãng phí nguồn lực quốc gia.

Trong kỷ nguyên chuyển đổi số, trí tuệ nhân tạo (AI) nổi lên như một giải pháp tiềm năng để giải quyết bài toán này thông qua các hệ thống hỏi đáp sức khỏe thông minh (Medical QA). Các nền tảng như Babylon Health hay Ada Health đã bắt đầu tích hợp AI để sàng lọc bệnh nhân từ xa. Tuy nhiên, thách thức lớn nhất hiện nay là ngay cả các mô hình lớn như Med-PaLM vẫn chủ yếu được huấn luyện trên các tập dữ liệu tĩnh.

Dựa trên nghiên cứu của Dr. Declan Kelly, tri thức y tế đang trải qua sự bùng nổ chưa từng có: từ chu kỳ nhân đôi 3,5 năm vào năm 2010 đã rút ngắn xuống chỉ còn 73 ngày vào năm 2020. Điều này dẫn đến hiện tượng "Chu kỳ bán rã của tri thức", nơi 50% kiến thức y khoa có thể trở nên lỗi thời chỉ sau 18-24 tháng. Hơn nữa, theo Liu và cộng sự (2025), sự suy giảm kiến thức không diễn ra đồng đều; những mảng kiến thức ít được sử dụng thường xuyên (Distant knowledge) có nguy cơ sai lệch cao gấp 2,31 lần.

Chính vì vậy, việc xây dựng một hệ thống không chỉ biết trả lời mà còn có khả năng tự giám sát và phát hiện sự lỗi thời của tri thức (Model Drift Detection) là yêu cầu cấp thiết. Dự án MedDrift-Sentinel được phát triển nhằm cung cấp một giải pháp NLP tin cậy, giúp sàng lọc nhu cầu y tế ban đầu, đồng thời đảm bảo tính cập nhật liên tục của tri thức, góp phần trực tiếp vào mục tiêu giảm tải hệ thống y tế và nâng cao an toàn cho bệnh nhân.

## Target users or stakeholders 
MedDrift sẽ là một công cụ tích hợp với các mô hình hỏi đáp y tế nhằm hỗ trợ sự sai lệch trong dữ liêu huấn luyệ. Vì thế, MedDrift sẽ đóng một vai trò gián tiếp hơn với các đối tượng sử dụng chính sẽ được chia thành hai nhóm chính:
Bác sĩ/Các chuyên viên y khoa sử dụng trong việc khám chữa bệnh: MedDrift sẽ luôn song hành cùng mô hình hỏi đáp để phát hiện kịp thời sự lệch miền dữ liệu, từ đó giúp các nhân viên y tế có thể đưa ra các hướng xử lí phù hợp hơn.
Ban Quản trị Bệnh viện / Bộ y tế: Hệ thống đảm bảo tính minh bạch thông qua việc giám sát liên tục (Monitoring), đáp ứng các yêu cầu về an toàn và bảo mật thông tin y tế. Ngoài ra, nó giúp tối ưu hóa quy trình, giảm thời gian chờ đợi của bệnh nhân và quản lý rủi ro sai sót y khoa do kiến thức cũ.

## Description of the problem being solved
Vấn đề cốt lõi: Các mô hình AI y tế hiện nay chỉ được huấn luyện trên dữ liệu tĩnh, trong khi tri thức y khoa nhân đôi sau mỗi 73 ngày.Chi tiết kỹ thuật: Sự lỗi thời của kiến thức (Knowledge Decay) dẫn đến việc mô hình đưa ra câu trả lời sai lệch nhưng vẫn tự tin cao.Minh chứng: Trích dẫn nghiên cứu của Liu (2025) về việc kiến thức "Distant" dễ bị suy giảm với $OR=2.31$. Đây chính là bài toán mà alibi-detect sẽ giải quyết thông qua việc phát hiện sai lệch phân phối (Drift Detection).

## Explanation of why NLP is required 
Sự đa dạng và nhạy cảm của ngữ nghĩa	
Xử lý "Dữ liệu ngoài vùng hiểu biết" (OOD)
Vấn đề: Người dùng Việt Nam có thể hỏi những câu "không đầu không đuôi" hoặc dùng tiếng lóng địa phương (ví dụ: "tào tháo đuổi", "trúng gió")	
Khả năng giải thích (Explainability) cho Stakeholders
Success metrics, including: 
- Business metrics (e.g., cost reduction, time saved, efficiency) ◦
- Technical metrics (e.g., accuracy, F1-score, latency)

