Chào Kiên, việc chuyển sang **React** và **React Router** là một bước đi rất đúng đắn để nâng cấp dự án **MedDrift Sentinel** từ một bản demo sang một sản phẩm phần mềm hoàn chỉnh.

Dưới đây là hướng dẫn chi tiết từng bước để bạn tái cấu trúc dự án. Tôi đã trình bày theo định dạng Markdown để bạn dễ dàng lưu lại và theo dõi.

---

# 🚀 Hướng dẫn Chuyển đổi MedDrift Sentinel sang React

Lộ trình này giúp bạn tích hợp **React** vào cấu trúc sẵn có mà không làm hỏng logic của **Express Proxy** và **FastAPI AI Service**.

## 📂 1. Tổ chức lại Cấu trúc Thư mục

Bạn nên tách biệt rõ ràng mã nguồn Frontend và Backend. Cấu trúc mới sẽ như sau:

```text
meddrift-vqa-root/
├── meddrift-ai-service/        # FastAPI (Python)
├── meddrift-server/            # Express (Node.js Proxy)
└── meddrift-frontend/          # React Project (Vite)
    ├── src/
    │   ├── components/         # Các thành phần dùng chung (Badge, Button)
    │   ├── pages/              # 5 Màn hình chính (Chat, Dashboard,...)
    │   ├── hooks/              # Xử lý Logic API (useDriftStatus)
    │   └── App.jsx             # Cấu hình React Router
    └── vite.config.js

```

---

## 🛠️ 2. Khởi tạo Dự án React với Vite

Vite nhanh và hiện đại hơn `create-react-app`, rất phù hợp với các dự án AI cần tốc độ phản hồi cao.

1. Mở terminal tại thư mục gốc:
```bash
# Xóa thư mục cũ hoặc tạo mới
npm create vite@latest meddrift-frontend -- --template react
cd meddrift-frontend
npm install react-router-dom lucide-react axios

```


* *Ghi chú: `lucide-react` là bộ icon tối giản giống phong cách Claude.*



---

## 🛣️ 3. Cấu hình React Router (5 Màn hình)

Trong file `src/App.jsx`, bạn thiết lập điều hướng để quản lý 5 màn hình đã thiết kế.

```jsx
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import ChatPage from './pages/ChatPage';
import DashboardPage from './pages/DashboardPage';
import ConfigPage from './pages/ConfigPage';
// ... import các trang khác

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<ChatPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/config" element={<ConfigPage />} />
        {/* Thêm các route khác tại đây */}
      </Routes>
    </Router>
  );
}

```

---

## 🔗 4. Kết nối với Express Proxy

Bạn không cần đổi cổng trong code React. Thay vào đó, hãy cấu hình `vite.config.js` để khi bạn phát triển (development), mọi yêu cầu `/api` sẽ được gửi đến Express (cổng 3000).

```javascript
// meddrift-frontend/vite.config.js
export default defineConfig({
  server: {
    proxy: {
      '/api': 'http://localhost:3000',
      '/health': 'http://localhost:3000'
    }
  }
})

```

---

## 🎨 5. Xây dựng Component Chat (Phong cách Claude)

Tận dụng logic trích xuất embedding từ CLIP và BioBERT của backend. Dưới đây là cách bạn xử lý State trong React:

```jsx
// src/pages/ChatPage.jsx
import React, { useState } from 'react';
import axios from 'axios';

const ChatPage = () => {
  const [messages, setMessages] = useState([]);
  const [file, setFile] = useState(null);

  const handleSend = async (text) => {
    const formData = new FormData();
    formData.append('question', text);
    formData.append('image', file);

    // Gửi đến endpoint collect của FastAPI thông qua Proxy
    const response = await axios.post('/api/drift/collect', formData);
    
    // Cập nhật UI với kết quả và trạng thái drift
    setMessages([...messages, { 
      text: "AI Response...", 
      isDrift: response.data.alert 
    }]);
  };

  return (
    <div className="flex h-screen bg-[#f9f8f6]">
      {/* Sidebar & Chat Area UI ở đây */}
    </div>
  );
};

```

---

## 📉 6. Màn hình Dashboard & Registry

Đây là nơi bạn thể hiện khả năng Software Engineering. Bạn sẽ gọi API `/api/drift/algorithms` để hiển thị danh sách thuật toán mà hệ thống hỗ trợ (MMD, KS Test,...).

> **Gợi ý:** Sử dụng thư viện `Recharts` để vẽ biểu đồ P-value timeline. Nó rất nhẹ và tương thích hoàn hảo với React.

---

## 🚢 7. Quy trình Triển khai (Deployment Workflow)

Khi bạn đã hoàn thiện code React:

1. **Build Frontend:** Chạy `npm run build` trong thư mục `meddrift-frontend`. Nó sẽ tạo ra thư mục `dist`.
2. **Cấu hình Express:** Cập nhật `server.js` để phục vụ thư mục `dist` này.
```javascript
const frontendPath = path.join(__dirname, '../../meddrift-frontend/dist');
app.use(express.static(frontendPath));

```


3. **Chạy hệ thống:** * Bật FastAPI (Cổng 8000).
* Bật Express (Cổng 3000).
* Truy cập `localhost:3000` để trải nghiệm ứng dụng Fullstack hoàn chỉnh.



---

### 💡 Tại sao bước này quan trọng với bạn?

Việc sử dụng **Registry Pattern** kết hợp với **React** không chỉ giúp hệ thống linh hoạt (pluggable) mà còn là minh chứng mạnh mẽ cho kỹ năng lập trình của bạn khi ứng tuyển vào các vị trí AI Engineer Intern.

Bạn có muốn tôi đi sâu vào code mẫu của màn hình **Configuration** để hiển thị cách thay đổi thuật toán Drift từ UI không?