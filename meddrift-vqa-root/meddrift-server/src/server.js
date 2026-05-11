import 'dotenv/config'; // Thay cho require("dotenv").config()
import express from 'express';
import { createProxyMiddleware } from 'http-proxy-middleware';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';

// Định nghĩa lại __dirname cho chuẩn ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://fastapi-service:8000';

app.use(cors());
app.use(express.json());

// Proxy cấu hình mới (đã sửa lỗi path stripping trước đó)
app.use(createProxyMiddleware({
    target: FASTAPI_URL,
    changeOrigin: true,
    secure: false,
    pathFilter: ['/api', '/health'],
}));

const frontendPath = path.join(__dirname, '../../meddrift-frontend/dist');
app.use(express.static(frontendPath));

// Sử dụng Regex bắt tất cả các route còn lại
app.get(/.*/, (req, res) => {
    if (req.url.startsWith('/api')) {
        return res.status(404).json({ error: "API route not found on Proxy" });
    }
    res.sendFile(path.join(frontendPath, "index.html"));
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
    console.log(`Proxying to FastAPI: ${FASTAPI_URL}`);
});