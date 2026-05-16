FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .

# Cài torch riêng trước — layer này được cache độc lập
# Chỉ tải lại khi bạn đổi version torch
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Cài phần còn lại
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]