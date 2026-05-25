"""
Tải dữ liệu reference (.npy embeddings) từ kho Hugging Face Datasets về thư mục /data local.
"""

import os
import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Vui lòng cài đặt thư viện by running: pip install huggingface_hub")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download reference data for MedDrift")

    parser.add_argument("--repo_id", type=str, default="dhkien23/meddrift_vqa", help="Hugging Face Dataset repo ID")
    
    # Mặc định tải về project-root/data
    root_data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    parser.add_argument("--local_dir", type=str, default=str(root_data_dir), help="Local directory to store data")
    
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face Access Token (cần nếu repo là Private)")
    parser.add_argument("--include_org", action="store_true", help="Nếu truyền cờ này, sẽ tải luôn cả các ảnh gốc cực lớn trong thư mục org/")
    
    args = parser.parse_args()

    print(f"Bắt đầu tải dataset từ kho Hugging Face: {args.repo_id}...")
    if args.token:
        print("🔒 Đã phát hiện HF Token, sẽ tải với quyền xác thực (Private mode).")
    else:
        print("🔓 Không có Token, chế độ tải mặc định (Public mode).")
    
    print(f"Đích đến: {args.local_dir}\n")

    os.makedirs(args.local_dir, exist_ok=True)
    
    # Download dataset
    try:
        kwargs = {
            "repo_id": args.repo_id,
            "repo_type": "dataset",
            "local_dir": args.local_dir,
            "token": args.token
        }
        
        if not args.include_org:
            kwargs["ignore_patterns"] = ["**/org/**", "*/org/*", "org/*"]
            print("🚫 Bỏ qua tải thư mục 'org/' (ảnh/text gốc) để tiết kiệm thời gian và dung lượng.")
            print("➡️  (Nếu muốn tải cả ảnh gốc nặng, hãy thêm tham số --include_org khi chạy script)")

        snapshot_download(**kwargs)
        print("\n✅ Tải xuống thành công! Cấu trúc thư mục reference data đã sẵn sàng.")
        print("Bạn có thể chạy `docker-compose up -d` và cấu hình MinIO được rồi.")
    except Exception as e:
        print(f"\n❌ Lỗi khi tải dữ liệu: {e}")
        print("Nếu Dataset của bạn là Private, hãy chắc chắn bạn đã cấu hình Token Hugging Face.")
        print("Chạy lệnh: set HF_TOKEN=thêm_token_của_bạn_vào_đây && python download_data.py")

if __name__ == "__main__":
    main()
