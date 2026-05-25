"""
scripts/migrate_npy_to_minio.py
=================================
Script chạy một lần duy nhất để upload toàn bộ file .npy
đang nằm trên local disk lên MinIO.

Chạy TRƯỚC khi deploy phiên bản mới:
  docker compose exec service python scripts/migrate_npy_to_minio.py

Hoặc chạy ngoài Docker (cần set env vars):
  MINIO_ENDPOINT=localhost:9000 python scripts/migrate_npy_to_minio.py

Script sẽ:
  1. Scan thư mục data/reference_data/ cho reference .npy
  2. Scan thư mục data/drift_scenarios/ cho scenario .npy
  3. Upload tất cả lên MinIO bucket 'reference-data'
  4. Báo cáo kết quả

An toàn để chạy lại — MinIO put_object sẽ overwrite file cũ.
"""

import sys
import os

# Thêm src/ vào path để import được db/minio_client
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from pathlib import Path
from db.minio_client import (
    init_buckets,
    upload_npy,
    reference_object_name,
    scenario_object_name,
)

ROOT = Path(__file__).parent.parent  # project root

def safe_encoder(model_id: str) -> str:
    return model_id.replace("/", "--")

def reverse_safe_encoder(safe: str) -> str:
    return safe.replace("--", "/")


def migrate_reference(data_dir: Path) -> list[tuple[str, str]]:
    """Upload reference .npy files."""
    results = []

    for sub in ["image", "text", 'res']:
        sub_dir = data_dir / sub
        if not sub_dir.exists():
            # Thử flat layout cũ
            for fname in ["ref_images.npy", "ref_questions.npy"]:
                flat = data_dir / fname
                if flat.exists():
                    is_img  = fname == "ref_images.npy"
                    encoder = "unknown/encoder"  # cần set thủ công nếu dùng flat layout
                    obj     = reference_object_name(is_img, encoder, fname)
                    try:
                        arr = np.load(str(flat))
                        upload_npy(arr, obj)
                        results.append(("OK", str(flat), obj))
                    except Exception as e:
                        results.append(("ERR", str(flat), str(e)))
            continue

        is_image = sub == "image"
        for enc_dir in sub_dir.iterdir():
            if not enc_dir.is_dir():
                continue
            encoder = reverse_safe_encoder(enc_dir.name)
            for npy_file in enc_dir.glob("*.npy"):
                obj = reference_object_name(is_image, encoder, npy_file.name)
                try:
                    arr = np.load(str(npy_file))
                    upload_npy(arr, obj)
                    results.append(("OK", str(npy_file), obj))
                    print(f"  ✅ {npy_file.name}  →  {obj}")
                except Exception as e:
                    results.append(("ERR", str(npy_file), str(e)))
                    print(f"  ❌ {npy_file.name}: {e}")
    return results


def migrate_scenarios(scenario_dir: Path) -> list[tuple[str, str]]:
    """Upload scenario .npy files."""
    results = []

    for sub in ["image", "text"]:
        sub_dir = scenario_dir / sub
        if not sub_dir.exists():
            continue
        is_image = sub == "image"
        for enc_dir in sub_dir.iterdir():
            if not enc_dir.is_dir():
                continue
            encoder = reverse_safe_encoder(enc_dir.name)
            for npy_file in enc_dir.glob("*.npy"):
                scenario_name = npy_file.stem  # filename sans .npy
                obj = scenario_object_name(is_image, encoder, scenario_name)
                try:
                    arr = np.load(str(npy_file))
                    upload_npy(arr, obj)
                    results.append(("OK", str(npy_file), obj))
                    print(f"  ✅ {npy_file.name}  →  {obj}")
                except Exception as e:
                    results.append(("ERR", str(npy_file), str(e)))
                    print(f"  ❌ {npy_file.name}: {e}")
    return results


def main():
    print("=" * 60)
    print("MedDrift — Migrate .npy files to MinIO")
    print("=" * 60)

    # Init buckets
    print("\n[1/4] Khởi tạo MinIO buckets...")
    init_buckets()
    print("  OK")

    # Migrate reference data
    ref_dir = ROOT.parent / "data" / "reference_data"
    print(f"\n[2/4] Migrate reference embeddings từ {ref_dir} ...")
    if ref_dir.exists():
        ref_results = migrate_reference(ref_dir)
        ok  = sum(1 for r in ref_results if r[0] == "OK")
        err = sum(1 for r in ref_results if r[0] == "ERR")
        print(f"  → {ok} file OK, {err} lỗi")
    else:
        print(f"  ⚠️  Thư mục không tồn tại: {ref_dir}")

    # Migrate scenarios
    scenario_dir = ROOT.parent / "data" / "drift_scenarios"
    print(f"\n[3/4] Migrate scenario embeddings từ {scenario_dir} ...")
    if scenario_dir.exists():
        sc_results = migrate_scenarios(scenario_dir)
        ok  = sum(1 for r in sc_results if r[0] == "OK")
        err = sum(1 for r in sc_results if r[0] == "ERR")
        print(f"  → {ok} file OK, {err} lỗi")
    else:
        print(f"  ⚠️  Thư mục không tồn tại: {scenario_dir}")

    print("\n[4/4] Xong!")
    print("=" * 60)
    print("Sau khi migration thành công, bạn có thể:")
    print("  - Xóa volumes data/ nếu muốn (không bắt buộc)")
    print("  - Không cần mount ./data:/data trong docker-compose.yml nữa")
    print("=" * 60)


if __name__ == "__main__":
    main()