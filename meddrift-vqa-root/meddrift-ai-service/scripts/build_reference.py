import numpy as np
import requests
import os
import yaml
import torch
from PIL import Image
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel 
from tqdm import tqdm

def load_config():
    current_script_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_script_path)
    service_dir = os.path.dirname(current_dir)
    root_dir = os.path.dirname(service_dir)
    config_dir = os.path.join(root_dir, 'configs/drift_config.yaml')
    print("Tiến hành đọc tập tin drift_config.yaml...")
    try:
        with open(config_dir, "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config
    except Exception as e:
        print(f"Lỗi: {e}")
        return None



def build_image_reference():
    # 1. Khởi tạo thiết bị và cấu hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = config['image']['encoder']
    save_path = config['reference']['image_path']
    num_samples = config['reference']['num_samples']
    dataset_path = config['reference']['data']
    batch_size = 16 

    # 2. Tải model và processor
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()

    # 3. Tải dữ liệu
    dataset = load_dataset(dataset_path, split="train")
    dataset = dataset.select(range(num_samples))
    
    embedded_list = []

    # 4. Xử lý theo batch
    for i in tqdm(range(0, len(dataset), batch_size), desc="Extracting Image Embeddings"):
        batch = dataset[i : i + batch_size]
        # Chuyển đổi list ảnh thành tensor
        inputs = processor(images=batch['image'], return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            # Sử dụng get_image_features để lấy visual embedding
            image_features = model.get_image_features(**inputs)
            
            # CLIP thường yêu cầu normalize embedding để tính similarity chính xác hơn
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            embedded_list.append(image_features.cpu().numpy())
    
    # 5. Gộp và lưu
    final_embeddings = np.vstack(embedded_list)
    np.save(save_path, final_embeddings)
    print(f"\nĐã lưu reference image embeddings tại: {save_path}")
    print(f"Shape: {final_embeddings.shape}")   


def build_questions_reference():
    # 1. Khởi tạo tham số
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    language_model = config['text']['encoder']
    dataset_path = config['reference']['data']
    current_script_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_script_path)
    service_dir = os.path.dirname(current_dir)
    root_dir = os.path.dirname(service_dir)
    save_path = os.path.join(root_dir, config['reference']['data_dir'], config['reference']['text_file'])
    print(save_path)
    num_samples = config['reference']['num_samples']
    batch_size = 32 # Tối ưu hóa tốc độ

    # 2. Tải dữ liệu và model
    dataset = load_dataset(dataset_path, split="train").select(range(num_samples))
    tokenizer = AutoTokenizer.from_pretrained(language_model)
    model = AutoModel.from_pretrained(language_model).to(device)
    model.eval()

    all_embeddings = []

    # 3. Xử lý theo từng batch (Thay vì loop từng câu)
    for i in range(0, len(dataset), batch_size):
        batch_questions = dataset['question'][i : i + batch_size]
        
        inputs = tokenizer(
            batch_questions,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding= "max_length"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        # Lấy CLS token của toàn bộ batch [batch_size, 0, hidden_size]
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)

    # 4. Gộp lại thành một ma trận duy nhất và lưu
    final_reference = np.vstack(all_embeddings)
    np.save(save_path, final_reference)
    print(f"Đã lưu reference embedding với shape: {final_reference.shape}")


if __name__ == "__main__":
    config  = load_config()
    # build_image_reference()
    build_questions_reference()