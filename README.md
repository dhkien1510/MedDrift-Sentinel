# MedDrift-Sentinel

MedDrift-Sentinel is an AI-powered medical image and text drift detection and monitoring architecture. Designed to monitor production models for conceptual and data drift, it utilizes dual drift tracking (image and text) to ensure reliability and robustness in medical VQA (Visual Question Answering) and related domains.

## ⚠️ Important Note on Encoders and Embeddings

While MedDrift-Sentinel supports various encoders for evaluating data drift, **you cannot simply provide any HuggingFace model name** to the configuration files (`drift_config.yaml`). The system strictly requires **pre-computed embeddings** to exist in the `data/reference_data` and `data/drift_scenarios` directories.

**Supported Encoders (Pre-computed in this repository):**
- **Image Encoders**:
  - `microsoft/rad-dino-maira-2`
  - `microsoft/rad-dino`
  - `facebook/dinov2-base`
  - `google/vit-base-patch16-224`
- **Text Encoders**:
  - `NeuML/pubmedbert-base-embeddings`
  - `NeuML/pubmedbert-base-embeddings-8M`
  - `NeuML/pubmedbert-base-embeddings-matryoshka`
  - `dmis-lab/biobert-v1.1`
  - `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`
  - `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`
  - `pritamdeka/S-PubMedBert-MS-MARCO`

> **Note on Multimodal Drift Detection:**
> If you are enabling the Multimodal Drift detection path (`multimodal.enabled: true` in config), please note that the joint PCA fusion embeddings have **only** been computed for the combination of:
> - **Image**: `microsoft/rad-dino-maira-2`
> - **Text**: `NeuML/pubmedbert-base-embeddings`
>
> Ensure your `drift_config.yaml` is set to these exact encoders if utilizing multimodal drift. Otherwise, the system will fail to find the PCA bundle.

## Architecture

The system consists of three main application components and robust data infrastructure orchestrated via Docker Compose.

### Application Services
- **`meddrift-ai-service` (FastAPI / Python)**: The core AI engine managing the LangGraph drift agent, integration with Gemini/OpenRouter, and calculating statistical tests (like Chi-Square) to detect data/concept drifts.
- **`meddrift-server` (Express / Node.js)**: The backend gateway routing requests and handling data flow between the frontend and the AI service.
- **`meddrift-frontend` (React / Vite)**: The user interface to interact with the drift monitoring system, view charts, and configure alerts.

### Infrastructure (Containerized)
- **MongoDB**: Persistent database for tracking drift history and saving structured reports.
- **Redis**: Fast cache to manage state, intermediate drift status, and speed up query performance.
- **MinIO**: High-performance object storage system for medical images and reference datasets.

## Directory Structure

```
project-root/
├── meddrift-ai-service/   # Core AI engine (FastAPI / Python) — src/, scripts/, Dockerfile
├── meddrift-server/       # Node.js backend gateway (Express)
├── meddrift-frontend/     # React + Vite web dashboard
├── data/                  # Reference embeddings & drift scenarios (downloaded via script)
├── models/                # Model documentation — encoder selection & evaluation
├── configs/               # YAML configuration for drift pipelines (drift_config.yaml)
├── doc/                   # Architecture & project documentation
├── docker-compose.yml     # Multi-container orchestration
├── .env.example           # Environment variable template
└── README.md
```

## Prerequisites

- **Docker** and **Docker Compose** installed on your machine.
- Node.js & npm (if running frontend/backend outside of Docker).
- Python 3.10+ (if running the AI service standalone).

## Getting Started

### 1. Environment Configuration
Ensure you have the required environment variables. We have provided template files for you to use.

1. **Root Database Credentials:**
   Copy the root environment example:
   ```bash
   cp .env.example .env
   ```
   *Edit `.env` to configure your database & object storage passwords if you wish to change defaults.*

2. **AI Service API Keys:**
   Copy the AI Service environment example:
   ```bash
   cp meddrift-ai-service/.env.example meddrift-ai-service/.env
   ```
   *Edit `meddrift-ai-service/.env` to insert your OPENROUTER_API_KEY, GEMINI_API_KEY, etc.*

### 2. Prepare Reference Data
Before running the system, you need the reference embeddings (which are too large to host on GitHub). We host these via Hugging Face.
Run the provided script to download the datasets into the data/ folder:
```bash
pip install huggingface_hub
python meddrift-ai-service/scripts/download_data.py
```


### 3. Run the Application with Docker Compose
Start all services in detached mode:

```bash
docker-compose up -d --build
```

### 4. Initialize Data into MinIO Storage
Once the containers are successfully running, you must transfer the downloaded reference datasets into MinIO so the AI service can access them dynamically. Run the migration script inside the Docker container:

```bash
docker-compose exec service python scripts/migrate_npy_to_minio.py
```

### 5. Access the Services
Once all containers are up and running, you can access the applications at:
- **Frontend App**: [http://localhost:3000](http://localhost:3000)
- **Node.js Backend API**: [http://localhost:5000](http://localhost:5000)
- **FastAPI AI Service (Docs)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **MinIO Admin Console**: [http://localhost:9001](http://localhost:9001)

## Drift Detection Flow

1. **Input Data**: The system accepts new medical images and text-based VQA questions.
2. **Image & Text Processing**: Dual processing to extract visual features and text embeddings.
3. **Statistical Testing**: Automated batch-based triggers execute tests (e.g., Chi-Square) against reference distributions stored in MinIO.
4. **Agentic Evaluation**: A LangGraph drift agent leverages LLMs to evaluate the qualitative significance of any mathematical drift detected.
5. **Alerts & Reporting**: The system aggregates the drift status (image, text, or both), saves the reports to MongoDB, caches ongoing monitoring states in Redis, and pushes updates to the frontend dashboard.

## Dependency Installation

### Running via Docker (Recommended)
All dependencies are automatically installed inside the Docker containers when you run the build command. No local dependency management is required:
```bash
docker-compose up -d --build
```

### Running Locally (Without Docker)
If you wish to run the components independently on your host machine:

1. **AI Service (Python)**
   ```bash
   cd meddrift-ai-service
   pip install -r requirements.txt
   ```
2. **Node.js Server**
   ```bash
   cd meddrift-server
   npm install
   ```
3. **React Frontend**
   ```bash
   cd meddrift-frontend
   npm install
   ```

## Model Training & Fine-Tuning
This project uses **zero-shot** / **pre-trained** image and text encoders (e.g., `microsoft/rad-dino`, `dmis-lab/biobert-v1.1`) to generate embeddings for drift detection.
Therefore, **no scratch training or fine-tuning is required natively** by this system to function. The architecture relies on these pre-trained foundational models to calculate statistical feature drifts dynamically. If you wish to use a custom fine-tuned model, you must train it separately and pre-compute its embeddings into the `data/reference_data` directory.

## Inference & Running the Deployed System
The system is built to perform real-time monitoring and inference continuously in production.
- Start the entire stack using Docker Compose:
  ```bash
  docker-compose up -d --build
  ```
- Trigger inference API calls via the Node.js server (Port `5000`) which proxies to the FastAPI service.
- Use the Frontend Dashboard (Port `3000`) to visualize live drift detection reports.

## Deployment Method
The project is containerized using **Docker** and orchestrated via **Docker Compose**, making it highly reproducible and easily deployable across any cloud or local environment supporting Docker engines.
The deployment creates a network interconnecting:
- **Stateful stores**: MongoDB (database), Redis (cache), MinIO (object storage).
- **Stateless services**: FastAPI (AI inference proxy), Express Server (API gateway), Vite/React (Frontend static/dev server).