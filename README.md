# MedDrift-Sentinel

MedDrift-Sentinel is an AI-powered medical image and text drift detection and monitoring architecture. Designed to monitor production models for conceptual and data drift, it utilizes dual drift tracking (image and text) to ensure reliability and robustness in medical VQA (Visual Question Answering) and related domains.

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

- `meddrift-ai-service/`: Core FastAPI AI monitoring pipeline.
- `meddrift-server/`: Node.js backend proxy.
- `meddrift-frontend/`: Vite + React web interface.
- `configs/`: YAML configurations for the drift pipelines (`drift_config.yaml`).
- `docker-compose.yml`: Multi-container orchestration setup.

## Prerequisites

- **Docker** and **Docker Compose** installed on your machine.
- Node.js & npm (if running frontend/backend outside of Docker).
- Python 3.10+ (if running the AI service standalone).

## Getting Started

### 1. Environment Configuration
Ensure you have the required environment variables. Create a `.env` file at the root of the project (if not existing) with the database credentials:

```env
DB_USER=admin
DB_PASS=secretpassword
REDIS_PASS=redispassword
MINIO_USER=minioadmin
MINIO_PASS=miniopassword
```

*Note: The `meddrift-ai-service` also expects its own `.env` file inside `meddrift-ai-service/.env` which contains API keys for LangChain, Gemini, or OpenRouter.*

### 2. Run the Application with Docker Compose
Start all services in detached mode (including frontend, backend, AI service, MongoDB, Redis, and MinIO):

```bash
docker-compose up -d --build
```

### 3. Access the Services
Once all containers are up and running, you can access the applications at:
- **Frontend App**: [http://localhost:3000](http://localhost:3000)
- **Node.js Backend API**: [http://localhost:5000](http://localhost:5000)
- **FastAPI AI Service (Docs)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **MinIO Admin Console**: [http://localhost:9001](http://localhost:9001)

##  Drift Detection Flow

1. **Input Data**: The system accepts new medical images and text-based VQA questions.
2. **Image & Text Processing**: Dual processing to extract visual features and text embeddings.
3. **Statistical Testing**: Automated batch-based triggers execute tests (e.g., Chi-Square) against reference distributions stored in MinIO.
4. **Agentic Evaluation**: A LangGraph drift agent leverages LLMs to evaluate the qualitative significance of any mathematical drift detected.
5. **Alerts & Reporting**: The system aggregates the drift status (image, text, or both), saves the reports to MongoDB, caches ongoing monitoring states in Redis, and pushes updates to the frontend dashboard.