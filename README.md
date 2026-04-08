# Detecting Fraudulent Customer Reviews Using Aspect-Based Sentiment and Emotion Analysis

This repository contains the code, data, and deployment files for a multidimensional fraud detection framework. The project synthesises Aspect-Based Sentiment Analysis (ABSA) and  emotion embeddings to identify human-written deceptive opinion spam.

The final hybrid model utilises a multi-stream neural architecture (DeBERTa-v3 and DistilRoBERTa), fused via a custom Multi-Layer Perceptron (MLP), and classified using XGBoost. The model achieved a peak F1-Score of **0.8631** on the unseen test set and has been deployed as both an interactive prototype and a containerized web API.

---

## 📂 Repository Architecture

To ensure a clean separation between data science research and software engineering deployment, this repository is divided into two primary environments:

### 1. `/Training` (Research & Reproducibility)
This directory contains the entire machine learning lifecycle, from data preprocessing to statistical validation. 
* Contains its own `requirements.txt` specifically for heavy data science libraries (Jupyter, Matplotlib, Seaborn, etc.).
* Houses the sequential Jupyter Notebooks (`1_preprocessing.ipynb` to `5_hybrid_model_training.ipynb`).
* Exports the final trained weights (`.pkl`, `.pth`) to be used by the prototype.

### 2. `/Prototype` (Deployment & MLOps)
This directory contains the production-ready code for both the interactive Streamlit GUI and the scalable FastAPI backend. 
* **`requirements.txt`**: Contains all dependencies to run both the Streamlit app and the API locally.
* **`requirements-api.txt` & `Dockerfile`**: Specifically optimised for containerizing the API using a lightweight `python:3.10.11-slim` base image and CPU-only PyTorch for cost-effective industry deployment.


---

## ⚙️ Prerequisites
To run this project locally without dependency conflicts, ensure your system meets the following requirement:
* **Python Version:** `3.10.11` (Strictly recommended to ensure compatibility with the PyTorch CPU variant and Hugging Face Transformers).
* **Docker:** Required only if you intend to build and run the containerized API (Part 3).

---

## Start Guide

### Part 1: Running the Interactive Prototype (Streamlit)
The prototype provides a user-friendly GUI for real-time inference, batch processing, and multi-model comparison.

1. Navigate to the Prototype folder:
   ```bash
   cd Prototype
   ```
2. Install the full requirements (includes Streamlit and API dependencies):
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```

### Part 2: Running the API Locally (FastAPI)
If you wish to run the backend API directly without Docker, you can use the built-in interactive launcher.
1. Ensure you are in the Prototype folder and requirements are installed.
2. Run the API script:
   ```bash
   python api.py
   ```
3. The terminal will prompt you to select a hosting mode:
   - Localhost Only (127.0.0.1): Best for local testing.
   - LAN (0.0.0.0): Best if you want to access the API from another device on the same Wi-Fi network.
4. Once running, you can make requests at: ```http://127.0.0.1:8000/predict``` and ```http://127.0.0.1:8000/predict-batch```

### Part 3: Deploying the API (Docker)
The FastAPI backend is optimised to run as a headless, containerized service for industry adoption.
1. Navigate to the Prototype folder:
   ```bash
   cd Prototype
   ```
2. Build the Docker image (this uses requirements-api.txt to minimise the container footprint):
   ```bash
   docker build -t fraud-detection-api .
   ```
3. Run the Docker container:
   ```bash
   docker run -d -p 8000:8000 fraud-detection-api
   ```

---

## 📖 API Usage Tutorial

Once the API is running (either locally or via Docker), you can send HTTP POST requests to the endpoints. The API supports three `model_type` parameters: `"hybrid"` (Default), `"absa"`, or `"emotion"`.

### 1. Single Review Inference
**Endpoint:** `POST /predict`

**Request Payload (JSON):**
```json
{
  "text": "The hotel room was absolutely filthy and the staff was extremely rude. Do not stay here!",
  "model_type": "hybrid",
  "metadata": {"source": "mobile_app", "user_id": "104A"}
}
```

**Successful Response:**
```json
{
  "status": "success",
  "model_used": "Hybrid Model",
  "result": {
    "verdict": "Fraudulent",
    "confidence_score": 0.892
  },
  "metadata": {
    "source": "mobile_app",
    "user_id": "104A"
  }
}
```

### 2. Batch Review Inference
**Endpoint:** `POST /predict-batch`

**Request Payload (JSON):**
```json
{
  "model_type": "hybrid",
  "reviews": [
    {
      "id": "REV-001",
      "text": "Amazing experience, the breakfast was delicious and the bed was very comfortable."
    },
    {
      "id": "REV-002",
      "text": "Best hotel ever. I love this place so much. Everyone should come here now."
    }
  ]
}
```

**Successful Response:**
```json
{
  "status": "success",
  "model_used": "Hybrid Model",
  "total_processed": 2,
  "results": [
    {
      "id": "REV-001",
      "verdict": "Genuine",
      "confidence_score": 0.941
    },
    {
      "id": "REV-002",
      "verdict": "Fraudulent",
      "confidence_score": 0.785
    }
  ]
}
```