# API Reference Documentation
**System:** Fraudulent Review Detection API  
**Version:** 1.2.0  

This document serves as the official reference for the Fraudulent Review Detection API. It is divided into two sections: **Part 1** for API users  and **Part 2** for API Maintainers.


---


## Part 1: User Guide (For API users)

The API provides RESTful endpoints to classify text-based reviews as "Genuine" or "Fraudulent". By default, the API is accessible at `http://localhost:8000` (or your machine's LAN IP).

### 1.1 Authentication
Currently, the API is open for internal use. No API Key or Bearer Token is required in the headers.

### 1.2 Endpoints Overview

#### 1. Health Check
* **Endpoint:** `GET /`
* **Description:** Verifies that the API server is online and reachable.
* **Response:**
  ```json
  {
    "status": "online", 
    "message": "API is running."
  }
  ```

#### 2. Single Review Inference
* **Endpoint:** `POST /predict`
* **Description:** Analyzes a single review text string.
* **Request Body (JSON):**
  * `text` (string, required): The review content.
  * `model_type` (string, optional): The model to use. Accepted values: `"hybrid"` (default), `"absa"`, `"emotion"`.
  * `metadata` (dict, optional): Custom key-value pairs (e.g., user ID, source) that will be echoed back in the response.

**Example Request (Python `requests`):**
```python
import requests

url = "http://localhost:8000/predict"
payload = {
    "text": "[Insert Review Here]",
    "model_type": "hybrid",
    "metadata": {"source": "mobile_app"}
}
response = requests.post(url, json=payload)
print(response.json())
```

**Example Response:**
```json
{
  "status": "success",
  "model_used": "Hybrid Model",
  "result": {
    "verdict": "FRAUD",
    "confidence_score": 0.8924
  },
  "metadata": {
    "source": "mobile_app"
  }
}
```

#### 3. Batch Review Inference
* **Endpoint:** `POST /predict-batch`
* **Description:** Analyzes multiple reviews in a single request. Highly optimized for processing `.csv` or tabular data.
* **Request Body (JSON):**
  * `model_type` (string, optional): `"hybrid"` (default), `"absa"`, `"emotion"`.
  * `reviews` (list, required): A list of objects containing `text` (required) and `id` (optional).

**Example Request (cURL):**
```bash
curl -X 'POST' \
  'http://localhost:8000/predict-batch' \
  -H 'Content-Type: application/json' \
  -d '{
  "model_type": "hybrid",
  "reviews": [
    {"id": "REV-01", "text": "[Insert Review Here]"},
    {"id": "REV-02", "text": "[Insert Review Here]"}
  ]
}'
```

**Example Response:**
```json
{
  "status": "success",
  "model_used": "Hybrid Model",
  "total_processed": 2,
  "results": [
    {
      "id": "REV-01",
      "verdict": "GENUINE",
      "confidence_score": 0.9412
    },
    {
      "id": "REV-02",
      "verdict": "FRAUD",
      "confidence_score": 0.7854
    }
  ]
}
```

### 1.3 HTTP Status Codes
* **200 OK:** Request processed successfully.
* **422 Unprocessable Entity:** Invalid JSON schema or incorrect `model_type` requested.
* **500 Internal Server Error:** Machine learning model failure or feature extraction error.
* **503 Service Unavailable:** The API is online, but the ML models are still loading into RAM. Please wait a few seconds and try again.


---

## Part 2: Developer Guide (For System Maintainers)

This section details the internal architecture, design patterns, and maintenance protocols for the backend application.

### 2.1 Core Architecture & Data Flow
The data flow follows this sequence:

1. **Routing & Validation (`api.py`):** FastAPI handles the HTTP request. Pydantic strictly validates the JSON payload to ensure `text` exists and `model_type` is valid.
2. **Orchestration (`services/pipeline.py`):** The `FraudDetectionPipeline` acts as the central controller. It receives the raw text, sanitizes it, and decides which downstream ML services to trigger based on the requested `model_type`.
3. **Feature Extraction (`services/feature_service.py`):** Acts as the interface to Hugging Face. The text is tokenized, padded/truncated to 512 tokens, and passed through `DeBERTa-v3` (for ABSA) and/or `DistilRoBERTa` (for Emotion). It extracts the `[CLS]` token's 768-dimensional hidden state (`last_hidden_state[:, 0, :]`).
4. **Trained Models (`services/model_loader.py`):** The `ModelManager` loads the trained models in this project. For the Hybrid model, the tensors are concatenated (1536 dimensions), passed through the PyTorch `DynamicFusionMLP` (compressing to 64 dimensions), and finally classified by the XGBoost estimator.
5. **Classification Pipeline:** The `pipeline.py` translates the raw probability float into a human-readable `verdict` and `confidence_score` and returns it to `api.py` for JSON serialisation.

### 2.2 Memory Management & Lifecycle Strategy
Transformer models are inherently memory-intensive. Loading two separate transformer architectures into RAM simultaneously requires strict memory management to prevent Out-Of-Memory (OOM) errors and inference latency.

* **The Singleton Pattern:** The `pipeline_engine` is instantiated globally as `None`. It is only populated once during the server startup. 
* **FastAPI Lifespan Context (`@asynccontextmanager`):** We utilize the ASGI lifespan protocol. Upon server startup (`uvicorn.run`), the `FraudDetectionPipeline` is instantiated. All `.pkl` (XGBoost) and `.pth` (PyTorch) weights, along with the Hugging Face tokenizers, are loaded into RAM. 
* **Safe Degradation:** If a user sends a `POST /predict` request while the models are still being loaded into RAM, the API will not crash. Instead, the global variable check will fail, cleanly returning a `503 Service Unavailable` status, prompting the client to retry in a few seconds.

### 2.3 Hardware Optimization & Deployment Constraints
This API is intentionally engineered to run on generic, CPU-only cloud instances (e.g., AWS EC2 t3.medium or standard Docker environments) to drastically reduce industry deployment costs.

* **Strict CPU Casting:** * In `pipeline.py`, the XGBoost estimators are explicitly forced to bypass GPU drivers via `self.manager.hybrid_xgb.set_params(device='cpu')`.
  * In `feature_service.py`, PyTorch tensors are routed according to `settings.DEVICE` (which is configured to `cpu` in `config/settings.py`).
* **Containerization Footprint:** The `Dockerfile` relies on `python:3.10.11-slim`. Furthermore, `requirements-api.txt` specifically requests the CPU-only wheel of PyTorch (`--extra-index-url https://download.pytorch.org/whl/cpu`). Failure to use this specific index URL during Docker builds will result in downloading the 2GB+ NVIDIA CUDA toolkit, bloated container sizes, causing deployment latency.

### 2.4 Maintenance: Adding or Updating Models
If future research yields an updated model architecture or improved weights, follow this strict protocol to integrate it into the API without breaking existing endpoints:

**Step 1: Update the Weights**
Place the newly trained `.pkl` or `.pth` file into the `/models` directory. Update the path variables in `config/settings.py` to point to the new file.

**Step 2: Update the ModelLoader (`model_loader.py`)**
If you are adding a completely new model (e.g., an LLM-based classifier), instantiate it in the `ModelManager` class:
```python
class ModelManager:
    def __init__(self):
        # ... existing models ...
        self.new_llm_model = None

    def load_new_model(self):
        with open(settings.NEW_LLM_PATH, 'rb') as f:
            self.new_llm_model = pickle.load(f)
```

**Step 3: Update Pydantic Validation (`api.py`)**
Add the new key to the validation arrays to prevent the API from rejecting the request with a `422` error:
```python
MODEL_NAME_MAP = {
    'hybrid': 'Hybrid Model',
    'absa': 'ABSA Baseline',
    'emotion': 'Emotion Baseline',
    'llm': 'New LLM Model'
}

class ReviewRequest(BaseModel):
    # ...
    @field_validator('model_type') 
    def validate_model_type(cls, v: str) -> str:
        allowed = ['hybrid', 'absa', 'emotion', 'llm']
```

**Step 4: Route the Logic (`pipeline.py`)**
In `predict_single` and `predict_batch`, add the routing logic to pass the text to your new model and format the output using the `build_single_output()` helper function.

### 2.5 Logging and Error Tracing
The API utilizes Python's standard `logging` library. Do not use standard `print()` statements for debugging in the production environment.
* **Initialisation Logs:** Model loading sequences trigger `logger.info()`.
* **Runtime Errors:** If a batch processing job fails (e.g., pandas encounters a malformed array), it triggers `logger.error(f"Batch Error: {e}")` and returns a `500` status code with the stack trace safely abstracted away from the end-user.
* **Checking Logs in Docker:** To view the API runtime logs when deployed via Docker, run: `docker logs fraud-detection-api --follow`.