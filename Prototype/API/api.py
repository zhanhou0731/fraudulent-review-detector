import socket
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import pandas as pd
import sys
import os
import logging

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)
from services.pipeline import FraudDetectionPipeline


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")



pipeline_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):

    global pipeline_engine
    logger.info("API SYSTEM: Initializing models...")
    try:
        pipeline_engine = FraudDetectionPipeline()
        logger.info("API SYSTEM: Models Loaded & Ready.")
    except Exception as e:
        logger.error(f"ERROR: {e}")
    yield


app = FastAPI(
    title="Fraudulent Review Detection API",
    description="FYP API: Detects fraudulent reviews using ABSA and emotion analysis.",
    version="1.2.0",
    lifespan=lifespan
)


MODEL_NAME_MAP = {
    'hybrid': 'Hybrid Model',
    'absa': 'ABSA Baseline',
    'emotion': 'Emotion Baseline'
}

class ReviewRequest(BaseModel):
    text: str
    model_type: str = "hybrid"
    metadata: Optional[Dict[str, Any]] = None

    @field_validator('model_type') 
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        allowed = ['hybrid', 'absa', 'emotion']
        if v.lower() not in allowed:
            raise ValueError(f"Invalid model_type. Must be one of: {allowed}")
        return v.lower()


class BatchReviewItem(BaseModel):
    id: Optional[str] = None
    text: str


class BatchReviewRequest(BaseModel):
    reviews: List[BatchReviewItem]
    model_type: str = "hybrid"

    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        allowed = ['hybrid', 'absa', 'emotion']
        if v.lower() not in allowed:
            raise ValueError(f"Invalid model_type. Must be one of: {allowed}")
        return v.lower()




@app.get("/")
async def root():
    return {"status": "online", "message": "API is running."}


@app.post("/predict")
async def predict_review(payload: ReviewRequest):
    if not pipeline_engine:
        raise HTTPException(status_code=503, detail="Model is loading...")

    try:
        m_type = payload.model_type.lower()
        use_hybrid = (m_type == 'hybrid')
        use_absa   = (m_type == 'absa')
        use_emo    = (m_type == 'emotion')

        results = pipeline_engine.predict_single(
            payload.text,
            use_hybrid=use_hybrid,
            use_absa=use_absa,
            use_emo=use_emo
        )


        target_key = MODEL_NAME_MAP[m_type]
        
        model_output = results.get(target_key)
        
        if model_output is None:
            raise HTTPException(
                status_code=500,
                detail=f"Model '{target_key}' failed to return a result."
            )
        
        if isinstance(model_output, str):
            raise HTTPException(status_code=500, detail=model_output)
        
        verdict = model_output["verdict"]
        confidence = model_output["confidence"]
        

        response = {
            "status": "success",
            "model_used": target_key,
            "result": {
                "verdict": verdict,
                "confidence_score": confidence
            }
        }

        if payload.metadata:
            response["metadata"] = payload.metadata
        
        return response

    except Exception as e:
        print(f"Error: {e}") 
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/predict-batch")
async def predict_batch(payload: BatchReviewRequest):
    if not pipeline_engine:
        raise HTTPException(status_code=503, detail="Model is loading...")

    try:
        m_type = payload.model_type.lower()
        
        texts = [item.text for item in payload.reviews]
        
        df = pd.DataFrame({"text": texts})
        
        results_df = pipeline_engine.predict_batch(
            df, 
            text_column_name="text", 
            use_hybrid=(m_type == "hybrid"), 
            use_absa=(m_type == "absa"), 
            use_emo=(m_type == "emotion")
        )

        response_results = []

        for i, item in enumerate(payload.reviews):
            result_item = {}
        
            if item.id:
                result_item["id"] = item.id
        
            if m_type == "hybrid":
                verdict = results_df.iloc[i]["Hybrid_Verdict"]
                confidence = results_df.iloc[i]["Hybrid_Confidence"]
        
            elif m_type == "absa":
                verdict = results_df.iloc[i]["ABSA_Verdict"]
                confidence = results_df.iloc[i]["ABSA_Confidence"]
        
            else:
                verdict = results_df.iloc[i]["Emotion_Verdict"]
                confidence = results_df.iloc[i]["Emotion_Confidence"]
        
            result_item.update({
                "verdict": verdict,
                "confidence_score": confidence
            })
        
            response_results.append(result_item)

        return {
            "status": "success",
            "model_used": MODEL_NAME_MAP[m_type],
            "total_processed": len(response_results),
            "results": response_results
        }

    except Exception as e:
        logger.error(f"Batch Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        


if __name__ == "__main__":
    print("\n" + "="*50)
    print("FRAUDULENT REVIEW DETECTION API - LAUNCHER")
    print("="*50)
    print("Select Hosting Mode:")
    print("  [1] Localhost Only (127.0.0.1)")
    print("  [2] LAN             (0.0.0.0)")

    print("="*50)

    while True:
        mode = input("Enter choice (1 or 2): ").strip()
        
        if mode in ["1", "2"]:
            break
        print("Invalid input. Please enter '1' or '2' only.")

        
    if mode == "2":
        host = "0.0.0.0"
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            my_ip = s.getsockname()[0]
            s.close()
            print(f"\nLAN Connection Active.")
            print(f"Connect to: http://{my_ip}:8000/predict or http://{my_ip}:8000/predict-batch")
        except:
            print("\nCould not detect IP. Check 'ipconfig'.")
            
    else:
        host = "127.0.0.1"
        print("\nLocalhost Active.")
        print("You can connect at: http://127.0.0.1:8000/predict or http://127.0.0.1:8000/predict-batch")

    print(f"Server starting on {host}:8000...")
    uvicorn.run(app, host=host, port=8000)