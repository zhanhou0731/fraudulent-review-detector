import os
import torch

MODEL_MAP = {
    "absa_local": "yangheng/deberta-v3-base-absa-v1.1",
    "emotion_local": "j-hartmann/emotion-english-distilroberta-base"
}

# feature extraction models
#EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
#ABSA_MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
HYBRID_DIR = os.path.join(MODEL_DIR, 'hybrid_model')
MODELS_CACHE_DIR = os.path.join(MODEL_DIR, 'models_cache')

# feature extraction model paths
ABSA_PRETRAINED_PATH = os.path.join(MODELS_CACHE_DIR, 'absa_local')
EMOTION_PRETRAINED_PATH = os.path.join(MODELS_CACHE_DIR, 'emotion_local')

# Model Paths
HYBRID_MLP_PATH = os.path.join(HYBRID_DIR, 'Best_MLP_Stage1.pth')
HYBRID_XGB_PATH = os.path.join(HYBRID_DIR, 'Final_Hybrid_XGBoost.pkl')
MLP_CONFIG_PATH = os.path.join(HYBRID_DIR, 'best_mlp_config.csv')

ABSA_DIR = os.path.join(MODEL_DIR, 'ABSA_baseline_model')
ABSA_MODEL_PATH = os.path.join(ABSA_DIR, 'ABSA_Baseline_best.pkl')

EMO_DIR = os.path.join(MODEL_DIR, 'emotion_baseline_model')
EMO_MODEL_PATH = os.path.join(EMO_DIR, 'Emotion_Baseline_best.pkl')
# Device Settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dimensions
ABSA_DIM = 768
EMO_DIM = 768
INPUT_DIM = ABSA_DIM + EMO_DIM


#API
PUBLIC_API_REQUEST_URL = "https://zhanhou0731-fraudulent-review-detection-absa-emotion-api.hf.space"
PUBLIC_API_DOCS_URL = "https://zhanhou0731-fraudulent-review-detection-absa-emotion-api.hf.space/docs"
