import os
import sys
import logging
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_models_exist():
 
    if not os.path.exists(settings.MODELS_CACHE_DIR):
        os.makedirs(settings.MODELS_CACHE_DIR)
        logger.info(f"Created model cache directory: {settings.MODELS_CACHE_DIR}")

    models_ready = True
    
    for folder_name, hf_id in settings.MODEL_MAP.items():
        save_path = os.path.join(settings.MODELS_CACHE_DIR, folder_name)
        
        if not os.path.exists(save_path) or not os.listdir(save_path):
            models_ready = False
            
 
            logger.info(f"Downloading {folder_name} ({hf_id})... This may take a minute.")
            print(f"Downloading {folder_name} model...") 
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(hf_id)
                model = AutoModel.from_pretrained(hf_id)
                
                tokenizer.save_pretrained(save_path)
                model.save_pretrained(save_path)
                
                logger.info(f"Saved {folder_name} to {save_path}")
                
            except Exception as e:
                logger.error(f"Failed to download {hf_id}. Error: {str(e)}")
                raise e 
        else:
            logger.info(f"Found existing model cache: {folder_name}")

    return True