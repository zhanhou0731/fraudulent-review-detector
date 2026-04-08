import os
import sys
import shutil
import logging
import time
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_models_exist():
    os.makedirs(settings.MODELS_CACHE_DIR, exist_ok=True)

    for folder_name, hf_id in settings.MODEL_MAP.items():
        save_path = os.path.join(settings.MODELS_CACHE_DIR, folder_name)

        if not os.path.exists(save_path) or not os.listdir(save_path):
            os.makedirs(save_path, exist_ok=True)  # create folder only when needed
            logger.info(f"Downloading {folder_name} ({hf_id})...")
            print(f"Downloading {folder_name} model...")

            try:
                tokenizer = AutoTokenizer.from_pretrained(hf_id)
                model = AutoModel.from_pretrained(hf_id)

                tokenizer.save_pretrained(save_path)
                model.save_pretrained(save_path)

                logger.info(f"Saved {folder_name} to {save_path}")

            except Exception as e:
                logger.error(f"Failed to download {hf_id}: {e}")
                shutil.rmtree(save_path, ignore_errors=True)  # remove incomplete folder
                continue

        else:
            logger.info(f"Found existing model cache: {folder_name}")

    return True