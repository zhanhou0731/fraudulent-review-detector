import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from utils.setup import ensure_models_exist

class FeatureExtractor:
    def __init__(self):
        ensure_models_exist()
        self.absa_tokenizer, self.absa_model = self._load_absa_model()
        self.emo_tokenizer, self.emo_model = self._load_emo_model()
        
    def _load_absa_model(_self):
        tokenizer = AutoTokenizer.from_pretrained(settings.ABSA_PRETRAINED_PATH, fix_mistral_regex=True)
        model = AutoModel.from_pretrained(settings.ABSA_PRETRAINED_PATH)
        model.to(settings.DEVICE)
        model.eval()
        return tokenizer, model
        

    def _load_emo_model(_self):
        tokenizer = AutoTokenizer.from_pretrained(settings.EMOTION_PRETRAINED_PATH, fix_mistral_regex=True)
        model = AutoModel.from_pretrained(settings.EMOTION_PRETRAINED_PATH)
        model.to(settings.DEVICE)
        model.eval()
        return tokenizer, model


    def _get_embedding(self, text, tokenizer, model):
        if isinstance(text, str):
            text = [text]
            
        inputs = tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(settings.DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
        
        return outputs.last_hidden_state[:, 0, :]

    def _get_batch_embedding(self, text_list, tokenizer, model, batch_size=8):
        all_embeddings = []
        total = len(text_list)

        for i in range(0, total, batch_size):
            batch_texts = text_list[i : i + batch_size]

            inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(settings.DEVICE)

            with torch.no_grad():
                outputs = model(**inputs)
                batch_emb = outputs.last_hidden_state[:, 0, :].cpu()
                all_embeddings.append(batch_emb)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        final_tensor = torch.cat(all_embeddings, dim=0).to(settings.DEVICE)
        return final_tensor

    def extract_absa(self, text):
        return self._get_embedding(text, self.absa_tokenizer, self.absa_model)

    def extract_emotion(self, text):
        return self._get_embedding(text, self.emo_tokenizer, self.emo_model)

    def extract_fused(self, text):
        absa_emb = self.extract_absa(text)
        emo_emb = self.extract_emotion(text)

        fused_tensor = torch.cat((absa_emb, emo_emb), dim=1)
        
        return absa_emb, emo_emb, fused_tensor

    def extract_batch_absa(self, text_list):
        emb_tensor = self._get_batch_embedding(text_list, self.absa_tokenizer, self.absa_model)
        return [t.unsqueeze(0) for t in emb_tensor]

    def extract_batch_emotion(self, text_list):
        emb_tensor = self._get_batch_embedding(text_list, self.emo_tokenizer, self.emo_model)
        return [t.unsqueeze(0) for t in emb_tensor]
    
    def extract_batch_fused(self, text_list):
        absa_tensor = self._get_batch_embedding(text_list, self.absa_tokenizer, self.absa_model)
        emo_tensor = self._get_batch_embedding(text_list, self.emo_tokenizer, self.emo_model)
        
        fused_tensor = torch.cat((absa_tensor, emo_tensor), dim=1)

        absa_list = [t.unsqueeze(0) for t in absa_tensor]
        emo_list = [t.unsqueeze(0) for t in emo_tensor]
        fused_list = [t.unsqueeze(0) for t in fused_tensor]
        
        return absa_list, emo_list, fused_list
    