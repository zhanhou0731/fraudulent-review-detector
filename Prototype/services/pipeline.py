import torch
import pandas as pd

from utils.preprocessor import clean_text
from services.feature_service import FeatureExtractor
from services.model_loader import get_manager

class FraudDetectionPipeline:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.manager = get_manager()

        if self.manager.hybrid_xgb:
            self.manager.hybrid_xgb.set_params(device='cpu')

        if self.manager.absa_baseline:
            try:
                self.manager.absa_baseline.set_params(device='cpu')
            except Exception:
                pass
                
        if self.manager.emo_baseline:
            try:
                self.manager.emo_baseline.set_params(device='cpu')
            except Exception:
                pass

    @staticmethod
    def compute_verdict(prob: float):
        verdict = "FRAUD" if prob > 0.5 else "GENUINE"
        confidence = prob if prob > 0.5 else (1.0 - prob)
        return verdict, round(float(confidence), 4)

    @staticmethod
    def build_single_output(verdict, confidence, prob):
        return {
                    "verdict": verdict,
                    "confidence": confidence,
                    "probability": prob
                }
        

    def predict_single(self, raw_text, use_hybrid=True, use_absa=False, use_emo=False):
        results = {}
        clean_input = clean_text(raw_text)
        fused_tensor = None
        absa_tensor = None
        emo_tensor = None

        if use_hybrid:
            absa_tensor, emo_tensor, fused_tensor = self.feature_extractor.extract_fused(clean_input)

        else:
            if use_absa:
                absa_tensor = self.feature_extractor.extract_absa(clean_input)
            if use_emo:
                emo_tensor = self.feature_extractor.extract_emotion(clean_input)

        
        if use_hybrid and fused_tensor is not None:
            if self.manager.hybrid_mlp and self.manager.hybrid_xgb:
                with torch.no_grad():
                    learned_features = self.manager.hybrid_mlp(fused_tensor).cpu().numpy()

                prob = self.manager.hybrid_xgb.predict_proba(learned_features)[0][1]
                prob = float(prob)
                verdict, confidence = self.compute_verdict(prob)
                results['Hybrid Model'] = self.build_single_output(verdict, confidence, prob)
            else:
                results['Hybrid Model'] = "Error: Model Not Loaded"

        if use_absa and absa_tensor is not None:
            if self.manager.absa_baseline:
                absa_np = absa_tensor.cpu().numpy()
                prob = self.manager.absa_baseline.predict_proba(absa_np)[0][1]
                prob = float(prob)
                verdict, confidence = self.compute_verdict(prob)
                results['ABSA Baseline'] = self.build_single_output(verdict, confidence, prob)
            else:
                results['ABSA Baseline'] = "Error: Model Not Loaded"

        if use_emo and emo_tensor is not None:
            if self.manager.emo_baseline:
                emo_np = emo_tensor.cpu().numpy()
                prob = self.manager.emo_baseline.predict_proba(emo_np)[0][1]
                prob = float(prob)
                verdict, confidence = self.compute_verdict(prob)
                results['Emotion Baseline'] = self.build_single_output(verdict, confidence, prob)
            else:
                results['Emotion Baseline'] = "Error: Model Not Loaded"

        return results

    
    def predict_batch(self, df, text_column_name, use_hybrid=True, use_absa=False, use_emo=False):

        df['clean_text'] = df[text_column_name].apply(clean_text)
        clean_texts = df['clean_text'].tolist()

        fused_list = None
        absa_list = None
        emo_list = None
        

        if use_hybrid:
            absa_list, emo_list, fused_list = self.feature_extractor.extract_batch_fused(clean_texts)
            

        else:
            if use_absa:
                absa_list = self.feature_extractor.extract_batch_absa(clean_texts)
            if use_emo:
                emo_list = self.feature_extractor.extract_batch_emotion(clean_texts)


        if use_hybrid and fused_list is not None:
            confidences = []
            verdicts = []
            if self.manager.hybrid_mlp and self.manager.hybrid_xgb:
                for t in fused_list:
                    with torch.no_grad():
                        learned_feat = self.manager.hybrid_mlp(t).cpu().numpy()
                    prob = self.manager.hybrid_xgb.predict_proba(learned_feat)[0][1]
                    verdict, confidence = self.compute_verdict(prob)

                    verdicts.append(verdict)
                    confidences.append(confidence)
    
                df['Hybrid_Confidence'] = confidences
                df['Hybrid_Verdict'] = verdicts

        if use_absa and absa_list is not None:
            confidences = []
            verdicts = []
            if self.manager.absa_baseline:
                for t in absa_list:
                    prob = self.manager.absa_baseline.predict_proba(t.cpu().numpy())[0][1]
                    verdict, confidence = self.compute_verdict(prob)

                    verdicts.append(verdict)
                    confidences.append(confidence)
    
                df['ABSA_Confidence'] = confidences
                df['ABSA_Verdict'] = verdicts

        if use_emo and emo_list is not None:
            confidences = []
            verdicts = []
            if self.manager.emo_baseline:
                for t in emo_list:
                    prob = self.manager.emo_baseline.predict_proba(t.cpu().numpy())[0][1]
                    verdict, confidence = self.compute_verdict(prob)

                    verdicts.append(verdict)
                    confidences.append(confidence)
    
                df['Emotion_Confidence'] = confidences
                df['Emotion_Verdict'] = verdicts


        return df
    
