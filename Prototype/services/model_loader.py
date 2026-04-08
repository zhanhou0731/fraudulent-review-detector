import torch
import torch.nn as nn
import pandas as pd
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

class DynamicFusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, fusion_dim, dropout_rate):
        super(DynamicFusionMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, 256))
        self.layers.append(nn.BatchNorm1d(256))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))
        current_dim = 256
        if hidden_layers == 2:
            self.layers.append(nn.Linear(256, 128))
            self.layers.append(nn.BatchNorm1d(128))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            current_dim = 128
        self.fusion_layer = nn.Linear(current_dim, fusion_dim)
        self.fusion_act = nn.ReLU()
        self.head = nn.Linear(fusion_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        fused_features = self.fusion_act(self.fusion_layer(x))
        return fused_features

class ModelManager:
    def __init__(self):
        self.hybrid_mlp = None
        self.hybrid_xgb = None
        self.absa_baseline = None
        self.emo_baseline = None

    def load_hybrid(self):
        if self.hybrid_mlp is not None: return
            
        try:
            config = pd.read_csv(settings.MLP_CONFIG_PATH).iloc[0].to_dict()
            self.hybrid_mlp = DynamicFusionMLP(
                settings.INPUT_DIM, int(config['hidden_layers']), 
                int(config['fusion_dim']), config['dropout']
            ).to(settings.DEVICE)
            self.hybrid_mlp.load_state_dict(torch.load(settings.HYBRID_MLP_PATH, map_location=settings.DEVICE))
            self.hybrid_mlp.eval()
            
            with open(settings.HYBRID_XGB_PATH, 'rb') as f:
                self.hybrid_xgb = pickle.load(f)
        except Exception as e:
            print(f"Error loading Hybrid Model: {e}")

    def load_baselines(self):
        if self.absa_baseline is None and os.path.exists(settings.ABSA_MODEL_PATH):
            try:
                with open(settings.ABSA_MODEL_PATH, 'rb') as f:
                    self.absa_baseline = pickle.load(f)
            except Exception as e:
                print(f"Error loading ABSA Baseline: {e}")
        
        if self.emo_baseline is None and os.path.exists(settings.EMO_MODEL_PATH):
            try:
                with open(settings.EMO_MODEL_PATH, 'rb') as f:
                    self.emo_baseline = pickle.load(f)
            except Exception as e:
                print(f"Error loading Emotion Baseline: {e}")


def get_manager():
    manager = ModelManager()
    manager.load_hybrid()
    manager.load_baselines()
    return manager