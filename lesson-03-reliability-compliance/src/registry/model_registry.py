# Simple model registry for lesson 3
import os
import json
import joblib
from typing import Dict, List

class ModelRegistry:
    def __init__(self, models_dir="./models"):
        self.models_dir = models_dir
        self.loaded_models = {}
        self.model_metadata = {}
        self.load_all_models()
    
    def load_all_models(self):
        print("Loading models from registry...")
        
        for version_dir in os.listdir(self.models_dir):
            version_path = os.path.join(self.models_dir, version_dir)
            
            if os.path.isdir(version_path):
                model_path = os.path.join(version_path, "model.pkl")
                
                if os.path.exists(model_path):
                    try:
                        model_package = joblib.load(model_path)
                        self.loaded_models[version_dir] = model_package
                        self.model_metadata[version_dir] = model_package.get('metadata', {})
                        print(f"✅ Loaded model {version_dir}")
                    except Exception as e:
                        print(f"❌ Failed to load model {version_dir}: {e}")
    
    def get_all_versions(self) -> List[str]:
        return list(self.loaded_models.keys())
    
    def predict(self, version: str, customer_data: dict):
        # Implementation same as in router above
        pass
    
    def get_best_model(self) -> str:
        """Get the version with highest AUC score"""
        if not self.model_metadata:
            return None
        
        best_version = None
        best_auc = 0
        
        for version, metadata in self.model_metadata.items():
            auc = metadata.get('auc_score', 0)
            if auc > best_auc:
                best_auc = auc
                best_version = version
                
        return best_version or list(self.loaded_models.keys())[0] if self.loaded_models else None
