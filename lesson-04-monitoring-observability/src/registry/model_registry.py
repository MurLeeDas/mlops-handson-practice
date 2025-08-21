import os
import json
import joblib
from typing import Dict, List, Optional
from datetime import datetime

class ModelRegistry:
    """Central registry for managing multiple model versions"""
    
    def __init__(self, models_dir="./models"):
        self.models_dir = models_dir
        self.loaded_models = {}
        self.model_metadata = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all available model versions"""
        print("Loading models from registry...")
        
        for version_dir in os.listdir(self.models_dir):
            version_path = os.path.join(self.models_dir, version_dir)
            
            if os.path.isdir(version_path):
                model_path = os.path.join(version_path, "model.pkl")
                metadata_path = os.path.join(version_path, "metadata.json")
                
                if os.path.exists(model_path):
                    try:
                        # Load model package
                        model_package = joblib.load(model_path)
                        self.loaded_models[version_dir] = model_package
                        
                        # Load metadata
                        if os.path.exists(metadata_path):
                            with open(metadata_path, 'r') as f:
                                self.model_metadata[version_dir] = json.load(f)
                        
                        print(f"✅ Loaded model {version_dir}: {model_package['metadata']['algorithm']}")
                        
                    except Exception as e:
                        print(f"❌ Failed to load model {version_dir}: {e}")
    
    def get_model(self, version: str):
        """Get a specific model version"""
        return self.loaded_models.get(version)
    
    def get_all_versions(self) -> List[str]:
        """Get all available model versions"""
        return list(self.loaded_models.keys())
    
    def get_model_metadata(self, version: str) -> Dict:
        """Get metadata for a specific model version"""
        return self.model_metadata.get(version, {})
    
    def get_best_model(self) -> str:
        """Get the version with highest AUC score"""
        best_version = None
        best_auc = 0
        
        for version, metadata in self.model_metadata.items():
            auc = metadata.get('auc_score', 0)
            if auc > best_auc:
                best_auc = auc
                best_version = version
                
        return best_version
    
    def predict(self, version: str, customer_data: dict):
        """Make prediction using specific model version"""
        model_package = self.get_model(version)
        if not model_package:
            raise ValueError(f"Model version {version} not found")
        
        # Convert to DataFrame if needed
        import pandas as pd
        if isinstance(customer_data, dict):
            customer_data = pd.DataFrame([customer_data])
        
        # Preprocess data using model's components
        X = self._preprocess_for_model(customer_data, model_package)
        
        # Make prediction
        model = model_package['model']
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0, 1]
        
        return {
            'prediction': bool(prediction),
            'probability': float(probability),
            'model_version': version,
            'algorithm': model_package['metadata']['algorithm']
        }
    
    def _preprocess_for_model(self, df, model_package):
        """Preprocess data using model-specific encoders and scalers"""
        import pandas as pd
        df_processed = df.copy()
        
        # Apply label encoders
        categorical_columns = ['internet_service', 'contract_type', 'payment_method']
        for col in categorical_columns:
            if col in df_processed.columns:
                encoder = model_package['label_encoders'].get(f"{model_package['metadata']['version']}_{col}")
                if encoder:
                    df_processed[col] = encoder.transform(df_processed[col])
        
        # Feature engineering
        df_processed['avg_monthly_spend'] = df_processed['total_charges'] / df_processed['months_active']
        df_processed['support_per_month'] = df_processed['support_tickets'] / df_processed['months_active']
        
        # Advanced features for v2.0
        if model_package['metadata']['version'] == "v2.0":
            df_processed['customer_lifetime_value'] = df_processed['monthly_charges'] * df_processed['months_active']
            df_processed['support_intensity'] = (df_processed['support_tickets'] > 3).astype(int)
            df_processed['high_value_customer'] = (df_processed['monthly_charges'] > 80).astype(int)
        
        # Select features
        feature_columns = model_package['feature_columns']
        X = df_processed[feature_columns]
        
        # Apply scaling
        scaler = model_package.get('scaler')
        if scaler:
            X = scaler.transform(X)
        
        return X