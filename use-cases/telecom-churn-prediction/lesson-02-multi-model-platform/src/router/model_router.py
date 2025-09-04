from typing import Dict, List
import random
import json
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from registry.model_registry import ModelRegistry

class ModelRouter:
    """Intelligent routing between multiple model versions"""
    
    def __init__(self, models_dir="./models", config_path="./configs/routing_config.json"):
        self.registry = ModelRegistry(models_dir)
        self.config_path = config_path
        self.routing_config = self.load_routing_config()
        self.prediction_log = []
    
    def load_routing_config(self) -> Dict:
        """Load traffic routing configuration"""
        default_config = {
            "traffic_split": {
                "v1.0": 50,  # 50% traffic
                "v1.1": 30,  # 30% traffic  
                "v2.0": 20   # 20% traffic
            },
            "auto_promotion": {
                "enabled": True,
                "min_predictions": 100,
                "promotion_threshold": 0.02  # 2% AUC improvement
            },
            "fallback_model": "v1.0"
        }
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}  # Merge with defaults
        except FileNotFoundError:
            # Create default config file
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def route_prediction(self, customer_data: dict, strategy="traffic_split") -> Dict:
        """Route prediction request to appropriate model"""
        
        if strategy == "traffic_split":
            version = self._select_model_by_traffic_split()
        elif strategy == "best_model":
            version = self.registry.get_best_model()
        elif strategy == "round_robin":
            version = self._select_model_round_robin()
        else:
            version = self.routing_config["fallback_model"]
        
        try:
            # Make prediction
            result = self.registry.predict(version, customer_data)
            
            # Log prediction for analysis
            self._log_prediction(customer_data, result, version)
            
            # Add routing metadata
            result.update({
                'routing_strategy': strategy,
                'timestamp': datetime.now().isoformat(),
                'router_version': '2.0'
            })
            
            return result
            
        except Exception as e:
            # Fallback to default model
            fallback_version = self.routing_config["fallback_model"]
            result = self.registry.predict(fallback_version, customer_data)
            result.update({
                'routing_strategy': 'fallback',
                'original_error': str(e),
                'fallback_model': fallback_version
            })
            return result
    
    def _select_model_by_traffic_split(self) -> str:
        """Select model based on configured traffic percentages"""
        traffic_split = self.routing_config["traffic_split"]
        
        # Create weighted list
        weighted_versions = []
        for version, weight in traffic_split.items():
            if version in self.registry.get_all_versions():
                weighted_versions.extend([version] * weight)
        
        return random.choice(weighted_versions) if weighted_versions else self.routing_config["fallback_model"]
    
    def _select_model_round_robin(self) -> str:
        """Simple round-robin selection"""
        versions = self.registry.get_all_versions()
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        version = versions[self._round_robin_index % len(versions)]
        self._round_robin_index += 1
        return version
    
    def _log_prediction(self, customer_data: dict, result: dict, version: str):
        """Log prediction for performance analysis"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_version': version,
            'prediction': result['prediction'],
            'probability': result['probability'],
            'customer_id': customer_data.get('customer_id', 'unknown'),
            'features': {k: v for k, v in customer_data.items() if k != 'customer_id'}
        }
        self.prediction_log.append(log_entry)
        
        # Keep only recent predictions (memory management)
        if len(self.prediction_log) > 10000:
            self.prediction_log = self.prediction_log[-5000:]
    
    def get_performance_stats(self) -> Dict:
        """Analyze performance across model versions"""
        if not self.prediction_log:
            return {"message": "No predictions logged yet"}
        
        stats = {}
        for version in self.registry.get_all_versions():
            version_predictions = [p for p in self.prediction_log if p['model_version'] == version]
            
            if version_predictions:
                stats[version] = {
                    'total_predictions': len(version_predictions),
                    'avg_probability': sum(p['probability'] for p in version_predictions) / len(version_predictions),
                    'positive_predictions': sum(1 for p in version_predictions if p['prediction']),
                    'last_used': max(p['timestamp'] for p in version_predictions)
                }
        
        return stats
    
    def update_traffic_split(self, new_split: Dict[str, int]):
        """Update traffic routing configuration"""
        self.routing_config["traffic_split"] = new_split
        
        # Save to file
        with open(self.config_path, 'w') as f:
            json.dump(self.routing_config, f, indent=2)
        
        print(f"Updated traffic split: {new_split}")