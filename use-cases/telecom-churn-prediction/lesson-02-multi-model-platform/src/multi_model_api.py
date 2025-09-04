from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, List
import os
import sys

# Fix Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from router.model_router import ModelRouter

# Initialize FastAPI
app = FastAPI(
    title="Multi-Model Telecom Churn Prediction Platform",
    description="A/B testing platform for churn prediction models",
    version="2.0.0"
)

# Initialize Model Router
router = ModelRouter(models_dir="./models", config_path="./configs/routing_config.json")

# Request models
class CustomerData(BaseModel):
    customer_id: Optional[str] = None
    age: int
    months_active: int
    monthly_charges: float
    total_charges: float
    internet_service: str  # 'DSL', 'Fiber', 'No'
    contract_type: str     # 'Month-to-month', 'One year', 'Two year'
    payment_method: str    # 'Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'
    support_tickets: int

class RoutingConfig(BaseModel):
    traffic_split: Dict[str, int]

# Response models
class PredictionResponse(BaseModel):
    customer_id: Optional[str]
    prediction: bool
    probability: float
    model_version: str
    algorithm: str
    routing_strategy: str
    timestamp: str
    confidence_level: str
    business_recommendation: str

@app.get("/")
async def root():
    return {
        "message": "Multi-Model Telecom Churn Prediction Platform", 
        "version": "2.0.0",
        "available_models": router.registry.get_all_versions()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "loaded_models": len(router.registry.get_all_versions()),
        "model_versions": router.registry.get_all_versions()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(
    customer: CustomerData,
    strategy: str = Query("traffic_split", description="Routing strategy: traffic_split, best_model, round_robin"),
    model_version: Optional[str] = Query(None, description="Force specific model version")
):
    """Predict customer churn with intelligent model routing"""
    try:
        customer_dict = customer.dict()
        
        if model_version:
            # Force specific model version
            if model_version not in router.registry.get_all_versions():
                raise HTTPException(status_code=400, detail=f"Model version {model_version} not available")
            result = router.registry.predict(model_version, customer_dict)
            result['routing_strategy'] = 'forced'
        else:
            # Use intelligent routing
            result = router.route_prediction(customer_dict, strategy)
        
        # Determine confidence level and recommendation
        probability = result['probability']
        if probability >= 0.8:
            confidence = "Very High"
            recommendation = "URGENT: Immediate retention intervention required within 24 hours"
        elif probability >= 0.6:
            confidence = "High"  
            recommendation = "High priority: Schedule retention call within 48 hours"
        elif probability >= 0.4:
            confidence = "Medium"
            recommendation = "Medium priority: Proactive engagement within 1 week"
        else:
            confidence = "Low"
            recommendation = "Low risk: Continue regular customer engagement"
        
        return PredictionResponse(
            customer_id=customer.customer_id,
            prediction=result['prediction'],
            probability=round(result['probability'], 4),
            model_version=result['model_version'],
            algorithm=result['algorithm'],
            routing_strategy=result['routing_strategy'],
            timestamp=result.get('timestamp', ''),
            confidence_level=confidence,
            business_recommendation=recommendation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/models")
async def get_available_models():
    """Get information about all available models"""
    models_info = {}
    for version in router.registry.get_all_versions():
        metadata = router.registry.get_model_metadata(version)
        models_info[version] = metadata
    
    return {
        "available_models": models_info,
        "best_model": router.registry.get_best_model(),
        "current_routing": router.routing_config["traffic_split"]
    }

@app.get("/performance")
async def get_performance_stats():
    """Get performance statistics across all models"""
    return router.get_performance_stats()

@app.post("/routing/update")
async def update_routing_config(config: RoutingConfig):
    """Update traffic split configuration"""
    try:
        # Validate that percentages add up to 100
        total_percentage = sum(config.traffic_split.values())
        if total_percentage != 100:
            raise HTTPException(
                status_code=400, 
                detail=f"Traffic split percentages must sum to 100, got {total_percentage}"
            )
        
        # Validate that all specified models exist
        available_models = router.registry.get_all_versions()
        for model_version in config.traffic_split.keys():
            if model_version not in available_models:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model version {model_version} not available. Available: {available_models}"
                )
        
        router.update_traffic_split(config.traffic_split)
        
        return {
            "message": "Routing configuration updated successfully",
            "new_config": config.traffic_split
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration update error: {str(e)}")

@app.post("/compare")
async def compare_models(customer: CustomerData):
    """Compare predictions across all available models"""
    try:
        customer_dict = customer.dict()
        results = {}
        
        for version in router.registry.get_all_versions():
            try:
                result = router.registry.predict(version, customer_dict)
                results[version] = {
                    'prediction': result['prediction'],
                    'probability': round(result['probability'], 4),
                    'algorithm': result['algorithm']
                }
            except Exception as e:
                results[version] = {'error': str(e)}
        
        # Find consensus
        predictions = [r.get('prediction') for r in results.values() if 'prediction' in r]
        consensus = max(set(predictions), key=predictions.count) if predictions else None
        
        return {
            'customer_id': customer.customer_id,
            'model_results': results,
            'consensus_prediction': consensus,
            'agreement_level': predictions.count(consensus) / len(predictions) if predictions else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model comparison error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)