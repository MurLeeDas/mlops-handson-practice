from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, List
import os
import sys
import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fixed_reliable_api")

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import components with error handling
try:
    from router.model_router import ModelRouter
    logger.info("✅ Router imported successfully")
except Exception as e:
    logger.error(f"❌ Router import failed: {e}")
    ModelRouter = None

# Initialize FastAPI
app = FastAPI(
    title="Fixed Reliable MLOps Platform",
    description="Reliable ML model serving with proper error handling",
    version="3.1.0"
)

# Initialize router with error handling
router = None
if ModelRouter:
    try:
        router = ModelRouter(models_dir="./models", config_path="./configs/routing_config.json")
        logger.info("✅ Router initialized successfully")
    except Exception as e:
        logger.error(f"❌ Router initialization failed: {e}")

class CustomerData(BaseModel):
    customer_id: Optional[str] = None
    age: int
    months_active: int
    monthly_charges: float
    total_charges: float
    internet_service: str
    contract_type: str
    payment_method: str
    support_tickets: int

class SimplePredictionResponse(BaseModel):
    customer_id: Optional[str]
    prediction: bool
    probability: float
    model_version: str
    algorithm: str
    routing_strategy: str
    timestamp: str
    confidence_level: str
    business_recommendation: str
    status: str

@app.get("/")
def root():
    return {
        "message": "Fixed Reliable MLOps Platform",
        "status": "healthy",
        "router_available": router is not None,
        "models_loaded": len(router.registry.get_all_versions()) if router else 0
    }

@app.get("/health")
async def health_check():
    """Simple health check with error handling"""
    try:
        if not router:
            return {
                "status": "degraded",
                "message": "Router not available",
                "models": []
            }
        
        available_models = router.registry.get_all_versions()
        
        return {
            "status": "healthy" if available_models else "no_models",
            "models_loaded": len(available_models),
            "available_models": available_models,
            "timestamp": "2025-01-01T00:00:00"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "models": []
        }

@app.post("/predict/reliable", response_model=SimplePredictionResponse)
async def reliable_predict_fixed(customer: CustomerData):
    """Make prediction with robust error handling"""
    
    try:
        customer_dict = customer.dict()
        
        # Check if router is available
        if not router:
            logger.warning("Router not available, using fallback prediction")
            return create_fallback_prediction(customer)
        
        # Check if models are available
        available_models = router.registry.get_all_versions()
        if not available_models:
            logger.warning("No models available, using fallback prediction")
            return create_fallback_prediction(customer)
        
        # Try to make prediction
        try:
            # Use the best available model
            best_model = router.registry.get_best_model()
            if not best_model:
                best_model = available_models[0]  # Use first available
            
            result = router.registry.predict(best_model, customer_dict)
            
            # Ensure result is not None
            if not result or not isinstance(result, dict):
                logger.warning("Model prediction returned invalid result, using fallback")
                return create_fallback_prediction(customer)
            
            # Add missing fields with defaults
            result.setdefault('routing_strategy', 'direct')
            result.setdefault('timestamp', '2025-01-01T00:00:00')
            
            # Determine confidence and recommendation
            probability = result.get('probability', 0.5)
            if probability >= 0.8:
                confidence = "Very High"
                recommendation = "URGENT: Immediate retention intervention required"
            elif probability >= 0.6:
                confidence = "High"
                recommendation = "High priority: Schedule retention call within 48 hours"
            elif probability >= 0.4:
                confidence = "Medium"
                recommendation = "Medium priority: Proactive engagement within 1 week"
            else:
                confidence = "Low"
                recommendation = "Low risk: Continue regular customer engagement"
            
            return SimplePredictionResponse(
                customer_id=customer.customer_id,
                prediction=result.get('prediction', False),
                probability=round(result.get('probability', 0.5), 4),
                model_version=result.get('model_version', best_model),
                algorithm=result.get('algorithm', 'Unknown'),
                routing_strategy=result.get('routing_strategy', 'direct'),
                timestamp=result.get('timestamp', '2025-01-01T00:00:00'),
                confidence_level=confidence,
                business_recommendation=recommendation,
                status="success"
            )
            
        except Exception as model_error:
            logger.error(f"Model prediction failed: {model_error}")
            return create_fallback_prediction(customer, error=str(model_error))
        
    except Exception as e:
        logger.error(f"Prediction request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction service error: {str(e)}")

def create_fallback_prediction(customer: CustomerData, error: str = None) -> SimplePredictionResponse:
    """Create a safe fallback prediction when models fail"""
    
    # Simple rule-based fallback logic
    risk_score = 0.3  # Base risk
    
    # Add risk based on customer characteristics
    if customer.contract_type == "Month-to-month":
        risk_score += 0.2
    if customer.monthly_charges > 80:
        risk_score += 0.15
    if customer.support_tickets > 3:
        risk_score += 0.25
    
    # Cap at reasonable maximum
    risk_score = min(risk_score, 0.9)
    prediction = risk_score > 0.5
    
    # Determine confidence and recommendation
    if risk_score >= 0.8:
        confidence = "High"
        recommendation = "High priority: Schedule retention call within 48 hours"
    elif risk_score >= 0.4:
        confidence = "Medium"
        recommendation = "Medium priority: Proactive engagement within 1 week"
    else:
        confidence = "Low"
        recommendation = "Low risk: Continue regular customer engagement"
    
    status = "fallback_success" if not error else f"fallback_after_error: {error}"
    
    return SimplePredictionResponse(
        customer_id=customer.customer_id,
        prediction=prediction,
        probability=round(risk_score, 4),
        model_version="fallback_v1.0",
        algorithm="RuleBased",
        routing_strategy="emergency_fallback",
        timestamp="2025-01-01T00:00:00",
        confidence_level=confidence,
        business_recommendation=recommendation,
        status=status
    )

@app.get("/debug/models")
async def debug_models():
    """Debug endpoint to check model status"""
    try:
        if not router:
            return {"error": "Router not initialized"}
        
        return {
            "router_available": True,
            "models_directory_exists": os.path.exists("./models"),
            "available_models": router.registry.get_all_versions(),
            "model_metadata": {v: router.registry.get_model_metadata(v) for v in router.registry.get_all_versions()},
            "best_model": router.registry.get_best_model()
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")