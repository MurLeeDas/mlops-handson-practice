from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Response
from pydantic import BaseModel
from typing import Optional, Dict, List
import time
import os
import sys
import pandas as pd
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import monitoring components
from metrics.metrics_collector import metrics_collector
from drift_detection.drift_detector import data_drift_detector, concept_drift_detector

# Import previous components (assume they exist from lessons 2-3)
try:
    from router.model_router import ModelRouter
    from reliability.circuit_breaker import circuit_manager
    router = ModelRouter(models_dir="./models", config_path="./configs/routing_config.json")
    logger.info("✅ Router and circuit breakers loaded successfully")
except Exception as e:
    logger.warning(f"⚠️ Could not load router/circuit breakers: {e}")
    router = None

# Initialize FastAPI with enhanced monitoring
app = FastAPI(
    title="Production MLOps Platform with Advanced Monitoring",
    description="Enterprise-grade ML platform with Prometheus metrics and drift detection",
    version="4.0.0"
)

# Request/Response models
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

class MonitoredPredictionResponse(BaseModel):
    customer_id: Optional[str]
    prediction: bool
    probability: float
    model_version: str
    algorithm: str
    routing_strategy: str
    timestamp: str
    confidence_level: str
    business_recommendation: str
    monitoring_info: Dict
    performance_metrics: Dict

# Middleware for request monitoring
@app.middleware("http")
async def monitoring_middleware(request, call_next):
    start_time = time.time()
    
    # Increment concurrent requests
    metrics_collector.api_concurrent_requests.inc()
    
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        logger.error(f"Request failed: {e}")
        status_code = 500
        response = Response("Internal Server Error", status_code=500)
    finally:
        # Record request metrics
        duration = time.time() - start_time
        metrics_collector.record_api_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=status_code,
            duration=duration
        )
        
        # Decrement concurrent requests
        metrics_collector.api_concurrent_requests.dec()
    
    return response

@app.get("/")
def root():
    return {
        "message": "Production MLOps Platform with Advanced Monitoring",
        "version": "4.0.0",
        "monitoring_features": [
            "Prometheus metrics collection",
            "Data drift detection",
            "Model performance monitoring",
            "Real-time alerting",
            "Business KPI tracking"
        ],
        "endpoints": {
            "predictions": "/predict/monitored",
            "metrics": "/metrics",
            "drift_analysis": "/monitoring/drift",
            "performance": "/monitoring/performance",
            "business_kpis": "/monitoring/business"
        }
    }

@app.get("/health")
async def comprehensive_health_check():
    """Comprehensive health check with monitoring status"""
    
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "api": "healthy",
                "metrics_collector": "healthy",
                "drift_detector": "healthy",
                "models": "unknown",
                "monitoring": "healthy"
            },
            "metrics_summary": metrics_collector.get_metrics_summary(),
            "drift_summary": data_drift_detector.get_drift_summary(time_window_hours=1)
        }
        
        # Check model availability
        if router and router.registry.get_all_versions():
            health_status["components"]["models"] = "healthy"
            health_status["available_models"] = router.registry.get_all_versions()
        else:
            health_status["components"]["models"] = "degraded"
            health_status["available_models"] = []
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/predict/monitored", response_model=MonitoredPredictionResponse)
async def monitored_prediction(
    customer: CustomerData,
    background_tasks: BackgroundTasks,
    model_version: Optional[str] = Query(None, description="Force specific model version"),
    enable_drift_detection: bool = Query(True, description="Enable drift detection")
):
    """Make prediction with comprehensive monitoring"""
    
    prediction_start_time = time.time()
    
    try:
        customer_dict = customer.dict()
        
        # Default fallback prediction
        prediction_result = {
            'prediction': False,
            'probability': 0.3,
            'model_version': 'fallback_v1.0',
            'algorithm': 'RuleBased',
            'routing_strategy': 'fallback'
        }
        
        # Try to get real model prediction
        if router and router.registry.get_all_versions():
            try:
                # Select model version
                if not model_version:
                    model_version = router.registry.get_best_model() or router.registry.get_all_versions()[0]
                
                # Make prediction
                prediction_result = router.registry.predict(model_version, customer_dict)
                prediction_result['routing_strategy'] = 'model_serving'
                
            except Exception as e:
                logger.warning(f"Model prediction failed, using fallback: {e}")
                prediction_result['routing_strategy'] = 'fallback_after_error'
        
        # Calculate prediction latency
        prediction_latency = time.time() - prediction_start_time
        
        # Record prediction metrics
        metrics_collector.record_prediction(
            model_version=prediction_result['model_version'],
            algorithm=prediction_result['algorithm'],
            prediction_type='churn_prediction',
            latency=prediction_latency,
            confidence=prediction_result['probability']
        )
        
        # Business logic
        probability = prediction_result['probability']
        if probability >= 0.8:
            confidence_level = "Very High"
            recommendation = "URGENT: Immediate retention intervention required"
        elif probability >= 0.6:
            confidence_level = "High"
            recommendation = "High priority: Schedule retention call within 48 hours"
        elif probability >= 0.4:
            confidence_level = "Medium"
            recommendation = "Medium priority: Proactive engagement within 1 week"
        else:
            confidence_level = "Low"
            recommendation = "Low risk: Continue regular customer engagement"
        
        # Record business metrics
        if probability >= 0.7:
            metrics_collector.record_business_metric(
                'churn_high_risk',
                1,
                {'time_window': '1h', 'customer_segment': 'all'}
            )
        
        # Background tasks for monitoring
        if enable_drift_detection:
            background_tasks.add_task(
                perform_drift_detection,
                customer_dict,
                prediction_result['model_version']
            )
        
        background_tasks.add_task(
            update_performance_metrics,
            prediction_result['model_version'],
            prediction_latency,
            probability
        )
        
        # Monitoring info
        monitoring_info = {
            "metrics_recorded": True,
            "drift_detection_enabled": enable_drift_detection,
            "prediction_latency_ms": round(prediction_latency * 1000, 2),
            "monitoring_timestamp": datetime.now().isoformat()
        }
        
        # Performance metrics
        performance_metrics = {
            "prediction_confidence": probability,
            "model_load_time": "cached",
            "feature_processing_time_ms": round(prediction_latency * 1000 * 0.3, 2),
            "model_inference_time_ms": round(prediction_latency * 1000 * 0.7, 2)
        }
        
        return MonitoredPredictionResponse(
            customer_id=customer.customer_id,
            prediction=prediction_result['prediction'],
            probability=round(prediction_result['probability'], 4),
            model_version=prediction_result['model_version'],
            algorithm=prediction_result['algorithm'],
            routing_strategy=prediction_result['routing_strategy'],
            timestamp=datetime.now().isoformat(),
            confidence_level=confidence_level,
            business_recommendation=recommendation,
            monitoring_info=monitoring_info,
            performance_metrics=performance_metrics
        )
        
    except Exception as e:
        logger.error(f"Monitored prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction service error: {str(e)}")

async def perform_drift_detection(customer_data: Dict, model_version: str):
    """Background task for drift detection"""
    try:
        # Convert to DataFrame for drift detection
        current_data = pd.DataFrame([customer_data])
        
        # Check if we have reference data
        if hasattr(data_drift_detector, 'reference_data'):
            drift_results = data_drift_detector.detect_drift(current_data)
            
            # Log significant drift
            drifted_features = [r for r in drift_results if r.drift_detected]
            if drifted_features:
                logger.warning(f"Drift detected in {len(drifted_features)} features for model {model_version}")
        
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")

async def update_performance_metrics(model_version: str, latency: float, confidence: float):
    """Background task for performance metric updates"""
    try:
        # This would typically calculate more sophisticated metrics
        # based on actual vs predicted outcomes over time
        
        # For demo purposes, simulate performance metrics
        simulated_accuracy = 0.85 + (confidence - 0.5) * 0.1  # Simulate accuracy based on confidence
        simulated_precision = 0.82
        simulated_recall = 0.88
        simulated_f1 = 0.85
        
        metrics_collector.record_model_performance(
            model_version=model_version,
            accuracy=simulated_accuracy,
            precision=simulated_precision,
            recall=simulated_recall,
            f1_score=simulated_f1
        )
        
    except Exception as e:
        logger.error(f"Performance metrics update failed: {e}")

@app.get("/metrics")
async def get_prometheus_metrics():
    """Endpoint for Prometheus metrics scraping"""
    metrics = metrics_collector.get_metrics()
    return Response(content=metrics, media_type="text/plain")

@app.get("/monitoring/drift")
async def get_drift_analysis(time_window_hours: int = Query(24, description="Time window for drift analysis")):
    """Get comprehensive drift analysis"""
    
    try:
        drift_summary = data_drift_detector.get_drift_summary(time_window_hours)
        
        # Add concept drift information
        concept_drift_info = {
            "concept_drift_monitoring": "active",
            "baseline_performance_set": concept_drift_detector.baseline_performance is not None,
            "performance_history_length": len(concept_drift_detector.performance_history)
        }
        
        return {
            "data_drift_summary": drift_summary,
            "concept_drift_summary": concept_drift_info,
            "monitoring_status": "active",
            "time_window_hours": time_window_hours
        }
        
    except Exception as e:
        logger.error(f"Drift analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Drift analysis error: {str(e)}")

@app.get("/monitoring/performance")
async def get_performance_metrics():
    """Get model performance metrics"""
    
    try:
        metrics_summary = metrics_collector.get_metrics_summary()
        
        # Add model-specific performance data
        performance_data = {
            "metrics_summary": metrics_summary,
            "model_performance": {
                "total_predictions_24h": metrics_summary.get('recent_predictions_1h', 0) * 24,
                "avg_latency_ms": round(metrics_summary.get('avg_prediction_latency', 0) * 1000, 2),
                "avg_confidence": round(metrics_summary.get('avg_confidence', 0), 3),
                "active_models": metrics_summary.get('model_versions_active', 0)
            },
            "system_health": {
                "api_status": "healthy",
                "monitoring_status": "active",
                "metrics_collection": "enabled"
            }
        }
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Performance metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Performance metrics error: {str(e)}")

@app.get("/monitoring/business")
async def get_business_kpis():
    """Get business KPI metrics"""
    
    try:
        # This would typically query actual business metrics
        # For demo purposes, simulate based on current metrics
        
        business_kpis = {
            "churn_prevention": {
                "high_risk_customers_identified_24h": 45,
                "retention_actions_triggered": 23,
                "estimated_revenue_saved_24h": 15600,
                "campaign_effectiveness_rate": 0.68
            },
            "model_performance_impact": {
                "predictions_accuracy_trend": "stable",
                "false_positive_rate": 0.12,
                "customer_satisfaction_impact": "+2.3%",
                "operational_efficiency": "+18%"
            },
            "cost_metrics": {
                "prediction_cost_per_request": 0.0023,
                "infrastructure_utilization": "74%",
                "roi_monthly": "340%"
            }
        }
        
        return business_kpis
        
    except Exception as e:
        logger.error(f"Business KPI retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Business KPI error: {str(e)}")

@app.post("/monitoring/set-reference-data")
async def set_reference_data():
    """Set reference data for drift detection"""
    
    try:
        # This would typically load from your training data
        # For demo, create sample reference data
        
        np.random.seed(42)
        reference_data = pd.DataFrame({
            'age': np.random.normal(42, 15, 1000),
            'months_active': np.random.exponential(24, 1000),
            'monthly_charges': np.random.normal(65, 20, 1000),
            'total_charges': np.random.normal(1500, 800, 1000),
            'support_tickets': np.random.poisson(2, 1000)
        })
        
        data_drift_detector.set_reference_data(reference_data)
        
        # Set baseline performance for concept drift
        concept_drift_detector.set_baseline_performance(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85
        )
        
        return {
            "message": "Reference data set successfully",
            "reference_samples": len(reference_data),
            "features_monitored": list(reference_data.columns),
            "baseline_performance_set": True
        }
        
    except Exception as e:
        logger.error(f"Setting reference data failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reference data error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import numpy as np
    
    # Initialize reference data on startup
    logger.info("🚀 Starting MLOps platform with advanced monitoring...")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")