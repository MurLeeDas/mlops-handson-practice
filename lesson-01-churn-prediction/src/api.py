from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from model import ChurnPredictor
import os

# Initialize FastAPI
app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Predict customer churn for telecom company",
    version="1.0.0"
)

# Load model on startup
predictor = ChurnPredictor()
model_path = "./models/churn_model.pkl"

if os.path.exists(model_path):
    predictor.load_model(model_path)
else:
    raise Exception("Model not found! Please train the model first.")

# Request model
class CustomerData(BaseModel):
    age: int
    months_active: int
    monthly_charges: float
    total_charges: float
    internet_service: str  # 'DSL', 'Fiber', 'No'
    contract_type: str     # 'Month-to-month', 'One year', 'Two year'
    payment_method: str    # 'Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'
    support_tickets: int

# Response model
class ChurnPrediction(BaseModel):
    customer_id: str = None
    will_churn: bool
    churn_probability: float
    risk_level: str
    recommendation: str

@app.get("/")
async def root():
    return {"message": "Telecom Churn Prediction API", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=ChurnPrediction)
async def predict_churn(customer: CustomerData):
    try:
        # Convert to dictionary
        customer_dict = customer.dict()
        
        # Make prediction
        prediction, probability = predictor.predict(customer_dict)
        
        # Determine risk level
        if probability[0] >= 0.7:
            risk_level = "High"
            recommendation = "Immediate intervention required. Contact customer within 24 hours."
        elif probability[0] >= 0.4:
            risk_level = "Medium"
            recommendation = "Schedule retention call within 1 week. Offer incentives."
        else:
            risk_level = "Low"
            recommendation = "Continue regular engagement. Monitor monthly."
        
        return ChurnPrediction(
            will_churn=bool(prediction[0]),
            churn_probability=round(probability[0], 4),
            risk_level=risk_level,
            recommendation=recommendation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(customers: list[CustomerData]):
    """Predict churn for multiple customers"""
    try:
        results = []
        for customer in customers:
            customer_dict = customer.dict()
            prediction, probability = predictor.predict(customer_dict)
            
            results.append({
                "customer_data": customer_dict,
                "will_churn": bool(prediction[0]),
                "churn_probability": round(probability[0], 4)
            })
        
        return {"predictions": results, "total_customers": len(results)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)