import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from datetime import datetime

class ChurnPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoders = {}
        self.feature_columns = []
        
    def preprocess_data(self, df):
        """Preprocess the data for training/prediction"""
        df_processed = df.copy()
        
        # Handle categorical variables
        categorical_columns = ['internet_service', 'contract_type', 'payment_method']
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        # Feature engineering
        df_processed['avg_monthly_spend'] = df_processed['total_charges'] / df_processed['months_active']
        df_processed['support_per_month'] = df_processed['support_tickets'] / df_processed['months_active']
        
        # Define feature columns
        feature_columns = ['age', 'months_active', 'monthly_charges', 'total_charges', 
                          'internet_service', 'contract_type', 'payment_method', 
                          'support_tickets', 'avg_monthly_spend', 'support_per_month']
        
        self.feature_columns = feature_columns
        return df_processed[feature_columns]
    
    def train(self, df):
        """Train the churn prediction model"""
        print("Preprocessing data...")
        X = self.preprocess_data(df)
        y = df['churn']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 5 Important Features:")
        print(feature_importance.head())
        
        return X_test, y_test, y_pred
    
    def predict(self, customer_data):
        """Predict churn for new customer data"""
        if isinstance(customer_data, dict):
            customer_data = pd.DataFrame([customer_data])
        
        X = self.preprocess_data(customer_data)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]  # Probability of churn
        
        return predictions, probabilities
    
    def save_model(self, filepath):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'training_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        print(f"Model loaded from {filepath}")
        print(f"Training date: {model_data['training_date']}")

if __name__ == "__main__":
    # Load data
    data = pd.read_csv('../data/telecom_customers.csv')
    
    # Initialize and train model
    predictor = ChurnPredictor()
    X_test, y_test, y_pred = predictor.train(data)
    
    # Save model
    predictor.save_model('./models/churn_model.pkl')
    
    # Test prediction on a single customer
    test_customer = {
        'age': 35,
        'months_active': 12,
        'monthly_charges': 85.0,
        'total_charges': 1020.0,
        'internet_service': 'Fiber',
        'contract_type': 'Month-to-month',
        'payment_method': 'Electronic check',
        'support_tickets': 5
    }
    
    prediction, probability = predictor.predict(test_customer)
    print(f"\nTest Customer Prediction:")
    print(f"Will churn: {'Yes' if prediction[0] else 'No'}")
    print(f"Churn probability: {probability[0]:.2%}")