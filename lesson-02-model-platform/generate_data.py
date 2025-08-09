import sys
sys.path.append('../lesson-01-churn-prediction/src')
from data_generator import generate_telecom_data
import os

# Create data directory
os.makedirs('data', exist_ok=True)

# Generate data
data = generate_telecom_data(10000)
data.to_csv('data/telecom_customers.csv', index=False)
print("Generated training data successfully!")
