import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_telecom_data(n_customers=10000):
    """Generate realistic telecom customer data for churn prediction"""
    
    np.random.seed(42)  # For reproducibility
    
    # Customer demographics
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]
    ages = np.random.normal(42, 15, n_customers).astype(int)
    ages = np.clip(ages, 18, 80)
    
    # Account information
    months_active = np.random.exponential(24, n_customers).astype(int)
    months_active = np.clip(months_active, 1, 120)
    
    monthly_charges = np.random.normal(65, 20, n_customers)
    monthly_charges = np.clip(monthly_charges, 20, 150)
    
    total_charges = monthly_charges * months_active + np.random.normal(0, 100, n_customers)
    total_charges = np.maximum(total_charges, monthly_charges)
    
    # Service features
    internet_service = np.random.choice(['DSL', 'Fiber', 'No'], n_customers, p=[0.4, 0.4, 0.2])
    contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                   n_customers, p=[0.5, 0.3, 0.2])
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
                                    n_customers, p=[0.3, 0.2, 0.25, 0.25])
    
    # Support tickets (higher = more likely to churn)
    support_tickets = np.random.poisson(2, n_customers)
    
    # Create churn based on realistic factors
    churn_probability = (
        0.1 +  # Base churn rate
        0.3 * (contract_type == 'Month-to-month') +  # Month-to-month customers churn more
        0.2 * (monthly_charges > 80) +  # High-paying customers churn more
        0.1 * (support_tickets > 3) +  # Many support tickets = unhappy customer
        0.15 * (months_active < 6)  # New customers more likely to churn
    )
    
    churn = np.random.binomial(1, churn_probability, n_customers)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'months_active': months_active,
        'monthly_charges': monthly_charges.round(2),
        'total_charges': total_charges.round(2),
        'internet_service': internet_service,
        'contract_type': contract_type,
        'payment_method': payment_method,
        'support_tickets': support_tickets,
        'churn': churn
    })
    
    return df

if __name__ == "__main__":
    # Generate data
    data = generate_telecom_data(10000)
    
    # Create data directory if it doesn't exist
    os.makedirs('../data', exist_ok=True)
    
    # Save to CSV
    data.to_csv('../data/telecom_customers.csv', index=False)
    
    print(f"Generated {len(data)} customer records")
    print(f"Churn rate: {data['churn'].mean():.2%}")
    print("\nSample data:")
    print(data.head())