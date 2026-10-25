"""
Logistics Data Analysis & Delay Prediction Pipeline
Scenario: E-Commerce Delivery Route Optimization & Delivery Delay Analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def generate_synthetic_data(num_records=1000):
    np.random.seed(42)
    data = {
        'shipment_id': [f"SHP_{i:04d}" for i in range(1, num_records + 1)],
        'distance_km': np.random.uniform(5.0, 120.0, num_records).round(2),
        'traffic_density_score': np.random.uniform(1.0, 10.0, num_records).round(1),
        'weather_condition': np.random.choice(['Clear', 'Rain', 'Storm'], num_records, p=[0.7, 0.2, 0.1]),
        'package_weight_kg': np.random.uniform(0.5, 25.0, num_records).round(2),
        'driver_experience_yrs': np.random.randint(1, 15, num_records),
    }
    df = pd.DataFrame(data)
    
    risk_score = (
        (df['distance_km'] / 120.0) * 0.4 +
        (df['traffic_density_score'] / 10.0) * 0.35 +
        (df['weather_condition'].map({'Clear': 0, 'Rain': 0.15, 'Storm': 0.35})) -
        (df['driver_experience_yrs'] / 15.0) * 0.1
    )
    df['is_delayed'] = (risk_score > 0.45).astype(int)
    return df

def clean_and_preprocess(df):
    processed_df = df.copy()
    processed_df = pd.get_dummies(processed_df, columns=['weather_condition'], drop_first=True)
    features = processed_df.drop(columns=['shipment_id', 'is_delayed'])
    target = processed_df['is_delayed']
    return features, target

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("=" * 50)
    print("LOGISTICS MODEL TRAINING SUMMARY")
    print("=" * 50)
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nKey Influencing Factors on Delivery Delays:")
    print(feature_importance.to_string(index=False))
    print("=" * 50)

if __name__ == '__main__':
    print("Generating logistics dataset...")
    df_raw = generate_synthetic_data(1000)
    df_raw.to_csv("logistics_data.csv", index=False)
    print("Data saved to 'logistics_data.csv'.")
    
    print("\nPreprocessing data...")
    X, y = clean_and_preprocess(df_raw)
    
    print("\nTraining machine learning model...")
    train_and_evaluate(X, y)
