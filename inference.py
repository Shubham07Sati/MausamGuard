import pandas as pd
import numpy as np
import joblib
import os
from features import extract_features

# The features our model expects, exactly as defined in train.py
MODEL_FEATURES = [
    'Lead_Time', 'Lat', 'Lon',
    'GFS_T2m', 'GFS_TP', 'GFS_Z500',
    'T2m_Anomaly', 'Z500_Anomaly', 'T2m_Temporal_Delta',
    'Z500_Grad_Lat', 'Z500_Grad_Lon', 'Z500_Grad_Mag'
]

def load_model(model_path="models/lgbm_bust_model.pkl"):
    """Loads the trained LightGBM model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    return joblib.load(model_path)

def run_inference(raw_forecast_df: pd.DataFrame, model_path="models/lgbm_bust_model.pkl") -> pd.DataFrame:
    """
    Takes a raw GFS forecast DataFrame (from Person B's data pipeline),
    runs feature engineering, and predicts the probability of a forecast bust.
    
    Returns a DataFrame with Lat, Lon, Lead_Time, and Bust_Probability.
    """
    print(f"Running inference on {len(raw_forecast_df)} forecast data points...")
    
    # 1. Feature Engineering
    # We use the exact same feature extraction logic from our training pipeline
    processed_df = extract_features(raw_forecast_df)
    
    # Ensure all required features are present
    missing_cols = [col for col in MODEL_FEATURES if col not in processed_df.columns]
    if missing_cols:
        raise ValueError(f"Missing features after preprocessing: {missing_cols}")
        
    X_inference = processed_df[MODEL_FEATURES]
    
    # 2. Load Model
    model = load_model(model_path)
    
    # 3. Predict Probabilities
    # predict_proba returns [Prob(Class 0), Prob(Class 1)]
    # We want the probability of a Bust (Class 1)
    probabilities = model.predict_proba(X_inference)[:, 1]
    
    # 4. Format Output for the Backend/Frontend
    results_df = processed_df[['Date', 'Lead_Time', 'Lat', 'Lon']].copy()
    results_df['Bust_Probability'] = probabilities
    
    # 5. Real SHAP Explainability
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_inference)
    
    # Extract the impact of the top 3 features for each row
    # In binary classification, shap_values is a list of arrays: [negative_class, positive_class]
    # We want the positive class (Bust) impact
    pos_shap = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    # For MVP: We'll just append the raw SHAP values for the most important features
    # so the frontend can read them dynamically instead of hardcoding.
    results_df['SHAP_Z500_Grad'] = pos_shap[:, MODEL_FEATURES.index('Z500_Grad_Mag')]
    results_df['SHAP_Temporal_Delta'] = pos_shap[:, MODEL_FEATURES.index('T2m_Temporal_Delta')]
    results_df['SHAP_Climatology'] = pos_shap[:, MODEL_FEATURES.index('T2m_Anomaly')]
    
    conditions = [
        (results_df['Bust_Probability'] < 0.3),
        (results_df['Bust_Probability'] >= 0.3) & (results_df['Bust_Probability'] < 0.7),
        (results_df['Bust_Probability'] >= 0.7)
    ]
    choices = ['Low Risk', 'Medium Risk', 'High Risk']
    results_df['Risk_Level'] = np.select(conditions, choices, default='Unknown')
    
    return results_df

if __name__ == "__main__":
    from config import Config
    print(f"[{Config.MODE} MODE] Simulating live API request...")
    
    mock_dates = pd.date_range(start="2023-08-01", periods=10).tolist() * 4
    mock_data = pd.DataFrame({
        "Date": mock_dates,
        "Lead_Time": list(range(1, 11)) * 4,
        "Lat": [28.0]*10 + [28.0]*10 + [20.0]*10 + [20.0]*10,
        "Lon": [77.0]*10 + [78.0]*10 + [88.0]*10 + [89.0]*10,
        "GFS_T2m": np.random.uniform(25, 35, 40),
        "GFS_TP": np.random.uniform(0, 50, 40),
        "GFS_Z500": np.random.uniform(5700, 5900, 40)
    })
    
    try:
        predictions = run_inference(mock_data)
        print("\n--- INFERENCE RESULTS WITH REAL SHAP ---")
        print(predictions.head(5))
        predictions.to_json(f"{Config.DATA_DIR}/sample_api_response.json", orient="records")
    except Exception as e:
        print(e)
