import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score, brier_score_loss
import shap
import matplotlib.pyplot as plt
import os
import joblib

def train_model():
    print("Loading preprocessed data...")
    try:
        df = pd.read_parquet("data/preprocessed_data.parquet")
    except FileNotFoundError:
        print("Data not found. Run features.py first.")
        return

    # Define features and target
    target = 'Bust_T2m'
    
    # Select feature columns (excluding metadata and targets/observations)
    features = [
        'Lead_Time', 'Lat', 'Lon',
        'GFS_T2m', 'GFS_TP', 'GFS_Z500',
        'T2m_Anomaly', 'Z500_Anomaly', 'T2m_Temporal_Delta',
        'Z500_Grad_Lat', 'Z500_Grad_Lon', 'Z500_Grad_Mag'
    ]
    
    X = df[features]
    y = df[target]
    
    # Temporal Train/Test split (e.g., train on first 20 days, test on last 10)
    # For dummy data, we will just use a random split or simple temporal split
    dates = df['Date'].unique()
    dates = np.sort(dates)
    split_date = dates[int(len(dates) * 0.8)] # 80/20 split
    
    train_mask = df['Date'] < split_date
    test_mask = df['Date'] >= split_date
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Class distribution in train: \n{y_train.value_counts(normalize=True)}")
    
    # Train LightGBM Model
    # is_unbalance=True helps with the fact that busts (1) are rare (~10%)
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        is_unbalance=True,
        random_state=42
    )
    
    print("\nTraining LightGBM model...")
    model.fit(X_train, y_train)
    
    # Evaluation
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    pr_auc = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Brier Score: {brier:.4f}")
    
    # Save Model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/lgbm_bust_model.pkl")
    print("\nModel saved to models/lgbm_bust_model.pkl")
    
    # SHAP Explainability (Day 5 Task preview)
    print("\nGenerating SHAP values (taking a sample of 100 for speed)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.head(100))
    
    # Save a summary plot
    os.makedirs("plots", exist_ok=True)
    shap.summary_plot(shap_values, X_test.head(100), show=False)
    plt.savefig("plots/shap_summary.png", bbox_inches='tight')
    print("SHAP summary plot saved to plots/shap_summary.png")

if __name__ == "__main__":
    train_model()
