import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score
import os

# Features to use
MODEL_FEATURES = [
    'Lead_Time', 'Lat', 'Lon',
    'GFS_T2m', 'GFS_TP', 'GFS_Z500',
    'T2m_Anomaly', 'Z500_Anomaly', 'T2m_Temporal_Delta',
    'Z500_Grad_Lat', 'Z500_Grad_Lon', 'Z500_Grad_Mag'
]

def objective(trial, X, y):
    # Hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'is_unbalance': True,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Time Series Cross Validation (to prevent data leakage in weather data)
    tscv = TimeSeriesSplit(n_splits=3)
    pr_auc_scores = []
    
    for train_index, valid_index in tscv.split(X):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        # We optimize for PR-AUC because busts are a rare event
        y_prob = model.predict_proba(X_valid)[:, 1]
        score = average_precision_score(y_valid, y_prob)
        pr_auc_scores.append(score)
        
    return np.mean(pr_auc_scores)

def tune_model():
    print("Loading preprocessed data for tuning...")
    try:
        df = pd.read_parquet("data/preprocessed_data.parquet")
    except FileNotFoundError:
        print("Data not found. Please ensure features.py has been run.")
        return
        
    # Sort by time to respect the TimeSeriesSplit
    df = df.sort_values(by=['Date', 'Lead_Time'])
    
    X = df[MODEL_FEATURES]
    y = df['Bust_T2m']
    
    print("Starting Optuna Hyperparameter Tuning...")
    # Maximize PR-AUC
    study = optuna.create_study(direction='maximize')
    
    # Run 20 trials for demonstration (in reality, you'd run 100+)
    study.optimize(lambda trial: objective(trial, X, y), n_trials=20)
    
    print("\n--- Tuning Complete ---")
    print(f"Best PR-AUC Score: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    # Save the best parameters for train.py to use later
    os.makedirs("models", exist_ok=True)
    best_params_df = pd.Series(study.best_params)
    best_params_df.to_json("models/best_lgbm_params.json")
    print("\nSaved best parameters to models/best_lgbm_params.json")

if __name__ == "__main__":
    tune_model()
