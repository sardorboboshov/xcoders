"""
Prediction functions for inference
"""

import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Dict, Optional
from pathlib import Path


def load_models(models_dir: str = 'models/checkpoints') -> Dict:
    """Load all trained models"""
    models = {}
    
    try:
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(f'{models_dir}/xgb_model.json')
        models['xgb'] = xgb_model
    except Exception as e:
        print(f"Warning: Could not load XGBoost: {e}")
    
    try:
        cat_model = CatBoostClassifier()
        cat_model.load_model(f'{models_dir}/catboost_model.cbm')
        models['catboost'] = cat_model
    except Exception as e:
        print(f"Warning: Could not load CatBoost: {e}")
    
    try:
        lgb_booster = lgb.Booster(model_file=f'{models_dir}/lightgbm_model.txt')
        models['lightgbm'] = lgb_booster
    except Exception as e:
        print(f"Warning: Could not load LightGBM: {e}")
    
    return models


def predict_batch(
    X: pd.DataFrame,
    models: Optional[Dict] = None,
    models_dir: str = 'models/checkpoints',
    threshold: float = 0.5
) -> Dict:
    """
    Make batch predictions on new data
    
    Parameters:
    -----------
    X : DataFrame
        Preprocessed features
    models : dict, optional
        Pre-loaded models (if None, will load from models_dir)
    models_dir : str
        Directory with models
    threshold : float
        Classification threshold
    
    Returns:
    --------
    dict: Predictions and probabilities
    """
    if models is None:
        models = load_models(models_dir)
    
    if not models:
        raise ValueError("No models loaded!")
    
    # Collect predictions from all models
    all_probas = []
    
    for name, model in models.items():
        try:
            if name == 'lightgbm':
                # LightGBM Booster
                proba = model.predict(X.values)
            else:
                # XGBoost and CatBoost
                proba = model.predict_proba(X)[:, 1]
            
            all_probas.append(proba)
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    # Ensemble: average probabilities
    ensemble_proba = np.mean(all_probas, axis=0)
    ensemble_pred = (ensemble_proba >= threshold).astype(int)
    
    return {
        'probabilities': ensemble_proba,
        'predictions': ensemble_pred,
        'individual_models': {name: proba for name, proba in zip(models.keys(), all_probas)}
    }

