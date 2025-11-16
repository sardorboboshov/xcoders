"""
Ensemble prediction functions
"""

import numpy as np
import pandas as pd
from typing import Dict


def create_ensemble(models: Dict, X_test: pd.DataFrame) -> np.ndarray:
    """
    Create ensemble predictions by averaging model probabilities
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : DataFrame
        Test features
    
    Returns:
    --------
    array: Ensemble probability predictions
    """
    print("\n" + "="*80)
    print("CREATING ENSEMBLE PREDICTIONS")
    print("="*80)
    
    predictions = []
    for name, model in models.items():
        try:
            # All models (XGBoost, CatBoost, LightGBM LGBMClassifier) support predict_proba
            pred_proba = model.predict_proba(X_test)[:, 1]
            predictions.append(pred_proba)
            print(f"{name} predictions collected")
        except Exception as e:
            print(f"Warning: Could not get predictions from {name}: {e}")
    
    # Average predictions
    ensemble_proba = np.mean(predictions, axis=0)
    
    return ensemble_proba

