"""
Training Pipeline: Complete model training workflow
"""

import duckdb
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.models.train import prepare_features, train_xgboost_model, train_catboost_model, train_lightgbm_model
from src.models.evaluate import evaluate_model, find_optimal_threshold
from src.models.ensemble import create_ensemble
from src.preprocessing.encoders import encode_categorical_features
from src.preprocessing.samplers import balance_dataset
from src.utils.paths import get_models_path, get_data_path


def run_training_pipeline(
    gold_db_path: str = None,
    models_dir: str = None,
    artifacts_dir: str = None
):
    """
    Complete training pipeline
    
    Parameters:
    -----------
    gold_db_path : str, optional
        Path to gold layer database (default: data/gold/modeling_dataset.duckdb)
    models_dir : str, optional
        Directory to save models (default: models/checkpoints)
    artifacts_dir : str, optional
        Directory to save artifacts (default: models/artifacts)
    """
    print("="*80)
    print("TRAINING PIPELINE")
    print("="*80)
    
    # Set default paths
    if gold_db_path is None:
        gold_db_path = str(get_data_path('gold') / 'modeling_dataset.duckdb')
    if models_dir is None:
        models_dir = str(get_models_path('checkpoints'))
    if artifacts_dir is None:
        artifacts_dir = str(get_models_path('artifacts'))
    
    # Create directories
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data from gold layer
    print("\nLoading data from gold layer...")
    con = duckdb.connect(gold_db_path)
    df = con.execute("SELECT * FROM modeling_dataset").fetch_df()
    con.close()
    
    # Prepare features
    X, y, categorical_cols, numeric_cols = prepare_features(df)
    
    # Split data: 70% train, 15% validation, 15% test
    print("\nSplitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Encode categorical features
    X_train_enc, X_val_enc, X_test_enc, label_encoders = encode_categorical_features(
        X_train, X_val, X_test, categorical_cols
    )
    
    # Balance training data
    X_train_balanced, y_train_balanced = balance_dataset(
        X_train_enc, y_train,
        categorical_cols=categorical_cols,
        strategy='adasyn',
        sampling_ratio=0.5,
        random_state=42
    )
    
    # Get categorical indices for CatBoost
    categorical_indices = [X_train_enc.columns.get_loc(col) for col in categorical_cols if col in X_train_enc.columns]
    
    # Train models
    models = {}
    xgb_model = train_xgboost_model(X_train_balanced, y_train_balanced, X_val_enc, y_val)
    models['xgb'] = xgb_model
    
    cat_model = train_catboost_model(X_train_balanced, y_train_balanced, X_val_enc, y_val, categorical_indices)
    models['catboost'] = cat_model
    
    lgb_model = train_lightgbm_model(X_train_balanced, y_train_balanced, X_val_enc, y_val)
    models['lightgbm'] = lgb_model
    
    # Evaluate on validation set and find optimal threshold
    print("\nEvaluating models on validation set...")
    val_results = {}
    for name, model in models.items():
        y_val_proba = model.predict_proba(X_val_enc)[:, 1]
        optimal_thresh = find_optimal_threshold(y_val, y_val_proba, metric='balanced_precision', target_precision=0.70)
        val_results[name] = evaluate_model(model, X_val_enc, y_val, optimal_thresh, name)
    
    # Create ensemble
    ensemble_proba = create_ensemble(models, X_val_enc)
    ensemble_threshold = find_optimal_threshold(y_val, ensemble_proba, metric='balanced_precision', target_precision=0.70)
    
    # Select best model
    best_model_name = max(val_results.items(), key=lambda x: x[1]['f1'])[0]
    print(f"\nBest model on validation: {best_model_name}")
    
    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    if best_model_name == 'ensemble':
        test_ensemble_proba = create_ensemble(models, X_test_enc)
        test_ensemble_pred = (test_ensemble_proba >= ensemble_threshold).astype(int)
        optimal_threshold = ensemble_threshold
    else:
        best_model = models[best_model_name]
        y_test_proba = best_model.predict_proba(X_test_enc)[:, 1]
        optimal_threshold = find_optimal_threshold(y_test, y_test_proba, metric='balanced_precision', target_precision=0.70)
        test_results = evaluate_model(best_model, X_test_enc, y_test, optimal_threshold, f"Best Model ({best_model_name})")
    
    # Save models and artifacts
    print("\nSaving models and artifacts...")
    
    # Save models
    import xgboost as xgb
    from catboost import CatBoostClassifier
    import lightgbm as lgb
    
    for name, model in models.items():
        if isinstance(model, xgb.XGBClassifier):
            model.save_model(f'{models_dir}/xgb_model.json')
        elif isinstance(model, CatBoostClassifier):
            model.save_model(f'{models_dir}/catboost_model.cbm')
        elif isinstance(model, lgb.LGBMClassifier):
            model.booster_.save_model(f'{models_dir}/lightgbm_model.txt')
    
    # Save artifacts
    with open(f'{artifacts_dir}/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    with open(f'{artifacts_dir}/optimal_threshold.pkl', 'wb') as f:
        pickle.dump(optimal_threshold, f)
    
    with open(f'{artifacts_dir}/feature_names.pkl', 'wb') as f:
        pickle.dump(X_train_enc.columns.tolist(), f)
    
    print("✓ Training pipeline complete!")
    return models, test_results, optimal_threshold

