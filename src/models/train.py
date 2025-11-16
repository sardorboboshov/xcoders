"""
Model training functions
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
from typing import Tuple, List, Optional


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Prepare features and target
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe with target column
    
    Returns:
    --------
    X, y, categorical_cols, numeric_cols : tuple
        Features, target, and column type lists
    """
    print("\n" + "="*80)
    print("PREPARING FEATURES")
    print("="*80)
    
    # Separate target
    if 'default' not in df.columns:
        raise ValueError("Target column 'default' not found")
    
    y = df['default'].astype(int)
    
    # Drop identifier and target columns
    drop_cols = ['default', 'customer_ref', 'application_id']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Identify column types
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Default rate: {y.mean():.4f}")
    print(f"\nNumeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    
    return X, y, categorical_cols, numeric_cols


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> xgb.XGBClassifier:
    """Train XGBoost model with optimized parameters for imbalanced data"""
    print("\n" + "="*80)
    print("TRAINING XGBOOST MODEL")
    print("="*80)
    
    # Calculate scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    print(f"Class distribution - Negative: {neg_count}, Positive: {pos_count}")
    print(f"Base scale pos weight: {scale_pos_weight:.2f}")
    
    # Increase weight more aggressively
    scale_pos_weight_boosted = scale_pos_weight * 2.0
    print(f"Boosted scale pos weight: {scale_pos_weight_boosted:.2f}")
    
    # Optimized XGBoost parameters for imbalanced data
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'aucpr'],
        'max_depth': 6,
        'learning_rate': 0.03,
        'n_estimators': 2000,
        'min_child_weight': 2,
        'gamma': 0.2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'scale_pos_weight': scale_pos_weight_boosted,
        'reg_alpha': 0.3,
        'reg_lambda': 1.0,
        'max_delta_step': 3,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
        'early_stopping_rounds': 150,
        'verbosity': 0
    }
    
    model = xgb.XGBClassifier(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )
    
    print(f"Best iteration: {model.best_iteration}")
    print(f"Best score: {model.best_score:.4f}")
    
    return model


def train_catboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    categorical_indices: Optional[List[int]] = None
) -> CatBoostClassifier:
    """Train CatBoost model with improved imbalanced data handling"""
    print("\n" + "="*80)
    print("TRAINING CATBOOST MODEL")
    print("="*80)
    
    # Calculate class weights
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    class_weights = [1.0, neg_count / pos_count * 2.0]
    
    print(f"Class distribution - Negative: {neg_count}, Positive: {pos_count}")
    print(f"Class weights: {class_weights}")
    
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=2,
        loss_function='Logloss',
        eval_metric='AUC',
        class_weights=class_weights,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=150,
        cat_features=categorical_indices if categorical_indices else None,
        min_data_in_leaf=2,
        max_leaves=31
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=100
    )
    
    print(f"Best iteration: {model.best_iteration_}")
    print(f"Best score: {model.best_score_['validation']['AUC']:.4f}")
    
    return model


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> lgb.LGBMClassifier:
    """Train LightGBM model with improved imbalanced data handling"""
    print("\n" + "="*80)
    print("TRAINING LIGHTGBM MODEL")
    print("="*80)
    
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    print(f"Class distribution - Negative: {neg_count}, Positive: {pos_count}")
    print(f"Base scale pos weight: {scale_pos_weight:.2f}")
    
    scale_pos_weight_boosted = scale_pos_weight * 2.0
    print(f"Boosted scale pos weight: {scale_pos_weight_boosted:.2f}")
    
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.03,
        n_estimators=2000,
        max_depth=6,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight_boosted,
        reg_alpha=0.3,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=150), lgb.log_evaluation(period=100)]
    )
    
    print(f"Best iteration: {model.best_iteration_}")
    print(f"Best score: {model.best_score_['valid_1']['auc']:.4f}")
    
    return model

