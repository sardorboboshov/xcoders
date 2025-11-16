"""
Categorical encoding functions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, List


def encode_categorical_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical features using label encoding
    
    Parameters:
    -----------
    X_train, X_val, X_test : DataFrame
        Training, validation, and test feature sets
    categorical_cols : list
        List of categorical column names
    
    Returns:
    --------
    X_train_encoded, X_val_encoded, X_test_encoded : DataFrame
        Encoded feature sets
    label_encoders : dict
        Dictionary of label encoders for each categorical column
    """
    label_encoders = {}
    X_train_encoded = X_train.copy()
    X_val_encoded = X_val.copy()
    X_test_encoded = X_test.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on training data
        X_train_encoded[col] = le.fit_transform(X_train[col].astype(str))
        label_encoders[col] = le
        
        # Transform validation and test
        # Handle unseen categories
        val_unique = set(X_val[col].astype(str).unique())
        train_unique = set(le.classes_)
        unseen = val_unique - train_unique
        
        if unseen:
            # Map unseen to a default value
            X_val_encoded[col] = X_val[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in train_unique else 0
            )
        else:
            X_val_encoded[col] = le.transform(X_val[col].astype(str))
        
        test_unique = set(X_test[col].astype(str).unique())
        unseen = test_unique - train_unique
        
        if unseen:
            X_test_encoded[col] = X_test[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in train_unique else 0
            )
        else:
            X_test_encoded[col] = le.transform(X_test[col].astype(str))
    
    return X_train_encoded, X_val_encoded, X_test_encoded, label_encoders


def encode_for_inference(
    X: pd.DataFrame,
    label_encoders: Dict[str, LabelEncoder],
    categorical_cols: List[str]
) -> pd.DataFrame:
    """
    Encode categorical features for inference using pre-fitted encoders
    
    Parameters:
    -----------
    X : DataFrame
        Feature set to encode
    label_encoders : dict
        Pre-fitted label encoders
    categorical_cols : list
        List of categorical column names
    
    Returns:
    --------
    X_encoded : DataFrame
        Encoded feature set
    """
    X_encoded = X.copy()
    
    for col in categorical_cols:
        if col in label_encoders and col in X_encoded.columns:
            le = label_encoders[col]
            known_classes = set(le.classes_)
            X_encoded[col] = X_encoded[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in known_classes else 0
            )
    
    return X_encoded

