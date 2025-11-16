"""
Data balancing/sampling functions
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List

# Import imbalanced-learn for SMOTE
try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
    from imblearn.combine import SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Warning: imbalanced-learn not installed. Install with: pip install imbalanced-learn")


def balance_dataset(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_cols: Optional[List[str]] = None,
    strategy: str = 'borderline_smote',
    sampling_ratio: float = 0.5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balance training dataset using SMOTE or similar techniques
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target
    categorical_cols : list, optional
        List of categorical column names (needed to preserve integer types after SMOTE)
    strategy : str
        - 'borderline_smote': BorderlineSMOTE (focuses on borderline samples) - RECOMMENDED
        - 'smote': Standard SMOTE
        - 'adasyn': ADASYN (adaptive synthetic sampling)
        - 'smoteenn': SMOTE + Edited Nearest Neighbours
        - 'none': No balancing
    sampling_ratio : float
        Target ratio of minority to majority class (0.5 = 1:2, 1.0 = 1:1)
    random_state : int
        Random seed
    
    Returns:
    --------
    X_resampled, y_resampled : DataFrame, Series
        Balanced training data
    """
    if not IMBLEARN_AVAILABLE or strategy == 'none':
        print("Skipping data balancing (imbalanced-learn not available or strategy='none')")
        return X_train, y_train
    
    # Convert to numpy arrays if DataFrame
    if isinstance(X_train, pd.DataFrame):
        X_array = X_train.values
        feature_names = X_train.columns.tolist()
        original_dtypes = X_train.dtypes.to_dict()
        return_dataframe = True
    else:
        X_array = X_train
        feature_names = None
        original_dtypes = None
        return_dataframe = False
    
    y_array = y_train.values if isinstance(y_train, pd.Series) else y_train
    
    # Calculate current class distribution
    neg_count = (y_array == 0).sum()
    pos_count = (y_array == 1).sum()
    current_ratio = pos_count / neg_count
    
    print(f"\nBalancing dataset:")
    print(f"  Before: {neg_count:,} negative, {pos_count:,} positive (ratio: {current_ratio:.3f})")
    print(f"  Strategy: {strategy}, Target ratio: {sampling_ratio:.2f}")
    
    try:
        if strategy == 'borderline_smote':
            sampler = BorderlineSMOTE(
                sampling_strategy=sampling_ratio,
                random_state=random_state,
                k_neighbors=min(5, pos_count - 1) if pos_count > 1 else 1
            )
            X_resampled, y_resampled = sampler.fit_resample(X_array, y_array)
            
        elif strategy == 'smote':
            sampler = SMOTE(
                sampling_strategy=sampling_ratio,
                random_state=random_state,
                k_neighbors=min(5, pos_count - 1) if pos_count > 1 else 1
            )
            X_resampled, y_resampled = sampler.fit_resample(X_array, y_array)
            
        elif strategy == 'adasyn':
            sampler = ADASYN(
                sampling_strategy=sampling_ratio,
                random_state=random_state,
                n_neighbors=min(5, pos_count - 1) if pos_count > 1 else 1
            )
            X_resampled, y_resampled = sampler.fit_resample(X_array, y_array)
            
        elif strategy == 'smoteenn':
            sampler = SMOTEENN(
                sampling_strategy=sampling_ratio,
                random_state=random_state,
                smote=SMOTE(k_neighbors=min(5, pos_count - 1) if pos_count > 1 else 1)
            )
            X_resampled, y_resampled = sampler.fit_resample(X_array, y_array)
        else:
            print(f"Unknown strategy: {strategy}. Returning original data.")
            return X_train, y_train
        
        # Convert back to DataFrame if input was DataFrame
        if return_dataframe:
            X_resampled = pd.DataFrame(X_resampled, columns=feature_names, index=range(len(X_resampled)))
            y_resampled = pd.Series(y_resampled, name=y_train.name if hasattr(y_train, 'name') else 'target')
            
            # Fix categorical columns: SMOTE can create float values, convert back to integers
            if categorical_cols is not None:
                for col in categorical_cols:
                    if col in X_resampled.columns:
                        X_resampled[col] = X_resampled[col].round().fillna(0).astype(int)
                        X_resampled[col] = X_resampled[col].clip(lower=0)
            
            # Also restore original dtypes for other columns where possible
            for col in X_resampled.columns:
                if col in original_dtypes and col not in (categorical_cols or []):
                    if pd.api.types.is_integer_dtype(original_dtypes[col]):
                        X_resampled[col] = X_resampled[col].round().fillna(0).astype(int)
        
        new_neg = (y_resampled == 0).sum()
        new_pos = (y_resampled == 1).sum()
        new_ratio = new_pos / new_neg
        print(f"  After: {new_neg:,} negative, {new_pos:,} positive (ratio: {new_ratio:.3f})")
        
        return X_resampled, y_resampled
        
    except Exception as e:
        print(f"Error during balancing: {e}. Returning original data.")
        import traceback
        traceback.print_exc()
        return X_train, y_train

