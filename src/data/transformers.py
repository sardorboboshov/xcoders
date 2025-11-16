"""
Data transformation functions (joining, feature engineering, final cleaning)
"""

import pandas as pd
import numpy as np
from typing import Dict


def join_all_tables(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join all tables on customer identifiers"""
    print("\n" + "="*80)
    print("JOINING ALL TABLES")
    print("="*80)
    
    # Clean each table
    app_meta = dfs['application_metadata']
    credit = dfs['credit_history']
    demo = dfs['demographics']
    financial = dfs['financial_ratios']
    geo = dfs['geographic_data']
    loan = dfs['loan_details']
    
    # Start with application_metadata as base (has default target)
    df = app_meta.copy()
    print(f"Starting with application_metadata: {df.shape}")
    
    # Join credit_history
    if 'customer_ref' in df.columns and 'customer_number' in credit.columns:
        df = df.merge(credit, left_on='customer_ref', right_on='customer_number', how='left', suffixes=('', '_credit'))
        df = df.drop('customer_number', axis=1, errors='ignore')
        print(f"Joined credit_history: {df.shape}")
    else:
        print("Warning: Could not join credit_history - missing join keys")
    
    # Join demographics
    if 'customer_ref' in df.columns and 'cust_id' in demo.columns:
        df = df.merge(demo, left_on='customer_ref', right_on='cust_id', how='left', suffixes=('', '_demo'))
        df = df.drop('cust_id', axis=1, errors='ignore')
        print(f"Joined demographics: {df.shape}")
    else:
        print("Warning: Could not join demographics - missing join keys")
    
    # Join financial_ratios
    if 'customer_ref' in df.columns and 'cust_num' in financial.columns:
        df = df.merge(financial, left_on='customer_ref', right_on='cust_num', how='left', suffixes=('', '_financial'))
        df = df.drop('cust_num', axis=1, errors='ignore')
        print(f"Joined financial_ratios: {df.shape}")
    else:
        print("Warning: Could not join financial_ratios - missing join keys")
    
    # Join geographic_data
    if 'customer_ref' in df.columns and 'id' in geo.columns:
        df = df.merge(geo, left_on='customer_ref', right_on='id', how='left', suffixes=('', '_geo'))
        df = df.drop('id', axis=1, errors='ignore')
        print(f"Joined geographic_data: {df.shape}")
    else:
        print("Warning: Could not join geographic_data - missing join keys")
    
    # Join loan_details
    if 'customer_ref' in df.columns and 'customer_id' in loan.columns:
        df = df.merge(loan, left_on='customer_ref', right_on='customer_id', how='left', suffixes=('', '_loan'))
        df = df.drop('customer_id', axis=1, errors='ignore')
        print(f"Joined loan_details: {df.shape}")
    else:
        print("Warning: Could not join loan_details - missing join keys")
    
    # Handle duplicate columns from merges
    cols_to_drop = [col for col in df.columns if any(col.endswith(suffix) for suffix in ['_credit', '_demo', '_financial', '_geo', '_loan'])]
    if cols_to_drop:
        for col in cols_to_drop:
            original_col = col.rsplit('_', 1)[0]
            if original_col in df.columns and col in df.columns:
                df = df.drop(col, axis=1)
    
    print(f"\nFinal joined dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    if 'default' in df.columns:
        print(f"Default rate: {df['default'].mean():.4f}")
    
    return df


def final_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Final data cleaning steps"""
    print("\n" + "="*80)
    print("FINAL CLEANING")
    print("="*80)
    
    # Drop identifier columns (keep customer_ref for reference if needed)
    id_cols = ['customer_ref', 'application_id', 'previous_zip_code', 'loan_officer_id']
    cols_to_drop = [col for col in id_cols if col in df.columns]
    if cols_to_drop:
        print(f"Dropping identifier columns: {cols_to_drop}")
    
    # Handle infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # Fill remaining missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['object', 'category']:
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = df[col].fillna(df[col].median())
    
    # Remove duplicate rows
    initial_shape = df.shape[0]
    df = df.drop_duplicates()
    if df.shape[0] < initial_shape:
        print(f"Removed {initial_shape - df.shape[0]} duplicate rows")
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Missing values remaining: {df.isnull().sum().sum()}")
    
    return df

