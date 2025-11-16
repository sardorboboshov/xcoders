"""
Helper utility functions
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def analyze_dataframe(df: pd.DataFrame, table_name: str, exclude_cols: Optional[List[str]] = None):
    """
    Analyzes a pandas DataFrame by calculating statistics for numeric and categorical columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to analyze
    table_name : str
        Name of the table
    exclude_cols : list, optional
        List of columns to exclude from analysis
    """
    if exclude_cols is None:
        exclude_cols = []
    
    print("=" * 80)
    print(f"ANALYSIS OF TABLE: {table_name}")
    print("=" * 80)

    print(f"Columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    # Numeric columns analysis
    if numeric_cols:
        print("\n" + "=" * 80)
        print("NUMERIC COLUMNS ANALYSIS")
        print("=" * 80)
        
        for col in numeric_cols:
            print(f"\n--- {col} ---")
            print(f"Datatype        : {df[col].dtype}")
            print(f"Null Count      : {df[col].isna().sum()}")
            print(f"Mean            : {df[col].mean():.4f}")
            print(f"Median          : {df[col].median():.4f}")
            print(f"Min             : {df[col].min():.4f}")
            print(f"Max             : {df[col].max():.4f}")
            print(f"Std Dev         : {df[col].std():.4f}")
            print(f"25th Percentile : {df[col].quantile(0.25):.4f}")
            print(f"50th Percentile : {df[col].quantile(0.50):.4f}")
            print(f"75th Percentile : {df[col].quantile(0.75):.4f}")
    else:
        print("\nNo numeric columns found.")
    
    # Categorical columns analysis
    if categorical_cols:
        print("\n" + "=" * 80)
        print("CATEGORICAL COLUMNS ANALYSIS")
        print("=" * 80)
        
        for col in categorical_cols:
            print(f"\n--- {col} ---")
            print(f"Datatype        : {df[col].dtype}")
            print(f"Null Count      : {df[col].isna().sum()}")
            print(f"Distinct Values : {df[col].nunique()}")
            print(f"\nTop 10 Values with Counts:")
            value_counts = df[col].value_counts().head(10)
            for idx, (value, count) in enumerate(value_counts.items(), 1):
                print(f"  {idx}. {value}: {count}")
    else:
        print("\nNo categorical columns found.")
    
    print("\n" + "=" * 80)

