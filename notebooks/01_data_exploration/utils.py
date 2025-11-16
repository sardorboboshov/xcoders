import pandas as pd
import numpy as np

def analyze_dataframe(df, table_name, exclude_cols=[]):
    """
    Analyzes a pandas DataFrame by calculating statistics for numeric and categorical columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to analyze
    
    exclude_cols : list
        List of columns to exclude from analysis
    
    Returns:
    --------
    None (prints analysis results)    
    """
    
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
    
    # ============================================================================
    # NUMERIC COLUMNS ANALYSIS
    # ============================================================================
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
            print(f"1st Percentile  : {df[col].quantile(0.01):.4f}")
            print(f"2nd Percentile  : {df[col].quantile(0.02):.4f}")
            print(f"3rd Percentile  : {df[col].quantile(0.03):.4f}")
            print(f"4th Percentile  : {df[col].quantile(0.04):.4f}")
            print(f"5th Percentile  : {df[col].quantile(0.05):.4f}")
            print(f"25th Percentile : {df[col].quantile(0.25):.4f}")
            print(f"50th Percentile : {df[col].quantile(0.50):.4f}")
            print(f"75th Percentile : {df[col].quantile(0.75):.4f}")
            print(f"95th Percentile : {df[col].quantile(0.95):.4f}")
    else:
        print("\nNo numeric columns found.")
    
    # ============================================================================
    # CATEGORICAL COLUMNS ANALYSIS
    # ============================================================================
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
