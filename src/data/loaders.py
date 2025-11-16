"""
Data loading functions
"""

import duckdb
import pandas as pd
from typing import Dict


def load_data_from_duckdb(db_path: str = 'dataset.duckdb') -> Dict[str, pd.DataFrame]:
    """
    Load all tables from DuckDB
    
    Parameters:
    -----------
    db_path : str
        Path to DuckDB database file
    
    Returns:
    --------
    dict: Dictionary of table names and DataFrames
    """
    print("="*80)
    print("LOADING DATA FROM DUCKDB")
    print("="*80)
    
    con = duckdb.connect(db_path)
    
    # Load all tables
    tables = {
        'application_metadata': 'application_metadata',
        'credit_history': 'credit_history',
        'demographics': 'demographics',
        'financial_ratios': 'financial_ratios',
        'geographic_data': 'geographic_data',
        'loan_details': 'loan_details'
    }
    
    dfs = {}
    for key, table_name in tables.items():
        try:
            query = f"SELECT * FROM {table_name}"
            dfs[key] = con.execute(query).fetch_df()
            print(f"Loaded {table_name}: {dfs[key].shape}")
        except Exception as e:
            print(f"Warning: Could not load {table_name}: {e}")
    
    con.close()
    
    return dfs

