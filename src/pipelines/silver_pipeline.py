"""
Silver Layer Pipeline: Data cleaning and validation
"""

import duckdb
import pandas as pd
from src.data.loaders import load_data_from_duckdb
from src.utils.paths import get_data_path
from src.data.cleaners import (
    clean_application_metadata,
    clean_credit_history,
    clean_demographics,
    clean_financial_ratios,
    clean_geographic_data,
    clean_loan_details
)

def process_silver_layer(
    bronze_db_path: str,
    output_db_path: str = None
) -> pd.DataFrame:
    """
    Silver layer: Clean and validate data
    
    Parameters:
    -----------
    bronze_db_path : str
        Path to bronze layer DuckDB database
    output_db_path : str, optional
        Path to save silver layer data (default: data/silver/cleaned_dataset.duckdb)
    
    Returns:
    --------
    DataFrame: Cleaned dataset
    """
    print("="*80)
    print("SILVER LAYER: Cleaning and Validating Data")
    print("="*80)
    
    # Load from bronze
    dfs = load_data_from_duckdb(bronze_db_path)
    
    # Join and clean tables
    df_cleaned = {
        'application_metadata': clean_application_metadata(dfs['application_metadata']),
        'credit_history': clean_credit_history(dfs['credit_history']),
        'demographics': clean_demographics(dfs['demographics']),
        'financial_ratios': clean_financial_ratios(dfs['financial_ratios']),
        'geographic_data': clean_geographic_data(dfs['geographic_data']),
        'loan_details': clean_loan_details(dfs['loan_details'])
    }
    
    # Save to silver layer
    if output_db_path is None:
        silver_path = get_data_path('silver')
        silver_path.mkdir(parents=True, exist_ok=True)
        output_db_path = str(silver_path / 'cleaned_dataset.duckdb')
    
    con = duckdb.connect(output_db_path)
    for table_name, df in df_cleaned.items():
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
        print(f"Saved {table_name} to silver layer")
    con.close()
    print(f"✓ Silver layer saved to: {output_db_path}")
    return df_cleaned

