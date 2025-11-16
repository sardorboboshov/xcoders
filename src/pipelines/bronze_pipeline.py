"""
Bronze Layer Pipeline: Raw data ingestion
"""

import duckdb
from typing import Dict
import pandas as pd
from src.utils.paths import get_data_path
import os

def load_application_metadata(input_file_path: str) -> pd.DataFrame:
    """
    Load application metadata from CSV file
    """
    df = pd.read_csv(input_file_path)
    return df

def load_credit_history(input_file_path: str) -> pd.DataFrame:
    """
    Load credit history from Parquet file
    """
    df = pd.read_parquet(input_file_path)
    return df

def load_demographics(input_file_path: str) -> pd.DataFrame:
    """
    Load demographics from CSV file
    """
    df = pd.read_csv(input_file_path)
    return df

def load_financial_ratios(input_file_path: str) -> pd.DataFrame:
    """
    Load financial ratios from JSONL file
    """
    df = pd.read_json(input_file_path, lines=True)
    return df

def load_geographic_data(input_file_path: str) -> pd.DataFrame:
    """
    Load geographic data from XML file
    """
    df = pd.read_xml(input_file_path)
    return df

def load_loan_details(input_file_path: str) -> pd.DataFrame:
    """
    Load loan details from Excel file
    """
    df = pd.read_excel(input_file_path)
    return df

def process_bronze_layer(
    raw_data_path: str,
    output_db_path: str = None
) -> Dict[str, pd.DataFrame]:
    """
    Bronze layer: Ingest raw data from source
    
    Parameters:
    -----------
    raw_data_path : str
        Path to raw data directory
    output_db_path : str, optional
        Path to save bronze layer data (default: data/bronze/dataset.duckdb)
    
    Returns:
    --------
    dict: Dictionary of DataFrames
    """
    print("="*80)
    print("BRONZE LAYER: Ingesting Raw Data")
    print("="*80)
    
    # Load raw data
    dfs = {
        'application_metadata': load_application_metadata(os.path.join(raw_data_path, 'application_metadata.csv')),
        'credit_history': load_credit_history(os.path.join(raw_data_path, 'credit_history.parquet')),
        'demographics': load_demographics(os.path.join(raw_data_path, 'demographics.csv')),
        'financial_ratios': load_financial_ratios(os.path.join(raw_data_path, 'financial_ratios.jsonl')),
        'geographic_data': load_geographic_data(os.path.join(raw_data_path, 'geographic_data.xml')),
        'loan_details': load_loan_details(os.path.join(raw_data_path, 'loan_details.xlsx'))
    }
    
    # Save to bronze layer
    if output_db_path is None:
        bronze_path = get_data_path('bronze')
        bronze_path.mkdir(parents=True, exist_ok=True)
        output_db_path = str(bronze_path / 'dataset.duckdb')
    
    con = duckdb.connect(output_db_path)
    for table_name, df in dfs.items():
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
        print(f"Saved {table_name} to bronze layer")
    con.close()
    
    print(f"✓ Bronze layer saved to: {output_db_path}")
    return dfs

