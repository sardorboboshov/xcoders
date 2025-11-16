"""
Gold Layer Pipeline: Feature engineering
"""

import duckdb
from pathlib import Path
import pandas as pd
from src.data.loaders import load_data_from_duckdb
from src.data.transformers import final_cleaning, join_all_tables
from src.features.interactions import create_interaction_features
from src.utils.paths import get_data_path


def process_gold_layer(
    silver_db_path: str,
    output_db_path: str = None
) -> pd.DataFrame:
    """
    Gold layer: Feature engineering and final preparation
    
    Parameters:
    -----------
    silver_db_path : str
        Path to silver layer DuckDB database
    output_db_path : str, optional
        Path to save gold layer data (default: data/gold/modeling_dataset.duckdb)
    
    Returns:
    --------
    DataFrame: Feature-engineered dataset ready for modeling
    """
    print("="*80)
    print("GOLD LAYER: Feature Engineering")
    print("="*80)
    
    # Load from silver
    dfs = load_data_from_duckdb(silver_db_path)
    
    merged_df = join_all_tables(dfs)
    
    # Create interaction features
    df_features = create_interaction_features(merged_df)
    
    # Final cleaning
    df_final = final_cleaning(df_features)
    
    # Save to gold layer
    if output_db_path is None:
        gold_path = get_data_path('gold')
        gold_path.mkdir(parents=True, exist_ok=True)
        output_db_path = str(gold_path / 'modeling_dataset.duckdb')
    
    con = duckdb.connect(output_db_path)
    con.execute("CREATE OR REPLACE TABLE modeling_dataset AS SELECT * FROM df_final")
    con.close()
    
    print(f"✓ Gold layer saved to: {output_db_path}")
    return df_final

