"""
Inference Pipeline: Process new evaluation data and make predictions
"""

import pickle
import pandas as pd
from pathlib import Path
from typing import Dict

from src.pipelines.bronze_pipeline import process_bronze_layer
from src.pipelines.silver_pipeline import process_silver_layer
from src.pipelines.gold_pipeline import process_gold_layer
from src.preprocessing.encoders import encode_for_inference
from src.models.predict import predict_batch, load_models
from src.utils.paths import get_models_path, get_data_path, get_outputs_path


def preprocess_for_inference(
    df: pd.DataFrame,
    label_encoders: Dict,
    feature_names: list,
    drop_target: bool = True
) -> pd.DataFrame:
    """
    Preprocess new data for prediction using same transformations as training
    
    Parameters:
    -----------
    df : DataFrame
        New data (may or may not have 'default' column)
    label_encoders : dict
        Label encoders from training
    feature_names : list
        Expected feature names in order
    drop_target : bool
        Whether to drop 'default' column if present
    
    Returns:
    --------
    DataFrame: Preprocessed data ready for prediction
    """
    df = df.copy()
    
    # Drop target if present
    if drop_target and 'default' in df.columns:
        df = df.drop(columns=['default'])
    
    # Drop identifier columns
    id_cols = ['customer_ref', 'application_id']
    df = df.drop(columns=[col for col in id_cols if col in df.columns])
    
    # Encode categorical features
    categorical_cols = list(label_encoders.keys())
    df = encode_for_inference(df, label_encoders, categorical_cols)
    
    # Ensure all expected features are present
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        for feat in missing_features:
            df[feat] = 0
    
    # Remove any extra features
    extra_features = set(df.columns) - set(feature_names)
    if extra_features:
        print(f"Warning: Extra features (will be dropped): {extra_features}")
        df = df.drop(columns=extra_features)
    
    # Reorder columns to match training order
    df = df[feature_names]
    
    # Handle missing values
    for col in df.columns:
        if df[col].isnull().any():
            if col in label_encoders:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(0)
    
    return df


def run_inference_pipeline(
    raw_data_path: str,
    models_dir: str = None,
    artifacts_dir: str = None,
    output_path: str = None
) -> pd.DataFrame:
    """
    Complete inference pipeline for new evaluation data
    
    Parameters:
    -----------
    raw_data_path : str
        Path to raw new data (DuckDB file)
    models_dir : str, optional
        Directory containing trained models (default: models/checkpoints)
    artifacts_dir : str, optional
        Directory containing preprocessing artifacts (default: models/artifacts)
    output_path : str, optional
        Where to save predictions (default: outputs/predictions/evaluation_predictions.csv)
    
    Returns:
    --------
    DataFrame: Predictions with probabilities and risk levels
    """
    print("="*80)
    print("INFERENCE PIPELINE FOR NEW EVALUATION DATA")
    print("="*80)
    
    # Set default paths
    if models_dir is None:
        models_dir = str(get_models_path('checkpoints'))
    if artifacts_dir is None:
        artifacts_dir = str(get_models_path('artifacts'))
    if output_path is None:
        output_path = str(get_outputs_path('predictions') / 'evaluation_predictions.csv')
    
    # Step 1: Bronze - Ingest raw data
    bronze_db = str(get_data_path('bronze') / 'evaluation_data.duckdb')
    process_bronze_layer(raw_data_path, bronze_db)
    
    # Step 2: Silver - Clean data
    silver_db = str(get_data_path('silver') / 'evaluation_cleaned.duckdb')
    df_silver = process_silver_layer(bronze_db, silver_db)
    
    # Step 3: Gold - Feature engineering
    gold_db = str(get_data_path('gold') / 'evaluation_features.duckdb')
    df_gold = process_gold_layer(silver_db, gold_db)
    
    # Step 4: Load preprocessing artifacts
    print("\nLoading preprocessing artifacts...")
    with open(f'{artifacts_dir}/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    with open(f'{artifacts_dir}/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    with open(f'{artifacts_dir}/optimal_threshold.pkl', 'rb') as f:
        threshold = pickle.load(f)
    
    # Step 5: Preprocess for inference
    print("\nPreprocessing for inference...")
    df_processed = preprocess_for_inference(
        df_gold,
        label_encoders,
        feature_names,
        drop_target=True
    )
    
    # Step 6: Load models and predict
    print("\nMaking predictions...")
    models = load_models(models_dir)
    predictions = predict_batch(
        df_processed,
        models=models,
        threshold=threshold
    )
    
    # Step 7: Combine with original data and save
    results_df = df_gold.copy()
    if 'customer_ref' in results_df.columns:
        results_df = results_df[['customer_ref']].copy()
    else:
        results_df = pd.DataFrame(index=df_gold.index)
    
    results_df['prob'] = predictions['probabilities'].round(5)
    results_df['default'] = predictions['predictions'].astype(float)
    results_df.rename(columns={'customer_ref': 'customer_id'}, inplace=True)
    # results_df['risk_level'] = results_df['default_probability'].apply(
    #     lambda x: 'High' if x >= 0.6 else ('Medium' if x >= 0.3 else 'Low')
    # )
    
    # Save predictions
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Predictions saved to: {output_path}")
    print(f"  Total records: {len(results_df)}")
    # print(f"  High risk: {(results_df['risk_level'] == 'High').sum()}")
    # print(f"  Medium risk: {(results_df['risk_level'] == 'Medium').sum()}")
    # print(f"  Low risk: {(results_df['risk_level'] == 'Low').sum()}")
    
    return results_df

