# Quick Start Guide

## 🚀 Getting Started

### 1. Install Dependencies

```bash
cd PROJECT
pip install -r requirements.txt
```

### 2. Process Data Through Medallion Layers

```bash
# Step 1: Bronze Layer - Ingest raw data
python scripts/run_bronze_pipeline.py --input data/raw/

# Step 2: Silver Layer - Clean and validate
python scripts/run_silver_pipeline.py --input data/bronze/dataset.duckdb

# Step 3: Gold Layer - Feature engineering
python scripts/run_gold_pipeline.py --input data/silver/cleaned_dataset.duckdb
```

### 3. Train Models

```bash
python scripts/train_model.py
```

### 4. Evaluate New Data (Inference)

For new evaluation data without the `default` column:

```bash
python scripts/evaluate_new_data.py --input data/evaluation_raw
```

### 5. Launch Streamlit App

```bash
streamlit run app/main.py
```

## 📋 Common Workflows

### Complete Training Pipeline

```bash
# Run all data processing steps
python scripts/run_bronze_pipeline.py --input ../dataset.duckdb
python scripts/run_silver_pipeline.py --input data/bronze/dataset.duckdb
python scripts/run_gold_pipeline.py --input data/silver/cleaned_dataset.duckdb

# Train models
python scripts/train_model.py
```

### Inference on New Data

```bash
# Process new data and get predictions
python scripts/evaluate_new_data.py \
    --input data/raw/new_applications.duckdb \
    --output outputs/predictions/new_predictions.csv
```

## 🔧 Using Modules in Python

```python
# Import modules
from src.data.loaders import load_data_from_duckdb
from src.data.cleaners import clean_application_metadata
from src.features.interactions import create_interaction_features
from src.models.train import train_xgboost_model
from src.models.predict import predict_batch

# Use in your code
dfs = load_data_from_duckdb('data/bronze/dataset.duckdb')
df_clean = clean_application_metadata(dfs['application_metadata'])
```

## 📁 Key Directories

- `data/raw/` - Original source data files
- `data/bronze/` - Raw ingested data
- `data/silver/` - Cleaned, validated data
- `data/gold/` - Feature-engineered data
- `models/checkpoints/` - Trained model files
- `models/artifacts/` - Preprocessing artifacts (encoders, thresholds)
- `outputs/predictions/` - Prediction results
- `outputs/visualizations/` - Generated plots and charts

## ⚠️ Important Notes

1. **Data Flow**: Always process data through Bronze → Silver → Gold layers in order
2. **Feature Names**: The order of features matters! Always use the saved `feature_names.pkl`
3. **Inference**: New data must go through the same preprocessing pipeline
4. **Models**: Models are saved in `models/checkpoints/`, artifacts in `models/artifacts/`

## 🆘 Troubleshooting

### Import Errors
Make sure you're running scripts from the PROJECT directory or have added it to your Python path.

### Missing Files
Check that you've run the data processing pipelines in order (Bronze → Silver → Gold).

### Model Not Found
Ensure you've run `train_model.py` and models are saved in `models/checkpoints/`.

