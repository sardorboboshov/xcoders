# Loan Default Prediction System

A comprehensive machine learning system for predicting loan defaults using Medallion Architecture (Bronze-Silver-Gold) with advanced feature engineering, ensemble modeling, and a professional Streamlit web application.

## 🎯 Project Overview

This system predicts loan defaults using multiple gradient boosting models (XGBoost, CatBoost, LightGBM) with advanced feature engineering to achieve high precision and recall on imbalanced data. The project follows a modular architecture with clear separation of concerns and follows the Medallion data architecture pattern.

## 📁 Project Structure

```
PROJECT/
├── data/                          # Data storage (Medallion Architecture)
│   ├── raw/                       # Raw source data files
│   ├── bronze/                    # Bronze Layer: Raw ingested data
│   ├── silver/                    # Silver Layer: Cleaned, validated data
│   └── gold/                      # Gold Layer: Feature-engineered, modeling-ready
│
├── src/                           # Source code (Python modules)
│   ├── data/                      # Data processing modules
│   │   ├── loaders.py             # Data loading functions
│   │   ├── cleaners.py            # Data cleaning functions
│   │   └── transformers.py        # Data transformation functions
│   │
│   ├── features/                  # Feature engineering
│   │   └── interactions.py        # Interaction feature creation
│   │
│   ├── models/                    # Model-related code
│   │   ├── train.py               # Training functions
│   │   ├── evaluate.py            # Evaluation functions
│   │   ├── predict.py             # Prediction functions
│   │   └── ensemble.py            # Ensemble methods
│   │
│   ├── preprocessing/             # Preprocessing utilities
│   │   ├── encoders.py            # Label encoders
│   │   └── samplers.py            # SMOTE, ADASYN, etc.
│   │
│   ├── pipelines/                 # End-to-end pipelines
│   │   ├── bronze_pipeline.py     # Bronze layer processing
│   │   ├── silver_pipeline.py     # Silver layer processing
│   │   ├── gold_pipeline.py       # Gold layer processing
│   │   ├── training_pipeline.py   # Model training pipeline
│   │   └── inference_pipeline.py  # Inference pipeline for new data
│   │
│   └── utils/                     # General utilities
│       ├── paths.py               # Path management
│       └── helpers.py             # Helper functions
│
├── notebooks/                     # Jupyter notebooks for EDA
│   ├── 01_data_exploration/       # Initial data exploration
│   ├── 02_data_quality/           # Data quality analysis
│   ├── 03_feature_analysis/       # Feature analysis
│   └── 04_model_analysis/         # Model analysis
│
├── scripts/                       # Executable scripts
│   ├── run_bronze_pipeline.py     # Ingest raw data
│   ├── run_silver_pipeline.py     # Clean and validate
│   ├── run_gold_pipeline.py       # Feature engineering
│   ├── train_model.py             # Train models
│   └── evaluate_new_data.py       # Evaluate new data (inference)
│
├── models/                        # Trained models and artifacts
│   ├── checkpoints/               # Model checkpoints
│   ├── artifacts/                 # Preprocessing artifacts
│   └── metadata/                  # Model metadata
│
├── outputs/                       # Generated outputs
│   ├── reports/                   # Analysis reports
│   ├── visualizations/            # Plots and charts
│   └── predictions/               # Prediction outputs
│
├── config/                        # Configuration files
│   ├── data_config.yaml           # Data paths and settings
│   ├── model_config.yaml          # Model hyperparameters
│   └── pipeline_config.yaml       # Pipeline settings
│
├── app/                           # Streamlit application
│   ├── main.py                    # Streamlit app entry point
│   ├── pages/                     # Multi-page app structure
│   └── components/                # Reusable components
│
└── requirements.txt               # Python dependencies
```

## 🏗️ Architecture: Medallion Pattern

The project follows the **Medallion Architecture** pattern for data warehousing:

1. **Bronze Layer** (`data/bronze/`): Raw ingested data, exactly as received from source
2. **Silver Layer** (`data/silver/`): Cleaned, validated, and standardized data
3. **Gold Layer** (`data/gold/`): Feature-engineered, business-ready data for modeling

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd PROJECT
pip install -r requirements.txt
```

### 2. Data Processing Pipeline

#### Step 1: Bronze Layer - Ingest Raw Data

```bash
python scripts/run_bronze_pipeline.py --input ../dataset.duckdb
```

This loads raw data from the source DuckDB database into the bronze layer.

#### Step 2: Silver Layer - Clean and Validate

```bash
python scripts/run_silver_pipeline.py --input data/raw/
```

This cleans, standardizes, and validates the data.

#### Step 3: Gold Layer - Feature Engineering

```bash
python scripts/run_gold_pipeline.py --input data/silver/cleaned_dataset.duckdb
```

This creates advanced features and prepares data for modeling.

### 3. Train Models

```bash
python scripts/train_model.py
```

This will:
- Load data from gold layer
- Train XGBoost, CatBoost, and LightGBM models
- Find optimal classification threshold
- Evaluate models and create ensemble
- Save models and artifacts

### 4. Evaluate New Data (Inference)

For new evaluation data (same structure but without `default` column):

```bash
python scripts/evaluate_new_data.py --input path/to/new_data_raw
```

This will:
- Process new data through Bronze → Silver → Gold layers
- Apply same preprocessing as training
- Make predictions using trained models
- Save predictions with probabilities and risk levels

### 5. Launch Streamlit App

```bash
streamlit run app/main.py
```

## 📊 Key Features

### Advanced Feature Engineering

- **Temporal Features**: Business hours, weekends, account age
- **Credit Risk Indicators**: Credit score categories, utilization ratios, delinquency flags
- **Financial Health Metrics**: DTI ratios, cash flow indicators, debt service capacity
- **Interaction Features**: Credit-DTI interactions, income-loan ratios, payment coverage
- **Categorical Standardization**: Normalized employment types, loan types, education levels

### Model Training

- **Ensemble Approach**: Combines XGBoost, CatBoost, and LightGBM
- **Class Imbalance Handling**: Optimized scale_pos_weight and class weights
- **Threshold Optimization**: Finds optimal threshold for balanced precision/recall
- **Early Stopping**: Prevents overfitting with validation monitoring

### Inference Pipeline

- **Complete Pipeline**: Processes new data through all layers
- **Consistent Preprocessing**: Uses same transformations as training
- **Handles Missing Target**: Automatically handles data without `default` column
- **Batch Processing**: Efficient processing of multiple records

## 🔧 Usage Examples

### Using Modules in Python

```python
from src.data.loaders import load_data_from_duckdb
from src.data.cleaners import clean_application_metadata
from src.features.interactions import create_interaction_features
from src.models.train import train_xgboost_model
from src.models.predict import predict_batch

# Load data
dfs = load_data_from_duckdb('data/bronze/dataset.duckdb')

# Clean data
df_clean = clean_application_metadata(dfs['application_metadata'])

# Create features
df_features = create_interaction_features(df_clean)

# Train model
model = train_xgboost_model(X_train, y_train, X_val, y_val)

# Make predictions
predictions = predict_batch(X_new, models_dir='models/checkpoints')
```

### Running Complete Pipeline

```python
from src.pipelines.inference_pipeline import run_inference_pipeline

# Process new evaluation data
results = run_inference_pipeline(
    raw_data_path='data/raw/new_applications.duckdb',
    models_dir='models/checkpoints',
    artifacts_dir='models/artifacts',
    output_path='outputs/predictions/predictions.csv'
)
```

## 📈 Model Artifacts

After training, the following files are created:

- `models/checkpoints/xgb_model.json`: XGBoost model
- `models/checkpoints/catboost_model.cbm`: CatBoost model
- `models/checkpoints/lightgbm_model.txt`: LightGBM model
- `models/artifacts/label_encoders.pkl`: Categorical encoders
- `models/artifacts/optimal_threshold.pkl`: Optimal classification threshold
- `models/artifacts/feature_names.pkl`: Feature names (order matters!)
- `models/metadata/feature_importance.csv`: Feature importance scores

## 🎨 Streamlit Application

The Streamlit app provides:
- **Single Prediction**: Interactive form for loan application details
- **Batch Prediction**: Upload CSV files for batch processing
- **Model Performance Dashboard**: View metrics and visualizations
- **Real-time Risk Assessment**: Get instant predictions with probabilities

## 📝 Data Requirements

The system expects the following tables in DuckDB:

1. **application_metadata**: Application details, customer engagement
2. **credit_history**: Credit scores, accounts, payment history
3. **demographics**: Customer demographics, employment
4. **financial_ratios**: Income, debt, cash flow metrics
5. **geographic_data**: Regional economic indicators
6. **loan_details**: Loan characteristics, terms, purpose

## 🔍 Inference Pipeline for New Data

The inference pipeline handles new evaluation data that:
- Has the same structure as training data
- **Does NOT have the `default` column** (target variable)
- Needs to be processed through the same pipeline

The pipeline automatically:
1. Processes data through Bronze → Silver → Gold layers
2. Applies same cleaning and feature engineering
3. Uses saved preprocessing artifacts (encoders, feature names)
4. Makes predictions using trained models
5. Outputs probabilities, predictions, and risk levels

## 📚 Technical Details

### Class Imbalance Handling

The dataset has ~5% default rate. The system handles this through:
- Weighted loss functions
- Optimized class weights
- SMOTE/ADASYN oversampling
- Threshold tuning
- Ensemble averaging

### Feature Engineering Strategy

1. **Cleaning**: Handle missing values, standardize formats
2. **Encoding**: Label encoding for categoricals
3. **Binning**: Create categorical features from continuous
4. **Interactions**: Multiply and divide related features
5. **Risk Indicators**: Binary flags for high-risk conditions

### Model Selection

The ensemble combines:
- **XGBoost**: Best for structured data, handles missing values
- **CatBoost**: Excellent for categorical features
- **LightGBM**: Fast training, good performance

Final prediction is the average of all three models.

## 🤝 Contributing

To improve the model:
1. Experiment with additional features in `src/features/`
2. Try different ensemble weights
3. Tune hyperparameters in `config/model_config.yaml`
4. Add more models to the ensemble

## 📄 License

This project is developed for bank loan risk assessment.

## 🙏 Acknowledgments

Built with:
- XGBoost, CatBoost, LightGBM
- Streamlit for web interface
- DuckDB for data management
- Plotly for visualizations

---

**For questions or issues, please refer to the code comments or contact the development team.**

