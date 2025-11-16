"""
Script to run Training Pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.training_pipeline import run_training_pipeline
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run Training Pipeline')
    parser.add_argument('--input', type=str, default=None,
                       help='Path to gold layer database (default: data/gold/modeling_dataset.duckdb)')
    parser.add_argument('--models-dir', type=str, default=None,
                       help='Directory to save models (default: models/checkpoints)')
    parser.add_argument('--artifacts-dir', type=str, default=None,
                       help='Directory to save artifacts (default: models/artifacts)')
    
    args = parser.parse_args()
    
    run_training_pipeline(
        gold_db_path=args.input,
        models_dir=args.models_dir,
        artifacts_dir=args.artifacts_dir
    )
    print("\n✓ Training pipeline complete!")


if __name__ == "__main__":
    main()

