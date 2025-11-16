"""
Script to evaluate new data (without default column)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.inference_pipeline import run_inference_pipeline
import argparse


def main():
    parser = argparse.ArgumentParser(description='Evaluate new loan application data')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to raw new data (raw files directory)')
    parser.add_argument('--models-dir', type=str, default=None,
                       help='Directory containing trained models (default: models/checkpoints)')
    parser.add_argument('--artifacts-dir', type=str, default=None,
                       help='Directory containing preprocessing artifacts (default: models/artifacts)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for predictions (default: outputs/predictions/evaluation_predictions.csv)')
    
    args = parser.parse_args()
    
    results = run_inference_pipeline(
        raw_data_path=args.input,
        models_dir=args.models_dir,
        artifacts_dir=args.artifacts_dir,
        output_path=args.output
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"Results saved to: {args.output or 'outputs/predictions/evaluation_predictions.csv'}")


if __name__ == "__main__":
    main()

