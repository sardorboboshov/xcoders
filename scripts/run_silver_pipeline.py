"""
Script to run Silver Layer Pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.silver_pipeline import process_silver_layer
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run Silver Layer Pipeline')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to bronze layer DuckDB database')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save silver layer data (default: data/silver/cleaned_dataset.duckdb)')
    
    args = parser.parse_args()
    
    process_silver_layer(args.input, args.output)
    print("\n✓ Silver pipeline complete!")


if __name__ == "__main__":
    main()

