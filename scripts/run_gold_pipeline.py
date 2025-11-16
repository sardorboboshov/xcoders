"""
Script to run Gold Layer Pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.gold_pipeline import process_gold_layer
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run Gold Layer Pipeline')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to silver layer DuckDB database')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save gold layer data (default: data/gold/modeling_dataset.duckdb)')
    
    args = parser.parse_args()
    
    process_gold_layer(args.input, args.output)
    print("\n✓ Gold pipeline complete!")


if __name__ == "__main__":
    main()

