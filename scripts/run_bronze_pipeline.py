"""
Script to run Bronze Layer Pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.bronze_pipeline import process_bronze_layer
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run Bronze Layer Pipeline')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input DuckDB database with raw tables')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save bronze layer data (default: data/bronze/dataset.duckdb)')
    
    args = parser.parse_args()
    
    process_bronze_layer(args.input, args.output)
    print("\n✓ Bronze pipeline complete!")


if __name__ == "__main__":
    main()

