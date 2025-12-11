#!/usr/bin/env python3
"""
Split a JSONL file into train and validation sets.

Usage:
    python split_jsonl.py <input_file> <output_prefix> [--train_ratio 0.8] [--seed 42]

Example:
    python split_jsonl.py data.jsonl output/data --train_ratio 0.8 --seed 42

    This will create:
    - output/data_train.jsonl (80% of data)
    - output/data_val.jsonl (20% of data)
"""

import json
import random
import argparse
import sys
from pathlib import Path


def split_jsonl(input_file, output_prefix, train_ratio=0.8, seed=42):
    """
    Split a JSONL file into train and validation sets.

    Args:
        input_file: Path to input JSONL file
        output_prefix: Output file prefix (will create {prefix}_train.jsonl and {prefix}_val.jsonl)
        train_ratio: Ratio of training data (default: 0.8)
        seed: Random seed for reproducibility (default: 42)
    """

    # Set random seed for reproducibility
    random.seed(seed)

    # Read all lines from JSONL file
    data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON at line {line_num}: {e}", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    if not data:
        print("Error: No valid data found in input file.", file=sys.stderr)
        sys.exit(1)

    # Shuffle data
    random.shuffle(data)

    # Split data
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Create output directory if needed
    output_path = Path(output_prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write train set
    train_file = f"{output_prefix}_train.jsonl"
    try:
        with open(train_file, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Train set: {train_file} ({len(train_data)} samples)")
    except Exception as e:
        print(f"Error writing train file: {e}", file=sys.stderr)
        sys.exit(1)

    # Write validation set
    val_file = f"{output_prefix}_val.jsonl"
    try:
        with open(val_file, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Val set: {val_file} ({len(val_data)} samples)")
    except Exception as e:
        print(f"Error writing val file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal samples: {len(data)}")
    print(f"Train ratio: {train_ratio:.1%}")
    print(f"Val ratio: {1 - train_ratio:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Split a JSONL file into train and validation sets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python split_jsonl.py data.jsonl output/data
  python split_jsonl.py data.jsonl output/data --train_ratio 0.9
  python split_jsonl.py data.jsonl output/data --train_ratio 0.7 --seed 123
        """
    )

    parser.add_argument('input_file', help='Input JSONL file path')
    parser.add_argument('output_prefix', help='Output file prefix (will create {prefix}_train.jsonl and {prefix}_val.jsonl)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Validate train_ratio
    if not 0 < args.train_ratio < 1:
        print("Error: train_ratio must be between 0 and 1 (exclusive).", file=sys.stderr)
        sys.exit(1)

    split_jsonl(args.input_file, args.output_prefix, args.train_ratio, args.seed)


if __name__ == '__main__':
    main()
