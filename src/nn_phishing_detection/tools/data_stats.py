"""
Quick script to show dataset statistics and label distribution.

This module provides command-line utility for analyzing phishing detection datasets,
displaying statistics including label distribution, source breakdown,
text length statistics, and cross-tabulation of labels by source.

Author: David Schatz <david.schatz@cl.uni-heidelberg.de>
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from nn_phishing_detection.data_utils import load_data


def main():
    """
    Display dataset statistics and distributions.

    Provides detailed analysis of phishing detection datasets including:
    - Total sample count
    - Label distribution (legitimate vs malicious)
    - Source distribution across datasets
    - Text length statistics (mean, median, min, max, std)
    - Detection of empty text samples
    - Cross-tabulation of labels by source

    Returns
    -------
    None
        Prints formatted statistics to stdout.

    Notes
    -----
    Accepts command-line arguments:
    - --data: Path to dataset file or directory (default: data/raw)
    - --normalize: Apply normalization during data loading

    Outputs are formatted with section headers and proper alignment.
    Percentages are calculated for all distributions.
    Warns if empty text samples are detected in the dataset.
    """
    parser = argparse.ArgumentParser(description="Show dataset statistics")
    parser.add_argument("--data", type=str, default="data/raw", help="Path to data")
    parser.add_argument(
        "--normalize", action="store_true", help="Load with normalization"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    # Load data
    print(f"\n Loading data from {args.data}...")
    df = load_data(args.data, normalize=args.normalize)

    # Basic stats
    print(f"\n Total samples: {len(df):,}")

    # Label distribution
    print("\n Label distribution:")
    label_counts = df["label"].value_counts().sort_index()

    # Map numeric labels to names if applicable
    label_names = {0: "Legitimate", 1: "Malicious"}

    for label, count in label_counts.items():
        name = label_names.get(label, f"Label {label}")
        percentage = 100 * count / len(df)
        print(f"  {name} ({label}): {count:,} samples ({percentage:.1f}%)")

    # Source distribution if available
    if "source" in df.columns:
        print("\n Source distribution:")
        source_counts = df["source"].value_counts()
        for source, count in source_counts.items():
            percentage = 100 * count / len(df)
            print(f"  {source}: {count:,} samples ({percentage:.1f}%)")

    # Text length statistics
    print("\n Text length statistics:")
    text_lengths = df["text"].str.len()
    print(f"  Mean: {text_lengths.mean():.0f} chars")
    print(f"  Median: {text_lengths.median():.0f} chars")
    print(f"  Min: {text_lengths.min():.0f} chars")
    print(f"  Max: {text_lengths.max():.0f} chars")
    print(f"  Std: {text_lengths.std():.0f} chars")

    # Check for empty texts
    empty_texts = df[df["text"].str.len() == 0]
    if len(empty_texts) > 0:
        print(f"\n  Warning: {len(empty_texts)} empty text samples found!")

    # Cross-tabulation of labels and sources
    if "source" in df.columns:
        print("\n Label distribution by source:")
        cross_tab = pd.crosstab(df["source"], df["label"], margins=True)
        print(cross_tab.to_string())

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
