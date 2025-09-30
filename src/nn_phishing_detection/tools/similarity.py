"""
Text similarity analysis tools for phishing detection datasets.

This module provides TF-IDF cosine similarity analysis for identifying duplicate
and near-duplicate emails in phishing detection datasets. Supports sampling,
similarity computation, and visualization for data quality assessment.

Author: David Schatz <schatz@cl.uni-heidelberg.de>
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from nn_phishing_detection.data_utils import load_data


def sample_dataset(df: pd.DataFrame, sample_fraction: float = 0.25) -> pd.DataFrame:
    """
    Randomly sample a fraction of the dataset for similarity analysis.

    Performs simple random sampling without stratification since the dataset
    is already nearly balanced between legitimate and malicious emails.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with email data.
    sample_fraction : float, default=0.25
        Fraction of dataset to sample (0.0 to 1.0).

    Returns
    -------
    pd.DataFrame
        Randomly sampled subset of the input DataFrame.

    Notes
    -----
    Uses random_state=42 for reproducible sampling.
    Prints detailed sampling statistics including label distribution.
    """

    n_total = len(df)
    n_sample = int(n_total * sample_fraction)

    print(f"Sampling {sample_fraction*100:.0f}% of dataset:")
    print(f"  Total: {n_total:,} samples")
    print(f"  Sample: {n_sample:,} samples")

    # Simple random sampling
    print("  Using random sampling (no stratification)")
    sampled = df.sample(n=n_sample, random_state=42)

    # Show label distribution if available
    if "label" in df.columns:
        print("  Label distribution in sample:")
        for label, count in sampled["label"].value_counts().sort_index().items():
            pct = 100 * count / len(sampled)
            print(f"    {label}: {count:,} ({pct:.1f}%)")

    print(f"  Final sample size: {len(sampled):,}")
    return sampled


def compute_tfidf_similarities(
    df: pd.DataFrame, max_features: int | None = None
) -> dict:
    """
    Compute TF-IDF cosine similarities for all pairs in the dataset.

    Creates TF-IDF vectors using character n-grams and computes pairwise
    cosine similarities to identify duplicate and near-duplicate emails.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'text' column containing email content.
    max_features : int, optional
        Maximum number of TF-IDF features. If None, uses complete vocabulary.

    Returns
    -------
    dict
        Dictionary containing:
        - similarity_matrix (np.ndarray): Pairwise similarity matrix
        - similarities (list): All pairwise similarity scores
        - high_similarity_pairs (list): Pairs with similarity > 0.8
        - cross_label_high_sim (int): High similarity pairs with different labels
        - cross_source_high_sim (int): High similarity pairs from different sources
        - statistics (dict): Summary statistics

    Notes
    -----
    Uses character n-grams (3-5) for similarity detection.
    Considers pairs with similarity > 0.8 as potentially duplicates.
    Analyzes cross-label and cross-source similarities for data quality assessment.
    """
    # Using raw text
    texts = df["text"].values
    n = len(texts)
    total_pairs = (n * (n - 1)) // 2

    print("\nComputing TF-IDF similarities:")
    print(f"  Samples: {n:,}")
    print(f"  Total pairs: {total_pairs:,}")

    # Build TF-IDF
    if max_features:
        print(f"\n Building TF-IDF vectorizer (max_features={max_features:,})...")
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), max_features=max_features
        )
    else:
        print("\n Building TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
        )

    tfidf_matrix = vectorizer.fit_transform(texts)
    vocab_size = len(vectorizer.vocabulary_)
    density = tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])

    print(f" TF-IDF matrix: {tfidf_matrix.shape}")
    print(f"   Vocabulary: {vocab_size:,} character n-grams")
    print(f"   Sparsity: {100*(1-density):.1f}%")

    # Compute similarity matrix
    print("\n Computing cosine similarity matrix...")
    print("   This may take a while for large samples...")

    similarity_matrix = cosine_similarity(tfidf_matrix)

    print(f" Similarity matrix computed: {similarity_matrix.shape}")

    # Extract all similarities (upper triangle, no diagonal)
    print("\n Extracting similarity statistics...")
    similarities = []
    high_similarity_pairs = []
    cross_label_high_sim = 0
    cross_source_high_sim = 0

    # Extract labels as array for faster access
    labels = df["label"].values if "label" in df.columns else None

    for i in tqdm(range(n), desc="Processing similarities"):
        for j in range(i + 1, n):
            sim = similarity_matrix[i, j]
            similarities.append(sim)

            # Collect high similarity examples
            if sim >= 0.9 and len(high_similarity_pairs) < 100:
                pair_data = {
                    "idx1": i,
                    "idx2": j,
                    "similarity": float(sim),
                    "text1_preview": texts[i][:150],
                    "text2_preview": texts[j][:150],
                    "label1": labels[i] if labels is not None else "unknown",
                    "label2": labels[j] if labels is not None else "unknown",
                }
                if "source" in df.columns:
                    pair_data["source1"] = df["source"].values[i]
                    pair_data["source2"] = df["source"].values[j]

                high_similarity_pairs.append(pair_data)

                # Count cross-label/source high similarities
                if labels is not None and labels[i] != labels[j]:
                    cross_label_high_sim += 1
                if (
                    "source" in df.columns
                    and df["source"].values[i] != df["source"].values[j]
                ):
                    cross_source_high_sim += 1

    similarities = np.array(similarities)

    # Calculate statistics
    stats = {
        "n_samples": n,
        "total_pairs": len(similarities),
        "vocab_size": vocab_size,
        "mean": float(np.mean(similarities)),
        "std": float(np.std(similarities)),
        "min": float(np.min(similarities)),
        "max": float(np.max(similarities)),
        "percentiles": {
            "10%": float(np.percentile(similarities, 10)),
            "25%": float(np.percentile(similarities, 25)),
            "50%": float(np.percentile(similarities, 50)),
            "75%": float(np.percentile(similarities, 75)),
            "90%": float(np.percentile(similarities, 90)),
            "95%": float(np.percentile(similarities, 95)),
            "99%": float(np.percentile(similarities, 99)),
        },
        "threshold_ranges": {
            "<0.5": int(np.sum(similarities < 0.5)),
            "0.5-0.7": int(np.sum((similarities >= 0.5) & (similarities < 0.7))),
            "0.7-0.8": int(np.sum((similarities >= 0.7) & (similarities < 0.8))),
            "0.8-0.9": int(np.sum((similarities >= 0.8) & (similarities < 0.9))),
            "0.9-0.95": int(np.sum((similarities >= 0.9) & (similarities < 0.95))),
            "->0.95": int(np.sum(similarities >= 0.95)),
        },
        "range_percentages": {
            "<0.5": float(100 * np.mean(similarities < 0.5)),
            "0.5-0.7": float(
                100 * np.mean((similarities >= 0.5) & (similarities < 0.7))
            ),
            "0.7-0.8": float(
                100 * np.mean((similarities >= 0.7) & (similarities < 0.8))
            ),
            "0.8-0.9": float(
                100 * np.mean((similarities >= 0.8) & (similarities < 0.9))
            ),
            "0.9-0.95": float(
                100 * np.mean((similarities >= 0.9) & (similarities < 0.95))
            ),
            "->0.95": float(100 * np.mean(similarities >= 0.95)),
        },
        "cross_label_high_sim": cross_label_high_sim,
        "cross_source_high_sim": cross_source_high_sim,
        "high_similarity_pairs": high_similarity_pairs,
        "all_similarities": similarities,  # For histogram
    }

    return stats


def create_individual_plots(stats: dict, output_dir: Path):
    """
    Create visualization plots for similarity analysis results.

    Generates multiple plots including similarity distribution histogram
    and threshold analysis for duplicate detection optimization.

    Parameters
    ----------
    stats : dict
        Dictionary containing similarity statistics and data from compute_tfidf_similarities().
    output_dir : Path
        Directory to save the generated plots.

    Returns
    -------
    None
        Saves multiple plot files to the output directory.

    Notes
    -----
    Creates two main visualizations:
    1. Similarity distribution histogram with statistical markers
    2. Threshold analysis showing duplicate counts at different thresholds

    Saves plots as high-resolution PNG files (150 DPI).
    """
    similarities = stats["all_similarities"]

    # 1. Histogram plot
    plt.figure(figsize=(10, 6))
    n_bins = min(100, len(np.unique(similarities)) // 2)
    plt.hist(similarities, bins=n_bins, edgecolor="black", alpha=0.7, color="steelblue")
    plt.axvline(
        x=stats["mean"], color="red", linestyle="--", label=f"Mean: {stats['mean']:.3f}"
    )
    plt.axvline(x=0.9, color="orange", linestyle="--", label="High similarity (0.9)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Number of Pairs")
    plt.title(
        f'Distribution of TF-IDF Cosine Similarities\n({stats["n_samples"]:,} samples, {stats["total_pairs"]:,} pairs)'  # noqa: E501
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    hist_path = output_dir / "similarity_histogram.png"
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Histogram saved to {hist_path}")

    # 1b. Histogram with logarithmic scale
    plt.figure(figsize=(12, 6))
    n_bins = min(100, len(np.unique(similarities)) // 2)
    counts, bins, patches = plt.hist(
        similarities, bins=n_bins, alpha=0.7, edgecolor="black", color="steelblue"
    )
    plt.yscale("log")  # Logarithmic y-axis
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Number of Pairs (log scale)")
    plt.title(
        f'Distribution of TF-IDF Cosine Similarities - Log Scale\n({stats["n_samples"]:,} samples, {stats["total_pairs"]:,} pairs)'  # noqa: E501
    )
    plt.axvline(
        x=stats["mean"], color="red", linestyle="--", label=f"Mean: {stats['mean']:.3f}"
    )
    plt.axvline(
        x=0.9, color="orange", linestyle="--", alpha=0.7, label="High similarity (0.9)"
    )
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    hist_log_path = output_dir / "similarity_histogram_log.png"
    plt.savefig(hist_log_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Histogram (log scale) saved to {hist_log_path}")

    # 2. Cumulative distribution plot
    plt.figure(figsize=(10, 6))
    sorted_sims = np.sort(similarities)
    cumulative = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
    plt.plot(sorted_sims, cumulative, linewidth=2, color="navy")
    plt.axvline(x=0.9, color="orange", linestyle="--", alpha=0.7)
    plt.axhline(y=0.9, color="red", linestyle="--", alpha=0.7, label="90th percentile")
    plt.axhline(
        y=0.95, color="darkred", linestyle="--", alpha=0.7, label="95th percentile"
    )
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Distribution of Similarities")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    cdf_path = output_dir / "similarity_cumulative.png"
    plt.savefig(cdf_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Cumulative distribution saved to {cdf_path}")

    # 3. Threshold analysis plot
    plt.figure(figsize=(12, 6))
    ranges = ["<0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", "->0.95"]
    percentages = [stats["range_percentages"][r] for r in ranges]
    colors = ["lightblue", "lightgreen", "yellow", "orange", "red", "darkred"]

    bars = plt.bar(range(len(ranges)), percentages, color=colors, edgecolor="black")
    plt.xticks(range(len(ranges)), ranges, rotation=0)
    plt.ylabel("Percentage of Pairs (%)")
    plt.title("Distribution of Similarity Ranges")

    # Handle small values
    max_pct = max(percentages) if percentages else 1
    if max_pct < 1:
        plt.ylim(0, max(1, max_pct * 1.2))
    else:
        plt.ylim(0, max_pct * 1.1)

    # Add labels on bars
    for i, (bar, pct) in enumerate(zip(bars, percentages, strict=False)):
        height = bar.get_height()
        label_y = max(height + max_pct * 0.02, max_pct * 0.05)

        if pct >= 1:
            label = f"{pct:.1f}"
        elif pct >= 0.1:
            label = f"{pct:.2f}"
        else:
            label = f"{pct:.3f}"

        plt.text(i, label_y, label, ha="center", va="bottom", fontweight="bold")

    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    thresh_path = output_dir / "similarity_thresholds.png"
    plt.savefig(thresh_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Threshold analysis saved to {thresh_path}")


def main():
    """
    Main function with command-line interface for similarity analysis.

    Provides complete pipeline for TF-IDF similarity analysis including
    data loading, sampling, similarity computation, and visualization
    with statistics and duplicate detection.

    Notes
    -----
    Supports both single files and directories with multiple CSV files.
    Generates multiple visualizations and detailed statistics.
    Creates output directory structure for organized results.
    Includes cross-label and cross-source analysis.
    """
    parser = argparse.ArgumentParser(description="TF-IDF similarity analysis")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to data file or directory"
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.25,
        help="Fraction of data to sample (default: 0.25)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Max TF-IDF features",
    )
    parser.add_argument(
        "--out-dir", type=str, default="results/similarity", help="Output directory"
    )

    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TF-IDF COSINE SIMILARITY ANALYSIS")
    print("=" * 70)

    print("\n Loading dataset...")
    df = load_data(args.data, normalize=False)
    print(f" Loaded {len(df):,} samples (raw text, no normalization)")

    # Sample dataset
    sampled_df = sample_dataset(df, args.sample_fraction)

    # Compute similarities
    stats = compute_tfidf_similarities(sampled_df, args.max_features)

    # Create individual plots
    create_individual_plots(stats, out_dir)

    # Save high similarity examples
    if stats["high_similarity_pairs"]:
        examples_df = pd.DataFrame(stats["high_similarity_pairs"])
        examples_path = out_dir / "high_similarity_examples.csv"
        examples_df.to_csv(examples_path, index=False)
        print(f" High similarity examples saved to {examples_path}")

    # Print detailed results
    print("\n" + "=" * 70)
    print("SIMILARITY ANALYSIS RESULTS")
    print("=" * 70)

    print("\nDataset Summary:")
    print(
        f"• Sample analyzed: {stats['n_samples']:,} samples ({args.sample_fraction*100:.0f}% of dataset)"  # noqa: E501
    )
    print(f"• Pairs analyzed: {stats['total_pairs']:,}")
    print(f"• TF-IDF vocabulary: {stats['vocab_size']:,} character n-grams")

    print("\nSimilarity Distribution:")
    print(f"• Mean similarity: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"• Median: {stats['percentiles']['50%']:.4f}")
    print(f"• 90th percentile: {stats['percentiles']['90%']:.4f}")
    print(f"• 95th percentile: {stats['percentiles']['95%']:.4f}")
    print(f"• Maximum: {stats['max']:.4f}")

    print("\nHigh Similarity Analysis:")
    for range_name in ["0.8-0.9", "0.9-0.95", "->0.95"]:
        count = stats["threshold_ranges"][range_name]
        pct = stats["range_percentages"][range_name]
        print(f"• {range_name}: {count:,} pairs ({pct:.3f}%)")

    if stats["cross_label_high_sim"] > 0 or stats["cross_source_high_sim"] > 0:
        print("\nCross-Category Analysis:")
        if stats["cross_label_high_sim"] > 0:
            print(
                f"• Cross-label high similarity: {stats['cross_label_high_sim']:,} pairs"
            )
        if stats["cross_source_high_sim"] > 0:
            print(
                f"• Cross-source high similarity: {stats['cross_source_high_sim']:,} pairs"
            )

    print("\n" + "=" * 70)
    print(f"Analysis complete! Results saved to {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
