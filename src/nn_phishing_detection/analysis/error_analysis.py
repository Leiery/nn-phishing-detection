 #ruff: noqa: E501
"""Error analysis for misclassified samples."""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def save_error_analysis(
    model,
    dataloader,
    dataset_df,
    device,
    output_dir,
    max_samples=50
):
    """
    Analyze and save false positives and false negatives.

    Args:
        model: Trained model
        dataloader: Test dataloader
        dataset_df: DataFrame with original texts
        device: torch device
        output_dir: Directory to save analysis
        max_samples: Maximum samples to save per category
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_indices = []

    # Get predictions
    print("Getting model predictions...")
    with torch.no_grad():
        idx = 0
        for texts, labels, lengths in tqdm(dataloader):
            texts = texts.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs = model(texts, lengths)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            batch_size = len(labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_indices.extend(range(idx, idx + batch_size))
            idx += batch_size

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Find misclassifications
    misclassified = all_preds != all_labels
    misclassified_indices = np.where(misclassified)[0]

    # Separate false positives and false negatives
    false_positives = []  # Predicted malicious but actually legitimate
    false_negatives = []  # Predicted legitimate but actually malicious

    for idx in misclassified_indices:
        actual_idx = all_indices[idx]

        sample_info = {
            'index': int(actual_idx),
            'true_label': int(all_labels[idx]),
            'predicted_label': int(all_preds[idx]),
            'confidence': float(all_probs[idx][all_preds[idx]]),
            'legitimate_prob': float(all_probs[idx][0]),
            'malicious_prob': float(all_probs[idx][1]) if all_probs[idx].shape[0] > 1 else 0.0,
            'text': dataset_df.iloc[actual_idx]['text'] if actual_idx < len(dataset_df) else "Text not found",
            'source': dataset_df.iloc[actual_idx]['source'] if 'source' in dataset_df.columns and actual_idx < len(dataset_df) else "Unknown"
        }

        if all_labels[idx] == 0 and all_preds[idx] == 1:
            # False positive: legitimate classified as malicious
            false_positives.append(sample_info)
        elif all_labels[idx] == 1 and all_preds[idx] == 0:
            # False negative: malicious classified as legitimate
            false_negatives.append(sample_info)

    # Sort by confidence (ascending for easier analysis of uncertain predictions)
    false_positives = sorted(false_positives, key=lambda x: x['confidence'])
    false_negatives = sorted(false_negatives, key=lambda x: x['confidence'])

    # Limit samples if needed
    if max_samples:
        false_positives = false_positives[:max_samples]
        false_negatives = false_negatives[:max_samples]

    # Create output directory
    analysis_dir = Path(output_dir) / 'error_analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Save false positives
    fp_file = analysis_dir / 'false_positives.json'
    with open(fp_file, 'w') as f:
        json.dump(false_positives, f, indent=2)
    print(f"Saved {len(false_positives)} false positives to {fp_file}")

    # Save false negatives
    fn_file = analysis_dir / 'false_negatives.json'
    with open(fn_file, 'w') as f:
        json.dump(false_negatives, f, indent=2)
    print(f"Saved {len(false_negatives)} false negatives to {fn_file}")

    # Create summary report
    summary = {
        'total_samples': len(all_labels),
        'correct_predictions': int(np.sum(~misclassified)),
        'misclassified': int(np.sum(misclassified)),
        'accuracy': float(np.mean(~misclassified)),
        'false_positives_count': len([x for x in misclassified_indices if all_labels[x] == 0 and all_preds[x] == 1]),
        'false_negatives_count': len([x for x in misclassified_indices if all_labels[x] == 1 and all_preds[x] == 0]),
        'false_positive_rate': float(np.sum((all_labels == 0) & (all_preds == 1)) / np.sum(all_labels == 0)) if np.sum(all_labels == 0) > 0 else 0,
        'false_negative_rate': float(np.sum((all_labels == 1) & (all_preds == 0)) / np.sum(all_labels == 1)) if np.sum(all_labels == 1) > 0 else 0,
    }

    summary_file = analysis_dir / 'error_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nError Analysis Summary saved to {summary_file}")

    # Create readable text files for easy inspection
    fp_text_file = analysis_dir / 'false_positives.txt'
    with open(fp_text_file, 'w') as f:
        f.write("FALSE POSITIVES (Legitimate emails classified as Malicious)\n")
        f.write("=" * 80 + "\n\n")

        for i, sample in enumerate(false_positives[:20], 1):  # First 20 for readability
            f.write(f"Sample {i}:\n")
            f.write(f"Confidence: {sample['confidence']:.2%}\n")
            f.write(f"Source: {sample['source']}\n")
            f.write(f"Text (first 500 chars):\n{sample['text'][:500]}...\n")
            f.write("-" * 40 + "\n\n")

    fn_text_file = analysis_dir / 'false_negatives.txt'
    with open(fn_text_file, 'w') as f:
        f.write("FALSE NEGATIVES (Malicious emails classified as Legitimate)\n")
        f.write("=" * 80 + "\n\n")

        for i, sample in enumerate(false_negatives[:20], 1):  # First 20 for readability
            f.write(f"Sample {i}:\n")
            f.write(f"Confidence: {sample['confidence']:.2%}\n")
            f.write(f"Source: {sample['source']}\n")
            f.write(f"Text (first 500 chars):\n{sample['text'][:500]}...\n")
            f.write("-" * 40 + "\n\n")

    print("\nReadable error analysis saved to:")
    print(f"  - {fp_text_file}")
    print(f"  - {fn_text_file}")

    # Print summary statistics
    print("\n" + "=" * 50)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total test samples: {summary['total_samples']}")
    print(f"Correct predictions: {summary['correct_predictions']} ({summary['accuracy']:.2%})")
    print(f"Misclassified: {summary['misclassified']}")
    print(f"  - False Positives: {summary['false_positives_count']} (FPR: {summary['false_positive_rate']:.2%})")
    print(f"  - False Negatives: {summary['false_negatives_count']} (FNR: {summary['false_negative_rate']:.2%})")

    return summary, false_positives, false_negatives