"""
Data utilities for phishing detection models.

This module provides dataset classes, data loading, preprocessing, and splitting
utilities for neural network-based phishing detection. It includes support for
word-level models, transformer models, and various data formats.

Author: David Schatz <schatz@cl.uni-heidelberg.de>
"""

# ruff: noqa: E501
import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm


class WordPhishingDataset(Dataset):
    """
    Dataset for word-level phishing detection with vocabulary encoding.

    It returns tuples containing encoded sequences, labels, and sequence lengths.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'text' and 'label' columns.
    vocab : object
        Vocabulary object with an encode() method that converts text to indices.
    max_len : int
        Maximum sequence length for encoding.

    Notes
    -----
    The vocab object must implement an encode(text, max_len) method that returns
    a tuple of (encoded_tensor, actual_length).
    """

    def __init__(self, df, vocab, max_len):
        """
        Initialize the WordPhishingDataset.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'text' and 'label' columns.
        vocab : object
            Vocabulary encoder with encode() method.
        max_len : int
            Maximum sequence length.
        """
        self.texts = df["text"].values
        self.labels = df["label"].values
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing:
            - encoded (torch.Tensor): Encoded sequence of shape (max_len,)
            - label (torch.Tensor): Label tensor of shape (1,)
            - length (torch.Tensor): Actual sequence length tensor of shape (1,)
        """
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Encode text
        encoded, length = self.vocab.encode(text, self.max_len)

        return (
            encoded,
            torch.tensor(label, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
        )


class TransformerDataset(Dataset):
    """
    Dataset for transformer models (DistilBERT, BERT, etc.).

    It returns dictionaries containing input_ids, attention_mask, and labels
    in the format expected by HuggingFace models.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'text' and 'label' columns.
    tokenizer : transformers.PreTrainedTokenizer
        HuggingFace tokenizer for encoding text.
    max_length : int, default=512
        Maximum sequence length for tokenization.
    normalize : bool, default=True
        Whether to apply text normalization before tokenization.
    label_key : str, default="labels"
        Key name for labels in return dict ('labels' for training, 'label' for inference).

    Notes
    -----
    Text normalization includes URL/email replacement, lowercasing, and whitespace
    normalization. See normalize_text() function for details.
    """

    def __init__(
        self, df, tokenizer, max_length=512, normalize=True, label_key="labels"
    ):
        """
        Initialize the TransformerDataset.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'text' and 'label' columns.
        tokenizer : transformers.PreTrainedTokenizer
            HuggingFace tokenizer.
        max_length : int, default=512
            Maximum sequence length.
        normalize : bool, default=True
            Whether to normalize text.
        label_key : str, default="labels"
            Key for labels in return dict.
        """
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.normalize = normalize
        self.label_key = label_key

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        dict
            Dictionary containing:
            - input_ids (torch.Tensor): Token IDs of shape (max_length,)
            - attention_mask (torch.Tensor): Attention mask of shape (max_length,)
            - labels/label (torch.Tensor): Label tensor based on label_key
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # Apply normalization if requested
        if self.normalize:
            text = normalize_text(text)

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            self.label_key: torch.tensor(label, dtype=torch.long),
        }


def load_csv_file(filepath: Path, normalize: bool = True) -> pd.DataFrame:
    """
    Load and preprocess a single CSV file for phishing detection.

    This function handles various CSV formats, column mappings, and data
    cleaning operations. It supports files with different text column names
    (body, text, subject) and ensures consistent output format.

    Parameters
    ----------
    filepath : Path
        Path to the CSV file to load.
    normalize : bool, default=True
        Whether to apply text normalization to the content.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns: 'text', 'label', 'source'.
        Empty DataFrame if no valid data found.

    Raises
    ------
    ValueError
        If no 'label' column found or no text column found.

    Notes
    -----
    The function automatically detects text columns in this priority:
    1. 'body' column
    2. 'text' column
    3. Combined 'subject' + 'body'
    4. 'subject' column only

    Label filtering removes rows with invalid labels (not 0, 1, '0', '1').
    """
    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
    except pd.errors.ParserError:
        # Fall back to Python engine for problematic files
        print(f"  Using Python engine for {filepath.name}")
        df = pd.read_csv(
            filepath, encoding="utf-8", on_bad_lines="skip", engine="python"
        )
    source_name = filepath.stem

    if "label" not in df.columns:
        raise ValueError(f"No 'label' column found in {filepath}")

    # Filter out rows where label is not a valid binary label
    # This handles malformed CSV where text spills into label column
    if df["label"].dtype == "object":
        # Keep only rows where label is '0', '1', 0, or 1
        valid_labels = ["0", "1", 0, 1]
        mask = df["label"].isin(valid_labels)
        print(
            f"  Filtering {filepath.name}: keeping {mask.sum()}/{len(df)} rows with valid labels"
        )
        df = df[mask].copy()

        if len(df) == 0:
            print(f"  Warning: No valid labels found in {filepath.name}")
            return pd.DataFrame(columns=["text", "label", "source"])

    text_col = None
    if "body" in df.columns:
        text_col = "body"
    elif "text" in df.columns:
        text_col = "text"
    elif "subject" in df.columns and "body" in df.columns:
        df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
        text_col = "text"
    elif "subject" in df.columns:
        text_col = "subject"
    else:
        raise ValueError(f"No text column found in {filepath}")

    result_df = pd.DataFrame(
        {
            "text": df[text_col].apply(normalize_text) if normalize else df[text_col],
            "label": df["label"],
            "source": source_name,
        }
    )

    result_df = result_df[result_df["text"].str.len() > 0]

    return result_df


def load_data(data_path: str | Path, normalize: bool = True) -> pd.DataFrame:
    """
    Load phishing detection data from file or directory.

    This function can load data from a single CSV file or from all CSV files
    in a directory. It combines multiple files into a single DataFrame with
    standardized format and handles label mapping and validation.

    Parameters
    ----------
    data_path : str or Path
        Path to a CSV file or directory containing CSV files.
    normalize : bool, default=True
        Whether to apply text normalization to the content.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with columns: 'text', 'label', 'source'.
        Labels are converted to integers (0=legitimate, 1=malicious).

    Raises
    ------
    ValueError
        If path doesn't exist, no valid data loaded, or invalid labels found.

    Notes
    -----
    The function handles both string and numeric labels with mapping:
    - 'legitimate' -> 0, 'malicious' -> 1
    - '0' -> 0, '1' -> 1
    - Numeric 0 -> 0, 1 -> 1

    Each loaded file contributes a 'source' column based on filename.
    """
    data_path = Path(data_path)
    all_data = []

    if data_path.is_file():
        print(f"Loading single file: {data_path}")
        all_data.append(load_csv_file(data_path, normalize=normalize))
    elif data_path.is_dir():
        csv_files = list(data_path.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files")
        for csv_file in tqdm(csv_files, desc="Loading files"):
            try:
                df = load_csv_file(csv_file, normalize=normalize)
                print(f"  {csv_file.name}: {len(df)} samples")
                all_data.append(df)
            except Exception as e:
                print(f"  Error loading {csv_file.name}: {e}")
    else:
        raise ValueError(f"Path {data_path} does not exist")

    if not all_data:
        raise ValueError("No valid data loaded")

    combined_df = pd.concat(all_data, ignore_index=True)

    # Handle both string and numeric labels
    label_mapping = {"legitimate": 0, "malicious": 1, "0": 0, "1": 1, 0: 0, 1: 1}
    if combined_df["label"].dtype == "object":
        # First try to convert numeric strings directly
        combined_df["label"] = pd.to_numeric(combined_df["label"], errors="coerce")
        # If that fails, try text mapping for remaining NaN values
        mask = combined_df["label"].isna()
        if mask.any():
            text_labels = (
                combined_df.loc[mask, "label"]
                .astype(str)
                .str.lower()
                .map(label_mapping)
            )
            combined_df.loc[mask, "label"] = text_labels

    combined_df = combined_df.dropna(subset=["label"])
    combined_df["label"] = combined_df["label"].astype(int)

    if not all(combined_df["label"].isin([0, 1])):
        unique_labels = combined_df["label"].unique()
        raise ValueError(
            f"Invalid labels found: {unique_labels}. Expected 0 (legitimate) and 1 (malicious)"
        )

    print(f"\nTotal samples loaded: {len(combined_df)}")
    print("Label distribution:")
    print(f"  Legitimate (0): {(combined_df['label'] == 0).sum()}")
    print(f"  Malicious (1): {(combined_df['label'] == 1).sum()}")
    print(f"Sources: {combined_df['source'].unique()}")

    return combined_df


def compute_text_hash(text: str) -> str:
    """
    Compute SHA-256 hash of text for deduplication purposes.

    Parameters
    ----------
    text : str
        Input text to hash.

    Returns
    -------
    str
        Hexadecimal SHA-256 hash of the text.

    Notes
    -----
    Used by remove_duplicates() function to identify exact text duplicates
    across the dataset efficiently.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """
    Text normalization for transformer-based models and zero-shot classification.

    This function applies text preprocessing including HTML tag removal,
    URL/email/number replacement, case normalization, and whitespace cleanup.
    Designed for transformer models that use subword tokenization.

    Parameters
    ----------
    text : str
        Input text to normalize. Can be None or NaN.

    Returns
    -------
    str
        Normalized text with standardized tokens, or empty string if input is None/NaN.

    Notes
    -----
    Used by:
    - DistilBERT fine-tuning and evaluation
    - Zero-shot classification
    - Text similarity analysis

    The CNN model uses WordVocab with spaCy tokenization instead of this function.
    Different models use different preprocessing approaches optimized for their
    tokenization methods (subword vs word-level).

    Normalization steps:
    1. Unicode normalization (NFKC)
    2. Lowercasing
    3. HTML tag removal
    4. URL replacement with <URL>
    5. Email replacement with <EMAIL>
    6. Number replacement with <NUM>
    7. Whitespace normalization
    """
    if pd.isna(text) or text is None:
        return ""

    # Convert to string and Unicode normalize
    text = str(text).strip()
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Strip simple HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Replace common artifacts
    text = re.sub(r"https?://[^\s]+", "<URL>", text)  # URLs
    text = re.sub(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "<EMAIL>", text
    )  # Emails
    text = re.sub(r"\b\d+\b", "<NUM>", text)  # Numbers

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_duplicates(df: pd.DataFrame, exact: bool = True) -> pd.DataFrame:
    """
    Remove exact duplicate texts from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with 'text' column.
    exact : bool, default=True
        Whether to perform exact duplicate removal. If False, returns original DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed, keeping first occurrence.

    Notes
    -----
    Uses SHA-256 hashing for efficient duplicate detection. Prints the number
    of duplicates removed. Only affects exact text matches.
    """
    if exact:
        df["text_hash"] = df["text"].apply(compute_text_hash)
        df_dedup = df.drop_duplicates(subset=["text_hash"], keep="first")
        df_dedup = df_dedup.drop(columns=["text_hash"])
        print(f"Removed {len(df) - len(df_dedup)} exact duplicates")
        return df_dedup
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test sets using stratified random sampling.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with 'label' column for stratification.
    test_size : float, default=0.1
        Proportion of data to use for test set (0.0 to 1.0).
    val_size : float, default=0.1
        Proportion of data to use for validation set (0.0 to 1.0).
    random_state : int, default=42
        Random seed for reproducible splits.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train, validation, and test DataFrames in that order.

    Notes
    -----
    All splits maintain the original label distribution through stratification.
    """

    print("Using stratified random split")
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=random_state
    )

    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df["label"],
        random_state=random_state,  # noqa: E501
    )

    print(
        f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    return train_df, val_df, test_df
