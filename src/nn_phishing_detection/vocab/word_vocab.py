# ruff: noqa: E501
"""
Word-level vocabulary builder with GloVe embeddings for neural network models.

This module provides vocabulary building capabilities for word-level
neural networks, including spaCy tokenization, GloVe embedding integration,
tokenization caching, and text encoding/decoding for CNN-BiGRU models.

Author: David Schatz <schatz@cl.uni-heidelberg.de>
"""

import pickle
import subprocess
import sys
from collections import Counter

import numpy as np
import pandas as pd
import spacy
import torch
from tqdm import tqdm


class WordVocab:
    """
    Word-level vocabulary builder with GloVe embeddings and spaCy tokenization.

    Builds vocabularies from text corpora with advanced tokenization, frequency
    filtering, GloVe embedding integration, and caching mechanisms.

    Parameters
    ----------
    texts : pd.Series or list[str]
        Text samples for vocabulary building.
    max_vocab_size : int, optional
        Maximum vocabulary size. If None, includes all words above min_freq.
    min_freq : int, default=3
        Minimum word frequency threshold for inclusion.
    glove_path : str, optional
        Path to GloVe embeddings file (required).
    embedding_dim : int, default=100
        Expected embedding dimension (auto-detected from GloVe file).
    cache_tokenization : bool, default=True
        Whether to cache tokenized texts for performance.
    cache_file : str, optional
        Path to existing tokenization cache file.

    Attributes
    ----------
    vocab_size : int
        Final vocabulary size including special tokens.
    word2idx : dict
        Mapping from words to indices.
    idx2word : dict
        Mapping from indices to words.
    embeddings : torch.Tensor
        GloVe embedding matrix of shape (vocab_size, embedding_dim).

    Notes
    -----
    Uses spaCy for advanced tokenization with automatic model downloading.
    Supports special tokens for URLs, emails, and numbers.
    Requires GloVe embeddings file for proper initialization.
    """

    def __init__(
        self,
        texts: pd.Series | list[str],
        max_vocab_size: int | None = 30000,
        min_freq: int = 3,
        glove_path: str | None = None,
        embedding_dim: int = 100,
        cache_tokenization: bool = True,
        cache_file: str | None = None,
    ):
        """
        Initialize word vocabulary with GloVe embeddings.

        Parameters
        ----------
        texts : pd.Series or list[str]
            Text samples for building vocabulary.
        max_vocab_size : int, optional
            Maximum vocabulary size (None for unlimited).
        min_freq : int, default=3
            Minimum word frequency to include in vocabulary.
        glove_path : str, optional
            Path to GloVe embeddings file (required).
        embedding_dim : int, default=100
            Expected embedding dimension.
        cache_tokenization : bool, default=True
            Whether to cache tokenized texts.
        cache_file : str, optional
            Path to existing tokenization cache.

        Raises
        ------
        ValueError
            If GloVe embeddings path is not provided.
        """
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.embedding_dim = embedding_dim
        self.cache_tokenization = cache_tokenization

        # Special tokens
        self.unk_token = "<UNK>"

        # Initialize spaCy with auto-download
        self._setup_spacy()

        # Tokenization cache - load existing cache if provided
        self._tokenized_cache = {} if cache_tokenization else None
        if cache_tokenization and cache_file:
            self.load_tokenization_cache(cache_file)

        # Build vocabulary
        self._build_vocab(texts)

        # Load GloVe embeddings (required)
        if not glove_path:
            raise ValueError("GloVe embeddings path is required.")

        self.embeddings = self._load_glove(glove_path)

    def _setup_spacy(self):
        """
        Initialize spaCy tokenizer with automatic model downloading.

        Sets up spaCy English model with disabled components for efficiency.
        Automatically downloads the model if not found locally.

        Returns
        -------
        None
            Configures self.nlp with spaCy pipeline.

        Notes
        -----
        Disables parser, NER, lemmatizer, and tagger for faster tokenization.
        Downloads en_core_web_sm model automatically if missing.
        """
        model_name = "en_core_web_sm"

        try:
            # Try to load the model
            self.nlp = spacy.load(
                model_name, disable=["parser", "ner", "lemmatizer", "tagger"]
            )
            print(" Using spaCy tokenization")
        except OSError:
            # Model not found, download it
            print(f"Downloading spaCy model '{model_name}'...")
            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", model_name]
            )
            self.nlp = spacy.load(
                model_name, disable=["parser", "ner", "lemmatizer", "tagger"]
            )
            print(" spaCy model downloaded and loaded")

        print(" Using CPU tokenization")

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text using spaCy with intelligent preprocessing.

        Performs advanced tokenization with special token handling for URLs,
        emails, numbers, and punctuation. Includes caching for performance.

        Parameters
        ----------
        text : str
            Input text to tokenize.

        Returns
        -------
        list[str]
            List of tokens including special tokens and punctuation.

        Notes
        -----
        Handles special cases:
        - URLs → '<URL>'
        - Emails → '<EMAIL>'
        - Numbers → '<NUM>'
        - Preserves important punctuation
        - Ensures at least one token per input (uses '<UNK>' for empty)
        """
        # Check cache first if enabled
        if self._tokenized_cache is not None and text in self._tokenized_cache:
            return self._tokenized_cache[text]

        doc = self.nlp(text.lower())
        words = []

        for token in doc:
            if token.is_alpha:
                # All alphabetic words (including single chars like "I", "a")
                words.append(token.text)
            elif token.like_url:
                words.append("<URL>")
            elif token.like_email:
                words.append("<EMAIL>")
            elif token.like_num:
                words.append("<NUM>")
            elif token.text.isalnum():
                # All alphanumeric tokens
                words.append(token.text)
            elif token.text in {"!", "?", ".", ",", ";", ":", "-", "_"}:
                # Important punctuation that could be meaningful
                words.append(token.text)

        # Ensure we always return at least one token for any input text
        # This prevents zero-length sequences
        if not words:
            words = ["<UNK>"]

        # Cache the result if caching is enabled
        if self._tokenized_cache is not None:
            self._tokenized_cache[text] = words

        return words

    def _build_vocab(self, texts: pd.Series | list[str]):
        """
        Build vocabulary from text corpus with frequency filtering.

        Tokenizes all texts, counts word frequencies, applies filtering criteria,
        and creates word-to-index mappings for neural network processing.

        Parameters
        ----------
        texts : pd.Series or list[str]
            Collection of text samples for vocabulary building.

        Returns
        -------
        None
            Updates instance attributes: word2idx, idx2word, vocab_size.

        Notes
        -----
        Filters words by minimum frequency and limits total vocabulary size.
        Reserves index 0 for '<UNK>' token. Sorts words by frequency.
        Includes validation to ensure index consistency.
        """
        print("Building word vocabulary...")

        # Count word frequencies
        word_counter = Counter()

        for text in tqdm(texts, desc="Tokenizing texts"):
            if pd.isna(text):
                continue
            words = self._tokenize(str(text))
            word_counter.update(words)

        # Filter by frequency and exclude special tokens from vocabulary
        filtered_words = [
            word
            for word, count in word_counter.items()
            if count >= self.min_freq and word != self.unk_token
        ]

        # Sort by frequency
        filtered_words = sorted(
            filtered_words, key=lambda w: word_counter[w], reverse=True
        )

        # Limit vocabulary size
        if self.max_vocab_size:
            filtered_words = filtered_words[
                : self.max_vocab_size - 1
            ]  # Reserve for UNK token

        # Create word to index mapping
        self.word2idx = {self.unk_token: 0}
        for idx, word in enumerate(filtered_words, start=1):
            self.word2idx[word] = idx

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Most common words: {filtered_words[:10]}")

        # Sanity check
        expected_max_idx = self.vocab_size - 1
        actual_max_idx = max(self.word2idx.values()) if self.word2idx else 0
        if actual_max_idx != expected_max_idx:
            raise ValueError(
                f"Vocabulary index mismatch: max_idx={actual_max_idx}, expected={expected_max_idx}. "
                f"This indicates a bug in vocabulary building."
            )

    def _load_glove(self, glove_path: str) -> torch.Tensor:
        """
        Load and integrate GloVe word embeddings.

        Reads GloVe embeddings file, auto-detects dimensions, and creates
        embedding matrix aligned with vocabulary indices.

        Parameters
        ----------
        glove_path : str
            Path to GloVe embeddings text file.

        Returns
        -------
        torch.Tensor
            Embedding matrix of shape (vocab_size, embedding_dim).

        Notes
        -----
        Auto-detects embedding dimension from file format.
        Reports coverage statistics for vocabulary overlap with GloVe.
        """
        print(f"Loading GloVe embeddings from {glove_path}...")

        # First pass: detect embedding dimension from file
        detected_dim = None
        with open(glove_path, encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line:
                values = first_line.split()
                detected_dim = len(values) - 1  # Minus 1 for the word itself
                print(f"Detected GloVe dimension: {detected_dim}")

        # Use detected dimension if different from expected
        if detected_dim and detected_dim != self.embedding_dim:
            print(
                f"Adjusting embedding dimension from {self.embedding_dim} to {detected_dim}"
            )
            self.embedding_dim = detected_dim

        # Initialize embeddings matrix
        print(
            f"Creating embeddings matrix: shape ({self.vocab_size}, {self.embedding_dim})"
        )
        embeddings = np.random.randn(self.vocab_size, self.embedding_dim) * 0.1

        # Load GloVe
        glove_dict = {}
        try:
            with open(glove_path, encoding="utf-8") as f:
                for line in tqdm(f, desc="Loading GloVe"):
                    values = line.split()
                    word = values[0]
                    if word in self.word2idx:
                        try:
                            vector = np.array(values[1:], dtype="float32")
                            if len(vector) == self.embedding_dim:
                                glove_dict[word] = vector
                        except ValueError:
                            # Skip entries that can't be converted to float (e.g., malformed lines)
                            continue
        except FileNotFoundError:
            print(
                f"Warning: GloVe file not found at {glove_path}. Using random initialization."
            )
            return torch.tensor(embeddings, dtype=torch.float32)

        # Fill embeddings with GloVe vectors
        found = 0
        for word, idx in self.word2idx.items():
            if word in glove_dict:
                embeddings[idx] = glove_dict[word]
                found += 1

        print(
            f"Found {found}/{self.vocab_size} words in GloVe ({found/self.vocab_size*100:.1f}%)"
        )

        return torch.tensor(embeddings, dtype=torch.float32)

    def save_tokenization_cache(self, cache_path: str):
        """
        Save tokenization cache to disk for performance optimization.

        Persists the tokenization cache dictionary to avoid re-tokenizing
        the same texts in future runs.

        Parameters
        ----------
        cache_path : str
            Output path for cache pickle file.

        Returns
        -------
        None
            Saves cache to disk and prints summary.

        Notes
        -----
        Only saves if caching is enabled. Uses pickle format for efficiency.
        """
        if self._tokenized_cache is None:
            print("No tokenization cache to save (caching disabled)")
            return

        with open(cache_path, "wb") as f:
            pickle.dump(self._tokenized_cache, f)
        print(
            f"Saved {len(self._tokenized_cache)} cached tokenizations to {cache_path}"
        )

    def load_tokenization_cache(self, cache_path: str):
        """
        Load tokenization cache from disk to avoid re-tokenization.

        Loads previously saved tokenization results and cleans up any
        empty tokenizations from older cache versions.

        Parameters
        ----------
        cache_path : str
            Path to cached tokenization pickle file.

        Returns
        -------
        None
            Updates self._tokenized_cache with loaded data.

        Notes
        -----
        Automatically fixes empty tokenizations with '<UNK>' tokens.
        Handles missing files and corruption gracefully.
        """
        if not self.cache_tokenization:
            print("Tokenization caching is disabled")
            return

        try:
            with open(cache_path, "rb") as f:
                loaded_cache = pickle.load(f)

            # Clean up any empty tokenizations from old cache files
            self._tokenized_cache = {}
            cleaned_count = 0
            for text, tokens in loaded_cache.items():
                if not tokens:  # Empty list
                    tokens = ["<UNK>"]
                    cleaned_count += 1
                self._tokenized_cache[text] = tokens

            print(
                f"Loaded {len(self._tokenized_cache)} cached tokenizations from {cache_path}"
            )
            if cleaned_count > 0:
                print(f"Fixed {cleaned_count} empty tokenizations with <UNK> tokens")
        except FileNotFoundError:
            print(f"Cache file not found: {cache_path}")
            self._tokenized_cache = {}
        except Exception as e:
            print(f"Error loading cache: {e}")
            self._tokenized_cache = {}

    def encode(self, text: str, max_len: int) -> tuple[torch.Tensor, int]:
        """
        Encode text to numerical indices for neural network input.

        Tokenizes text and converts words to vocabulary indices with length
        truncation and safety checks for empty sequences.

        Parameters
        ----------
        text : str
            Input text to encode.
        max_len : int
            Maximum sequence length (truncates if longer).

        Returns
        -------
        tuple[torch.Tensor, int]
            Tuple containing:
            - Encoded indices tensor of shape (actual_length,)
            - Actual sequence length (at least 1)

        Notes
        -----
        Maps unknown words to '<UNK>' token index.
        Ensures minimum length of 1 to prevent pack_padded_sequence errors.
        Truncates sequences longer than max_len.
        """
        words = self._tokenize(text)

        # Ensure we have at least one token
        if not words:
            words = ["<UNK>"]

        # Convert to indices
        indices = []
        for word in words[:max_len]:
            indices.append(self.word2idx.get(word, self.word2idx[self.unk_token]))

        actual_len = len(indices)

        # Ensure actual_len is at least 1 to avoid pack_padded_sequence errors
        if actual_len == 0:
            indices = [self.word2idx[self.unk_token]]
            actual_len = 1

        return torch.tensor(indices, dtype=torch.long), actual_len

    def decode(self, indices: torch.Tensor) -> str:
        """
        Decode numerical indices back to human-readable text.

        Converts vocabulary indices to words and joins them into text.
        Used for model interpretation and debugging.

        Parameters
        ----------
        indices : torch.Tensor
            Tensor of vocabulary indices to decode.

        Returns
        -------
        str
            Decoded text with words separated by spaces.

        Notes
        -----
        Maps unknown indices to '<UNK>' token.
        Preserves special tokens in the output text.
        """
        words = []
        for idx in indices:
            words.append(self.idx2word.get(idx.item(), self.unk_token))
        return " ".join(words)
