"""
Zero-shot phishing detection using large language models.

This module provides zero-shot classification capabilities for phishing detection
using OpenAI and Mistral APIs. Supports both sequential and batch processing with
cost tracking, rate limiting, and performance evaluation.

Author: Elizaveta Dovedova <dovedova@cl.uni-heidelberg.de>, David Schatz <schatz@cl.uni-heidelberg.de>
"""

# ruff: noqa: E501
import argparse
import builtins
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    MISTRAL_TOKENIZER_AVAILABLE = True
except ImportError:
    MISTRAL_TOKENIZER_AVAILABLE = False

try:
    from mistralai import Mistral

    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

# Add project root to path for imports
sys.path.append("src")
from nn_phishing_detection.data_utils import normalize_text

LABELS = ("legitimate", "malicious")  # 0 = legitimate, 1 = malicious


class RateLimiter:
    """
    Token bucket rate limiter for API request management.

    Implements a token bucket algorithm to ensure API requests
    do not exceed rate limits by controlling request frequency.

    Parameters
    ----------
    requests_per_minute : int
        Maximum number of requests allowed per minute.

    Notes
    -----
    Uses time.monotonic() for precise timing and automatically refills
    tokens based on elapsed time. Blocks when no tokens are available.
    """

    def __init__(self, requests_per_minute: int):
        self.capacity = max(1, requests_per_minute)
        self.tokens = self.capacity
        self.refill_time = 60.0  # one minute
        self.last = time.monotonic()

    def acquire(self):
        """
        Acquire a token for making an API request.

        Blocks execution until at least one token is available in the bucket.
        Automatically refills tokens based on elapsed time.

        Returns
        -------
        None
            Returns when a token has been acquired and consumed.

        Notes
        -----
        Uses a polling approach with 0.1-second sleep intervals when
        no tokens are available. Refills tokens proportionally to elapsed time.
        """
        while True:
            now = time.monotonic()
            elapsed = now - self.last
            # Refill tokens based on elapsed time
            refill = (elapsed / self.refill_time) * self.capacity
            if refill >= 1:
                self.tokens = min(self.capacity, self.tokens + int(refill))
                self.last = now
            if self.tokens >= 1:
                self.tokens -= 1
                return
            # No tokens available: wait a little before checking again
            time.sleep(0.1)


def parse_label(text: str) -> int | None:
    """
    Parse LLM output text to extract binary classification label.

    Extracts and validates the classification result from model responses,
    handling case-insensitive text parsing with strict validation.

    Parameters
    ----------
    text : str
        Raw text response from the language model.

    Returns
    -------
    int or None
        Classification label:
        - 0 for legitimate emails
        - 1 for malicious emails
        - None for invalid/unparseable responses

    Notes
    -----
    Only accepts exact matches for 'legitimate' or 'malicious' after
    stripping whitespace and converting to lowercase. Rejects any
    other format to ensure classification.
    """
    if not text:
        return None
    t = text.strip().lower()
    if t == "malicious":
        return 1
    if t == "legitimate":
        return 0
    return None


class ZSClassifier:
    """
    Zero-shot phishing email classifier using large language models.

    Provides zero-shot classification capabilities for phishing detection using
    OpenAI and Mistral APIs with cost tracking, rate limiting,
    and error handling. Supports both sequential and batch processing modes.

    Parameters
    ----------
    api_key : str
        API key for the selected provider (OpenAI or Mistral).
    model : str, default="gpt-4o-mini"
        Model name for classification. Examples: "gpt-4o-mini", "mistral-medium-latest".
    provider : str, default="openai"
        API provider ("openai" or "mistral").
    requests_per_minute : int, default=500
        Rate limit for API requests per minute.
    max_retries : int, default=5
        Maximum number of retry attempts for failed requests.
    base_backoff : float, default=1.0
        Initial backoff time in seconds for exponential backoff.
    backoff_cap : float, default=30.0
        Maximum backoff time in seconds.
    timeout : float, optional
        Request timeout in seconds. If None, uses provider defaults.
    track_metrics : bool, default=True
        Whether to track costs, token usage, and performance metrics.
    normalize_text : bool, default=True
        Whether to apply text normalization before classification.

    Attributes
    ----------
    metrics : dict
        Metrics including request counts, token usage, costs,
        response times, and error breakdown.
    model_costs : dict
        Pricing information per model for cost estimation.

    Notes
    -----
    Requires appropriate API keys and package installations:
    - OpenAI: requires 'openai' and 'tiktoken' packages
    - Mistral: requires 'mistralai' and 'mistral-common' packages

    Implements exponential backoff for error handling.
    Supports both text normalization and raw text processing modes.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        provider: str = "openai",  # "openai" or "mistral"
        requests_per_minute: int = 500,  # GPT-4o turbo supports 500 RPM
        max_retries: int = 5,
        base_backoff: float = 1.0,  # initial backoff in seconds
        backoff_cap: float = 30.0,  # maximum backoff time
        timeout: float | None = None,
        track_metrics: bool = True,  # Track costs and metrics
        normalize_text: bool = True,  # Apply text normalization
    ):
        self.provider = provider.lower()
        self.model = model
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.backoff_cap = backoff_cap
        self.timeout = timeout
        self.normalize_text = normalize_text
        self.ratelimiter = RateLimiter(requests_per_minute=requests_per_minute)

        # Initialize the appropriate client
        if self.provider == "openai":
            self.client = OpenAI(api_key=api_key)
        elif self.provider == "mistral":
            if not MISTRAL_AVAILABLE:
                raise ImportError(
                    "mistralai package not installed. Run: pip install mistralai"
                )
            self.client = Mistral(api_key=api_key)
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Use 'openai' or 'mistral'"
            )

        # Cost tracking (pricing per million tokens)
        self.track_metrics = track_metrics
        self.model_costs = {
            # OpenAI models
            "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
            # Mistral models - sequential pricing
            "mistral-medium-latest": {
                "input": 0.4 / 1_000_000,
                "output": 2.0 / 1_000_000,
            },  # $0.40/$2.00 per 1M
        }

        # Metrics tracking
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "response_times": [],
            "error_types": {
                "timeout": 0,
                "rate_limit": 0,
                "parse_error": 0,
                "other": 0,
            },
        }

    def estimate_cost(self, texts: list[str]) -> dict:
        """
        Estimate API costs for text classification using provider-specific tokenizers.

        Calculates approximate costs by tokenizing input texts and estimating
        output tokens for the classification task using official tokenizers.

        Parameters
        ----------
        texts : list[str]
            List of email texts to estimate costs for.

        Returns
        -------
        dict
            Dictionary containing:
            - input_tokens (int): Total estimated input tokens
            - output_tokens (int): Total estimated output tokens
            - estimated_cost (float): Total estimated cost in USD
            - cost_per_request (float): Average cost per request
            - error (str, optional): Error message if tokenization fails

        Notes
        -----
        Uses tiktoken for OpenAI models and mistral-common for Mistral models.
        Output tokens are estimated as 1 per request (single word responses).
        Includes system message and prompt formatting in token calculations.
        """
        total_input_tokens = 0
        total_output_tokens = len(
            texts
        )  # Output is always 1 token (malicious/legitimate)

        if self.provider == "openai":
            if not TIKTOKEN_AVAILABLE:
                return {"error": "tiktoken not installed. Run: pip install tiktoken"}

            # Use tiktoken for OpenAI models
            try:
                encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")

            system_msg = "You are a helpful email classifier. Reply with exactly one word: 'malicious' or 'legitimate'."
            for text in texts:
                messages_text = system_msg + self._prompt(text)
                total_input_tokens += len(encoding.encode(messages_text))

        elif self.provider == "mistral":
            if not MISTRAL_TOKENIZER_AVAILABLE:
                return {
                    "error": "mistral-common not installed. Run: pip install mistral-common tiktoken"
                }

            # Use Mistral's official tokenizer
            try:
                # Use v3 tekken for newer models, v3 for older ones
                if "nemo" in self.model.lower() or "ministral" in self.model.lower():
                    tokenizer = MistralTokenizer.v3(is_tekken=True)
                else:
                    tokenizer = MistralTokenizer.v3()

                # Tokenize each conversation
                for text in texts:
                    tokenized = tokenizer.encode_chat_completion(
                        ChatCompletionRequest(
                            messages=[UserMessage(content=self._prompt(text))],
                            model=self.model,
                        )
                    )
                    total_input_tokens += len(tokenized.tokens)

            except Exception as e:
                return {"error": f"Mistral tokenization failed: {e}"}

        else:
            return {
                "error": f"Cost estimation not supported for provider: {self.provider}"
            }

        if self.model in self.model_costs:
            input_cost = total_input_tokens * self.model_costs[self.model]["input"]
            output_cost = total_output_tokens * self.model_costs[self.model]["output"]
            total_cost = input_cost + output_cost

            return {
                "num_texts": len(texts),
                "actual_input_tokens": total_input_tokens,
                "actual_output_tokens": total_output_tokens,
                "estimated_cost": round(total_cost, 4),
                "model": self.model,
            }
        return {"error": f"Cost data not available for model {self.model}"}

    def get_metrics_summary(self) -> dict:
        """
        Get summary of API usage and performance metrics.

        Returns aggregated statistics including request counts, token usage,
        costs, success rates, and error breakdowns for analysis.

        Returns
        -------
        dict
            Dictionary containing:
            - total_requests (int): Total API requests made
            - successful_requests (int): Number of successful requests
            - failed_requests (int): Number of failed requests
            - success_rate (float): Success rate as percentage
            - total_input_tokens (int): Total input tokens consumed
            - total_output_tokens (int): Total output tokens generated
            - total_cost (float): Total cost in USD
            - avg_response_time (float): Average response time in seconds
            - error_breakdown (dict): Counts by error type

        Notes
        -----
        Only available when track_metrics=True during initialization.
        Costs are calculated based on current model pricing in model_costs.
        """
        if not self.track_metrics:
            return {"error": "Metrics tracking disabled"}

        avg_response_time = (
            np.mean(self.metrics["response_times"])
            if self.metrics["response_times"]
            else 0
        )
        success_rate = (
            self.metrics["successful_requests"] / max(1, self.metrics["total_requests"])
        ) * 100

        return {
            "total_requests": self.metrics["total_requests"],
            "success_rate": round(success_rate, 2),
            "failed_requests": self.metrics["failed_requests"],
            "total_input_tokens": self.metrics["total_input_tokens"],
            "total_output_tokens": self.metrics["total_output_tokens"],
            "total_cost": round(self.metrics["total_cost"], 4),
            "avg_response_time": round(avg_response_time, 3),
            "error_breakdown": self.metrics["error_types"],
            "cost_per_request": round(
                self.metrics["total_cost"] / max(1, self.metrics["total_requests"]), 6
            ),
        }

    def _prompt(self, text: str) -> str:
        """
        Generate classification prompt for email text.

        Creates a structured prompt for zero-shot phishing classification
        with optional text normalization and clear task instructions.

        Parameters
        ----------
        text : str
            Email text to classify.

        Returns
        -------
        str
            Formatted prompt with task instructions and email content.

        Notes
        -----
        Applies text normalization if self.normalize_text is True.
        Uses specific instructions to encourage single-word responses.
        """
        # Apply normalization if enabled, otherwise use raw text
        processed_text = normalize_text(text) if self.normalize_text else text

        return (
            "You are an email classifier.\n"
            "Classify the following email as either 'malicious' (harmful phishing message) "
            "or 'legitimate' (not malicious, wanted email).\n"
            f"Email text: {processed_text}\n"
            "Respond with only one word: either 'malicious' or 'legitimate'."
        )

    def _classify_once(self, text: str) -> tuple[int | None, float]:
        """
        Perform single classification request with provider-specific handling.

        Makes a single API request to classify one email text, handling
        provider differences in request format and response parsing.

        Parameters
        ----------
        text : str
            Email text to classify.

        Returns
        -------
        tuple[int | None, float]
            Tuple containing:
            - Classification label (0=legitimate, 1=malicious, None=failed)
            - Confidence score (provider-dependent, may be 0.0)

        Notes
        -----
        Handles OpenAI and Mistral API differences in request formatting.
        Updates internal metrics with token usage and response times.
        Confidence scores vary by provider (OpenAI: logprobs, Mistral: none).
        """
        start_time = time.time()
        self.ratelimiter.acquire()

        # Create API call with provider-specific parameters
        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful email classifier. Reply with exactly one word: 'malicious' or 'legitimate'.",
                    },
                    {"role": "user", "content": self._prompt(text)},
                ],
                temperature=0,
                max_tokens=2,  # allow for "malicious" or "legitimate"
                stop=["\n"],  # stop at newline
                timeout=self.timeout,
                logprobs=True,  # Enable log probabilities
                top_logprobs=2,  # Get top 2 alternative tokens
            )
        elif self.provider == "mistral":
            resp = self.client.chat.complete(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful email classifier. Reply with exactly one word: 'malicious' or 'legitimate'.",
                    },
                    {"role": "user", "content": self._prompt(text)},
                ],
                temperature=0,
                max_tokens=2,  # allow for "malicious" or "legitimate"
                stop=["\n"],  # stop at newline
            )

        # Track response time
        response_time = time.time() - start_time
        if self.track_metrics:
            self.metrics["response_times"].append(response_time)
            self.metrics["total_requests"] += 1

        raw = resp.choices[0].message.content
        label = parse_label(raw)

        # Track success/failure
        if self.track_metrics:
            if label is not None:
                self.metrics["successful_requests"] += 1
            else:
                self.metrics["failed_requests"] += 1
                self.metrics["error_types"]["parse_error"] += 1

        # Track token usage and costs
        if self.track_metrics and hasattr(resp, "usage"):
            input_tokens = resp.usage.prompt_tokens
            output_tokens = resp.usage.completion_tokens
            self.metrics["total_input_tokens"] += input_tokens
            self.metrics["total_output_tokens"] += output_tokens

            # Calculate cost
            if self.model in self.model_costs:
                cost = (
                    input_tokens * self.model_costs[self.model]["input"]
                    + output_tokens * self.model_costs[self.model]["output"]
                )
                self.metrics["total_cost"] += cost

        # Extract confidence from logprobs (OpenAI only)
        confidence = 0.0
        if (
            self.provider == "openai"
            and resp.choices[0].logprobs
            and resp.choices[0].logprobs.content
        ):
            # Get probability of first token
            logprob = resp.choices[0].logprobs.content[0].logprob
            confidence = np.exp(logprob)  # Convert log prob to probability
        # Mistral sequential API doesn't provide confidence scores

        return label, confidence

    def _update_batch_metrics(self, batch_results_path: str):
        """
        Extract token usage from batch API results and update internal metrics.

        Processes batch results file to extract token consumption, cost information,
        and success rates for both OpenAI and Mistral providers. Updates the
        internal metrics dictionary with aggregated statistics.

        Parameters
        ----------
        batch_results_path : str
            Path to the batch results file containing API responses in JSONL format.

        Returns
        -------
        None
            Updates self.metrics dictionary in place.

        Notes
        -----
        Handles provider-specific response formats:
        - OpenAI: Uses 'usage' field for token counts
        - Mistral: Uses 'usage' field for token counts

        Processes both successful and failed requests to maintain
        success/failure ratios. Calculates costs based on current model pricing.
        Only processes files if self.track_metrics is True.
        """
        if not self.track_metrics:
            return

        total_input_tokens = 0
        total_output_tokens = 0
        total_requests = 0
        successful_requests = 0

        with open(batch_results_path) as f:
            for line in f:
                result = json.loads(line)
                total_requests += 1

                if result.get("error"):
                    continue

                successful_requests += 1
                response_body = result["response"]["body"]

                # Extract token usage
                if "usage" in response_body:
                    usage = response_body["usage"]
                    total_input_tokens += usage.get("prompt_tokens", 0)
                    total_output_tokens += usage.get("completion_tokens", 0)

        # Update metrics
        self.metrics["total_requests"] = total_requests
        self.metrics["successful_requests"] = successful_requests
        self.metrics["failed_requests"] = total_requests - successful_requests
        self.metrics["total_input_tokens"] = total_input_tokens
        self.metrics["total_output_tokens"] = total_output_tokens

        # Calculate cost with 50% batch discount
        if self.model in self.model_costs:
            batch_discount = 0.5
            total_cost = (
                total_input_tokens
                * self.model_costs[self.model]["input"]
                * batch_discount
                + total_output_tokens
                * self.model_costs[self.model]["output"]
                * batch_discount
            )
            self.metrics["total_cost"] = total_cost

        print(f"Batch results: {successful_requests}/{total_requests} successful")
        print(
            f"Token usage: {total_input_tokens:,} input + {total_output_tokens:,} output = {total_input_tokens + total_output_tokens:,} total"
        )
        if (
            self.model in self.model_costs
            and total_input_tokens + total_output_tokens > 0
        ):
            print(f"Total cost: ${self.metrics['total_cost']:.4f}")

    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute function with exponential backoff retry logic.

        Implements robust retry mechanism with exponential backoff
        for handling API errors and rate limiting.

        Parameters
        ----------
        func : callable
            Function to execute with retry logic.
        *args : tuple
            Positional arguments to pass to func.
        **kwargs : dict
            Keyword arguments to pass to func.

        Returns
        -------
        Any
            Return value from successful function execution.

        Raises
        ------
        Exception
            Re-raises the last exception if all retries are exhausted.

        Notes
        -----
        Tracks error types in metrics for debugging and analysis.
        Respects max_retries, base_backoff, and backoff_cap settings.
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                retry_after = None
                # Check if server returned Retry-After
                try:
                    if hasattr(e, "response") and e.response is not None:
                        ra = e.response.headers.get("Retry-After")
                        if ra:
                            retry_after = float(ra)
                except Exception:
                    pass

                if retry_after:
                    sleep_s = retry_after
                else:
                    # Exponential backoff with jitter
                    base = self.base_backoff * (2**attempt)
                    sleep_s = min(self.backoff_cap, base)
                    sleep_s = np.random.uniform(0, sleep_s)
                time.sleep(sleep_s)
        raise last_error

    def classify(self, text: str) -> tuple[int | None, float]:
        """
        Classify a single email text with rate limiting and error handling.

        High-level interface for single email classification with automatic
        rate limiting, retry logic, and metrics tracking.

        Parameters
        ----------
        text : str
            Email text to classify.

        Returns
        -------
        tuple[int | None, float]
            Tuple containing:
            - Classification label (0=legitimate, 1=malicious, None=failed)
            - Confidence score (provider-dependent)

        Notes
        -----
        Applies rate limiting before making requests.
        Uses _retry_with_backoff for error handling.
        """
        try:
            return self._retry_with_backoff(self._classify_once, text)
        except Exception as e:
            print(f"Final failure: {repr(e)}")
            return None, 0.0

    def classify_batch(
        self, texts: list[str], batch_size: int = 20
    ) -> tuple[list[int | None], list[float]]:
        """
        Classify multiple emails using batch API processing.

        Processes large batches of emails efficiently using provider batch APIs
        with automatic file management and result processing.

        Parameters
        ----------
        texts : list[str]
            List of email texts to classify.
        batch_size : int, default=20
            Number of requests per batch file.

        Returns
        -------
        tuple[list[int | None], list[float]]
            Tuple containing:
            - List of classification labels (0=legitimate, 1=malicious, None=failed)
            - List of processing times per request

        Notes
        -----
        Uses provider-specific batch APIs for cost-effective processing.
        Automatically handles file upload, processing, and result retrieval.
        Updates internal metrics with token usage and costs.
        """
        preds: list[int | None] = []
        confs: list[float] = []

        # Simple progress bar for sequential processing
        for t in tqdm(texts, desc="Classifying emails"):
            label, confidence = self.classify(str(t))
            preds.append(label)
            confs.append(confidence)
        return preds, confs

    def create_batch_file(
        self, texts: list[str], output_path: str = "batch_input.jsonl"
    ) -> str:
        """
        Create batch input file in provider-specific JSONL format.

        Generates properly formatted batch request file for efficient
        bulk processing using provider batch APIs.

        Parameters
        ----------
        texts : list[str]
            List of email texts to include in batch.
        output_path : str, default="batch_input.jsonl"
            Path for the output batch file.

        Returns
        -------
        str
            Path to the created batch input file.

        Notes
        -----
        Creates provider-specific request formats:
        - OpenAI: Chat completions format with custom_id
        - Mistral: Chat completions format with custom_id

        Each request includes the classification prompt and unique identifier.
        """
        # Count tokens that will be sent
        if TIKTOKEN_AVAILABLE and self.provider == "openai":
            try:
                encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")

            total_input_tokens = 0
            system_msg = "You are a helpful email classifier. Reply with exactly one word: 'malicious' or 'legitimate'."

            for text in texts:
                messages_text = system_msg + self._prompt(text)
                total_input_tokens += len(encoding.encode(messages_text))

            total_output_tokens = len(texts)

            print(f"\n=== {self.provider.title()} Batch Token Analysis ===")
            print(f"Total requests: {len(texts)}")
            print(f"Input tokens to send: {total_input_tokens:,}")
            print(f"Expected output tokens: {total_output_tokens:,}")
            print(f"Total tokens: {total_input_tokens + total_output_tokens:,}")

        with open(output_path, "w") as f:
            for i, text in enumerate(texts):
                if self.provider == "openai":
                    # OpenAI batch format
                    request = {
                        "custom_id": f"request-{i}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.model,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a helpful email classifier. Reply with exactly one word: 'malicious' or 'legitimate'.",
                                },
                                {"role": "user", "content": self._prompt(text)},
                            ],
                            "temperature": 0,
                            "max_tokens": 2,
                            "stop": ["\n"],
                            "logprobs": True,
                            "top_logprobs": 2,
                        },
                    }
                elif self.provider == "mistral":
                    request = {
                        "custom_id": f"{i}",
                        "body": {
                            "model": self.model,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a helpful email classifier. Reply with exactly one word: 'malicious' or 'legitimate'.",
                                },
                                {"role": "user", "content": self._prompt(text)},
                            ],
                            "temperature": 0,
                            "max_tokens": 2,
                        },
                    }

                f.write(json.dumps(request) + "\n")

        print(
            f"Created {self.provider} batch file: {output_path} with {len(texts)} requests"
        )
        return output_path

    def submit_batch(self, batch_file_path: str, metadata: dict = None) -> str:
        """
        Submit batch file to provider API for processing.

        Uploads batch file and initiates processing job with the selected
        provider's batch API endpoint.

        Parameters
        ----------
        batch_file_path : str
            Path to the batch input file.
        metadata : dict, optional
            Additional metadata to attach to the batch job.

        Returns
        -------
        str
            Batch job ID for status checking and result retrieval.

        Notes
        -----
        Handles provider-specific batch submission formats.
        Job completion time varies by provider and batch size.
        Job ID is required for subsequent status checks and result retrieval.
        """
        if self.provider == "openai":
            # Upload file
            with open(batch_file_path, "rb") as f:
                batch_input_file = self.client.files.create(file=f, purpose="batch")

            # Create batch
            batch = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata=metadata or {},
            )

            print(f"OpenAI Batch submitted: {batch.id}")
            print(f"Status: {batch.status}")
            return batch.id

        elif self.provider == "mistral":
            # Upload file to Mistral with batch purpose
            with open(batch_file_path, "rb") as f:
                batch_input_file = self.client.files.upload(
                    file={"file_name": batch_file_path, "content": f}, purpose="batch"
                )

            # Create Mistral batch job
            batch = self.client.batch.jobs.create(
                input_files=[batch_input_file.id],
                model=self.model,
                endpoint="/v1/chat/completions",
                metadata=metadata or {},
            )

            print(f"Mistral Batch submitted: {batch.id}")
            print(f"Status: {batch.status}")
            return batch.id

    def check_batch_status(self, batch_id: str) -> dict:
        """
        Check status of submitted batch processing job.

        Queries provider API for current job status and progress information.

        Parameters
        ----------
        batch_id : str
            Batch job ID returned from submit_batch().

        Returns
        -------
        dict
            Status information containing:
            - status (str): Job status (e.g., 'in_progress', 'completed', 'failed')
            - Additional provider-specific status fields

        Notes
        -----
        Common status values include 'validating', 'in_progress', 'completed'.
        """
        if self.provider == "openai":
            batch = self.client.batches.retrieve(batch_id)
            return {
                "id": batch.id,
                "status": batch.status,
                "created_at": batch.created_at,
                "completed_at": batch.completed_at,
                "request_counts": {
                    "total": batch.request_counts.total,
                    "completed": batch.request_counts.completed,
                    "failed": batch.request_counts.failed,
                },
            }
        elif self.provider == "mistral":
            batch = self.client.batch.jobs.get(job_id=batch_id)
            return {
                "id": batch.id,
                "status": batch.status,
                "created_at": batch.created_at,
                "completed_at": getattr(batch, "completed_at", None),
                "request_counts": {
                    "total": getattr(batch, "total_requests", 0),
                    "completed": getattr(batch, "completed_requests", 0),
                    "failed": getattr(batch, "failed_requests", 0),
                },
            }

    def retrieve_batch_results(
        self, batch_id: str, output_path: str = "batch_output.jsonl"
    ) -> tuple[list[int | None], list[float]]:
        """
        Retrieve and process completed batch job results.

        Downloads batch results file and processes responses to extract
        classification labels and timing information.

        Parameters
        ----------
        batch_id : str
            Completed batch job ID.
        output_path : str, default="batch_output.jsonl"
            Local path to save results file.

        Returns
        -------
        tuple[list[int | None], list[float]]
            Tuple containing:
            - List of classification labels (0=legitimate, 1=malicious, None=failed)
            - List of processing times (zeros for batch processing)

        Notes
        -----
        Updates internal metrics with token usage and costs from batch results.
        Failed requests return None in the predictions list.
        Processing times are not available for batch requests.
        """
        if self.provider == "openai":
            batch = self.client.batches.retrieve(batch_id)

            if batch.status != "completed":
                print(f"OpenAI Batch not completed. Status: {batch.status}")
                return [], []

            # Download results
            file_response = self.client.files.content(batch.output_file_id)

            with open(output_path, "w") as f:
                f.write(file_response.text)

            # Parse OpenAI results
            results = {}
            confidences = {}

            with open(output_path) as f:
                for line in f:
                    result = json.loads(line)
                    custom_id = result["custom_id"]
                    request_idx = int(custom_id.split("-")[1])

                    if result["error"]:
                        results[request_idx] = None
                        confidences[request_idx] = 0.0
                    else:
                        response_body = result["response"]["body"]
                        response_text = response_body["choices"][0]["message"][
                            "content"
                        ]
                        results[request_idx] = parse_label(response_text)

                        # Extract confidence from logprobs
                        confidence = 0.0
                        logprobs = response_body["choices"][0].get("logprobs")
                        if logprobs and logprobs.get("content"):
                            logprob = logprobs["content"][0]["logprob"]
                            confidence = np.exp(logprob)
                        confidences[request_idx] = confidence

        elif self.provider == "mistral":
            batch = self.client.batch.jobs.get(job_id=batch_id)

            if batch.status != "SUCCESS":
                print(f"Mistral Batch not completed. Status: {batch.status}")
                return [], []

            # Download Mistral results
            output_file_stream = self.client.files.download(file_id=batch.output_file)

            with open(output_path, "wb") as f:
                f.write(output_file_stream.read())

            # Parse Mistral results
            results = {}
            confidences = {}

            with open(output_path) as f:
                for line in f:
                    result = json.loads(line)
                    custom_id = result["custom_id"]
                    request_idx = int(custom_id)

                    if result.get("error"):
                        results[request_idx] = None
                        confidences[request_idx] = 0.0
                    else:
                        response_body = result["response"]["body"]
                        response_text = response_body["choices"][0]["message"][
                            "content"
                        ]
                        results[request_idx] = parse_label(response_text)
                        confidences[request_idx] = 0.0

        # Extract token usage and update metrics from batch results
        self._update_batch_metrics(output_path)

        # Convert to ordered lists
        predictions = [results.get(i) for i in range(len(results))]
        conf_list = [confidences.get(i, 0.0) for i in range(len(results))]
        print(f"Retrieved {len(predictions)} {self.provider} batch results")
        return predictions, conf_list

    def classify_batch_async(
        self,
        texts: list[str],
        wait_for_completion: bool = False,
        poll_interval: int = 180,
    ) -> str:
        """
        Start asynchronous batch classification job.

        Initiates batch processing job and optionally waits for completion
        with automatic polling and status updates.

        Parameters
        ----------
        texts : list[str]
            List of email texts to classify.
        wait_for_completion : bool, default=False
            Whether to wait for job completion before returning.
        poll_interval : int, default=180
            Polling interval in seconds for job status checks.

        Returns
        -------
        str
            Batch job ID for later result retrieval.

        Notes
        -----
        Creates temporary batch files and uploads them to provider.
        If wait_for_completion=True, polls job status until finished.
        Returns job ID immediately if wait_for_completion=False.
        """
        # Create and submit batch
        batch_file = self.create_batch_file(texts)
        batch_id = self.submit_batch(batch_file)

        if not wait_for_completion:
            return batch_id

        # Poll for completion
        print(f"Waiting for batch completion (polling every {poll_interval}s)...")
        while True:
            status_info = self.check_batch_status(batch_id)
            print(
                f"Status: {status_info['status']} | "
                f"Completed: {status_info['request_counts']['completed']} | "
                f"Failed: {status_info['request_counts']['failed']}"
            )

            # OpenAI statuses: completed, failed, expired, cancelled
            # Mistral statuses: SUCCESS, FAILED, CANCELLED
            if status_info["status"] in [
                "completed",
                "failed",
                "expired",
                "cancelled",
                "SUCCESS",
                "FAILED",
                "CANCELLED",
            ]:
                break

            time.sleep(poll_interval)

        return batch_id

    def run_experiment(
        self,
        df: pd.DataFrame,
        batch_size: int = 20,
        sample_size: int = None,
        use_batch_api: bool = False,
        output_dir: str = None,
    ):
        """
        Run complete classification experiment with evaluation and visualization.

        Performs end-to-end classification experiment including prediction,
        evaluation metrics calculation, and result visualization.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing 'text' and 'label' columns.
        batch_size : int, default=20
            Batch size for processing (if using batch mode).
        sample_size : int, optional
            Number of samples to randomly select for classification.
        use_batch_api : bool, default=False
            Whether to use batch API for classification.
        output_dir : str, optional
            Directory to save results and visualizations.

        Returns
        -------
        dict
            Dictionary containing:
            - predictions (list): Classification results
            - metrics (dict): Evaluation metrics (precision, recall, F1, etc.)

        Notes
        -----
        Generates outputs including:
        - Confusion matrix visualization
        - Classification report
        - Cost analysis and metrics summary
        - Results CSV with predictions
        """
        # Ensure output directory exists if provided
        if output_dir:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        email_texts = df["text"].astype(str).tolist()
        true_labels = df["label"].astype(int).tolist()

        # Cost estimation with provider-specific tokenizers
        print("\n=== Cost Estimation ===")
        cost_estimate = self.estimate_cost(
            email_texts[: min(100, len(email_texts))]
        )  # Sample first 100 and extrapolate it, so we don't need to wait for too long
        if "error" in cost_estimate:
            print(f"Warning: {cost_estimate['error']}")
            print("Proceeding without cost estimation...")
        else:
            estimated_cost = cost_estimate["estimated_cost"] * (
                len(email_texts) / cost_estimate["num_texts"]
            )

            # Apply batch discount if using batch API
            if use_batch_api:
                estimated_cost *= 0.5  # 50% discount for both providers' batch API
                print(
                    f"Estimated cost for {len(email_texts)} texts: ${estimated_cost:.4f} (with 50% batch discount)"
                )
            else:
                print(
                    f"Estimated cost for {len(email_texts)} texts: ${estimated_cost:.4f} (sequential API)"
                )

            print(f"Model: {cost_estimate['model']}")
            confirm = input("Continue? (y/n): ")
            if confirm.lower() != "y":
                print("Aborted.")
                return

        if use_batch_api:
            batch_api_name = "OpenAI" if self.provider == "openai" else "Mistral"
            print(
                f"Using {batch_api_name} Batch API (50% cheaper, up to 24h processing)..."
            )
            batch_id = self.classify_batch_async(email_texts, wait_for_completion=True)
            preds_raw, confidences = self.retrieve_batch_results(batch_id)
        else:
            print("Using sequential API calls...")
            preds_raw, confidences = self.classify_batch(
                email_texts, batch_size=batch_size
            )

        # Filter out failed predictions
        valid_indices = [i for i, p in enumerate(preds_raw) if p is not None]
        failed_indices = [i for i, p in enumerate(preds_raw) if p is None]

        # Only keep valid predictions and corresponding true labels
        predictions = [preds_raw[i] for i in valid_indices]
        true_labels_filtered = [true_labels[i] for i in valid_indices]

        # Process results
        return self._process_results(
            email_texts,
            true_labels_filtered,
            preds_raw,
            confidences,
            failed_indices,
            predictions,
            output_dir,
        )

    def _process_results(
        self,
        email_texts,
        true_labels,
        preds_raw,
        confidences,
        failed_indices,
        predictions,
        output_dir,
    ):
        """
        Process classification results and generate evaluation outputs.

        Computes evaluation metrics, creates visualizations, and saves
        detailed results for analysis and reporting.

        Parameters
        ----------
        email_texts : list[str]
            Original email texts that were classified.
        true_labels : list[int]
            Ground truth labels (0=legitimate, 1=malicious).
        preds_raw : list[int | None]
            Raw predictions from classifier (may contain None for failures).
        confidences : list[float]
            Confidence scores for predictions.
        failed_indices : list[int]
            Indices of samples that failed classification.
        predictions : list[int]
            Processed predictions with failures handled.
        output_dir : str
            Directory to save results and visualizations.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics and confusion matrix.

        Notes
        -----
        Generates multiple output files:
        - Confusion matrix PNG
        - Classification report
        - Detailed results CSV
        - Metrics summary JSON
        """
        valid_confidences = [
            c for i, c in enumerate(confidences) if preds_raw[i] is not None
        ]
        has_confidence_data = valid_confidences and max(valid_confidences) > 0

        if has_confidence_data:
            avg_confidence = np.mean(valid_confidences)
            min_confidence = np.min(valid_confidences)
            max_confidence = np.max(valid_confidences)
            low_conf_threshold = 0.8
            low_conf_count = sum(1 for c in valid_confidences if c < low_conf_threshold)

            print("\nConfidence Statistics:")
            print(f"Average confidence: {avg_confidence:.4f}")
            print(f"Min/Max confidence: {min_confidence:.4f} / {max_confidence:.4f}")
            print(
                f"Low confidence (<{low_conf_threshold}): {low_conf_count} predictions"
            )

            # Create confidence distribution visualization
            if output_dir:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))

                # Histogram of confidence scores
                ax.hist(
                    valid_confidences,
                    bins=30,
                    edgecolor="black",
                    alpha=0.7,
                    color="steelblue",
                )
                ax.axvline(
                    avg_confidence,
                    color="red",
                    linestyle="--",
                    label=f"Mean: {avg_confidence:.3f}",
                )
                ax.axvline(
                    low_conf_threshold,
                    color="orange",
                    linestyle="--",
                    label=f"Threshold: {low_conf_threshold}",
                )
                ax.set_xlabel("Confidence Score (from logprobs)")
                ax.set_ylabel("Frequency")
                ax.set_title(f"{self.provider.title()} Confidence Distribution")
                ax.legend()
                ax.grid(True, alpha=0.3)

                confidence_plot_path = os.path.join(
                    output_dir, "confidence_distribution.png"
                )
                plt.savefig(confidence_plot_path, dpi=100, bbox_inches="tight")
                plt.close()
                print(f"\nConfidence visualization saved to: {confidence_plot_path}")
        else:
            print(
                f"\nNote: {self.provider.title()} batch API does not provide confidence scores (logprobs)"
            )

        if failed_indices:
            print(
                f"\n{len(failed_indices)} classifications failed – using fallback={LABELS[0]}"
            )

        print("\nComputing metrics...")
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average="macro", zero_division=0
        )
        precision_per_class, recall_per_class, f1_per_class, _ = (
            precision_recall_fscore_support(
                true_labels, predictions, average=None, zero_division=0
            )
        )
        cm = confusion_matrix(true_labels, predictions)

        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum() if cm.sum() else 0.0

        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {f1:.4f}")
        print(f"Macro Precision: {precision:.4f}")
        print(f"Macro Recall: {recall:.4f}")
        print(f"Legitimate F1: {f1_per_class[0]:.4f}")
        print(f"Malicious F1: {f1_per_class[1]:.4f}")
        print(f"Legitimate Precision: {precision_per_class[0]:.4f}")
        print(f"Malicious Precision: {precision_per_class[1]:.4f}")
        print(f"Legitimate Recall: {recall_per_class[0]:.4f}")
        print(f"Malicious Recall: {recall_per_class[1]:.4f}")

        print("\nConfusion Matrix:")
        print(cm)
        print("(rows=true labels, cols=predicted)")

        print("\nDetailed Classification Report:")
        print(
            classification_report(
                true_labels,
                predictions,
                target_names=["Legitimate", "Malicious"],
                digits=4,
            )
        )

        # Error analysis
        df_results = pd.DataFrame(
            {
                "email_text": email_texts,
                "true_label": true_labels,
                "predicted_label": predictions,
                "confidence": confidences,
            }
        )
        df_results["correct"] = (
            df_results["predicted_label"] == df_results["true_label"]
        )

        # Save predictions to CSV if output_dir is specified
        if output_dir:
            predictions_path = os.path.join(output_dir, "zs_predictions.csv")
            df_results.to_csv(predictions_path, index=False)
            print(f"\nPredictions saved to: {predictions_path}")

        errors = df_results[~df_results["correct"]]
        print("\nError Analysis:")
        total_samples = len(df_results)
        error_pct = len(errors) / total_samples * 100
        print(f"Total errors: {len(errors)} / {total_samples} ({error_pct:.2f}%)")

        false_pos = df_results[
            (df_results["true_label"] == 0) & (df_results["predicted_label"] == 1)
        ]
        false_neg = df_results[
            (df_results["true_label"] == 1) & (df_results["predicted_label"] == 0)
        ]

        fp_pct = len(false_pos) / total_samples * 100
        fn_pct = len(false_neg) / total_samples * 100
        print(f"False Positives: {len(false_pos)} ({fp_pct:.2f}%)")
        print(f"False Negatives: {len(false_neg)} ({fn_pct:.2f}%)")

        # Save confusion matrix visualization if output_dir is specified
        if output_dir:
            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=["Legitimate", "Malicious"]
            )
            disp.plot(ax=ax, cmap="Blues", values_format="d")
            plt.title("Confusion Matrix - Zero Shot Classification")
            cm_path = os.path.join(output_dir, "confusion_matrix.png")
            plt.savefig(cm_path, dpi=100, bbox_inches="tight")
            plt.close()
            print(f"Confusion matrix saved to: {cm_path}")

            # Save metrics summary to text file
            metrics_txt_path = os.path.join(output_dir, "zs_metrics.txt")
            with open(metrics_txt_path, "w") as f:
                f.write("Zero-Shot Classification Metrics\n")
                f.write("================================\n\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Macro F1: {f1:.4f}\n")
                f.write(f"Macro Precision: {precision:.4f}\n")
                f.write(f"Macro Recall: {recall:.4f}\n\n")
                f.write("Per-class metrics:\n")
                f.write(f"Legitimate F1: {f1_per_class[0]:.4f}\n")
                f.write(f"Malicious F1: {f1_per_class[1]:.4f}\n")
                f.write(f"Legitimate Precision: {precision_per_class[0]:.4f}\n")
                f.write(f"Malicious Precision: {precision_per_class[1]:.4f}\n")
                f.write(f"Legitimate Recall: {recall_per_class[0]:.4f}\n")
                f.write(f"Malicious Recall: {recall_per_class[1]:.4f}\n")
            print(f"Metrics summary saved to: {metrics_txt_path}")

        # Display and save metrics summary
        if self.track_metrics:
            print("\n" + "=" * 50)
            print("COST & PERFORMANCE METRICS")
            print("=" * 50)
            metrics_summary = self.get_metrics_summary()
            for key, value in metrics_summary.items():
                if key != "error_breakdown":
                    print(f"{key}: {value}")
            print("\nError breakdown:")
            for error_type, count in metrics_summary["error_breakdown"].items():
                print(f"  {error_type}: {count}")

            # Save metrics to JSON
            metrics_file = (
                os.path.join(output_dir, "zs_metrics_detailed.json")
                if output_dir
                else "zs_metrics_detailed.json"
            )
            with open(metrics_file, "w") as f:
                json.dump(
                    {
                        "metrics_summary": metrics_summary,
                        "confidence_stats": {
                            "avg": (
                                avg_confidence if "avg_confidence" in locals() else None
                            ),
                            "min": (
                                min_confidence if "min_confidence" in locals() else None
                            ),
                            "max": (
                                max_confidence if "max_confidence" in locals() else None
                            ),
                            "low_count": (
                                low_conf_count if "low_conf_count" in locals() else None
                            ),
                        },
                        "classification_metrics": {
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                        },
                    },
                    f,
                    indent=2,
                )
            print(f"\nDetailed metrics saved to: {metrics_file}")

        print("\n Results saved to output directory")

        return {
            "predictions": predictions,
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": cm,
            },
            "failed_indices": failed_indices,
        }


def main():
    """
    Main function with command-line interface for zero-shot classification.

    Provides complete pipeline for zero-shot phishing detection including data loading,
    model selection, cost estimation, classification execution, and results evaluation
    with performance metrics and visualizations.

    Notes
    -----
    Supports both OpenAI and Mistral providers with automatic API key detection.
    Implements both sequential and batch processing modes for different use cases.
    Generates confusion matrices, classification reports, and cost breakdowns.
    Includes sampling options for large datasets and detailed error analysis.
    """
    parser = argparse.ArgumentParser(
        description="Zero-shot phishing email classification using GPT"
    )

    # Required arguments
    parser.add_argument("--test-set", type=str, required=True, help="Path to test set")

    # Model and API settings
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "mistral"],
        help="AI provider to use (default: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use (default: gpt-4o-mini for OpenAI, mistral-medium for Mistral)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (default: use OPENAI_API_KEY or MISTRAL_API_KEY env var)",
    )

    # Processing options
    parser.add_argument(
        "--use-batch",
        action="store_true",
        help="Use Batch API (50%% cheaper, up to 24h)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for sequential processing (default: 20)",
    )
    parser.add_argument(
        "--rpm", type=int, default=500, help="Requests per minute limit (default: 500)"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for output files (default: current dir)",
    )
    parser.add_argument(
        "--no-cost-confirm", action="store_true", help="Skip cost confirmation prompt"
    )

    # Confidence settings
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
        help="Threshold for low confidence warnings (default: 0.8)",
    )

    # Text preprocessing
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable text normalization (use raw text)",
    )

    # Debugging
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed progress and metrics"
    )

    args = parser.parse_args()

    # Set default model for provider if not specified
    if args.model == "gpt-4o-mini" and args.provider == "mistral":
        args.model = "mistral-medium-latest"

    # Get API key based on provider
    if args.provider == "openai":
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        env_var = "OPENAI_API_KEY"
    elif args.provider == "mistral":
        api_key = args.api_key or os.getenv("MISTRAL_API_KEY")
        env_var = "MISTRAL_API_KEY"

    if not api_key:
        print(
            f"ERROR: No API key provided. Use --api-key or set {env_var} environment variable"
        )
        return 1

    # Load data
    print(f"Loading data from: {args.test_set}")
    df = pd.read_csv(args.test_set)

    # Limit samples if requested
    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"Limited to {args.max_samples} samples for testing")

    print(f"Loaded {len(df)} samples")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    # Initialize classifier
    classifier = ZSClassifier(
        api_key=api_key,
        model=args.model,
        provider=args.provider,
        requests_per_minute=args.rpm,
        track_metrics=True,
        normalize_text=not args.no_normalize,  # Invert the flag since default is True
    )

    # Create output directory but don't change to it
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Skip cost confirmation if requested
        if args.no_cost_confirm:
            # Monkey patch to skip confirmation
            original_input = builtins.input
            builtins.input = lambda _: "y"

        # Run experiment
        print(f"\nStarting {args.provider} classification with model: {args.model}")
        print(f"Using {'Batch' if args.use_batch else 'Sequential'} API")

        classifier.run_experiment(
            df=df,
            use_batch_api=args.use_batch,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )

        if args.verbose and classifier.track_metrics:
            print("\n" + "=" * 50)
            print("FINAL METRICS SUMMARY")
            print("=" * 50)
            summary = classifier.get_metrics_summary()
            for key, value in summary.items():
                print(f"{key}: {value}")

        print(f"\nResults saved to: {args.output_dir}/")
        return 0

    finally:
        # Restore original input if patched
        if args.no_cost_confirm and "original_input" in locals():
            builtins.input = original_input


if __name__ == "__main__":
    sys.exit(main())
