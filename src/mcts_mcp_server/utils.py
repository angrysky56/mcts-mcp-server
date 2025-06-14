#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility Functions for MCTS
==========================

This module provides various utility functions used across the MCTS package.
"""
import logging
import re
import json
from typing import List, Any, cast # For type hints if sklearn is not available

# Setup logger for this module's internal use
logger = logging.getLogger(__name__)

# Try to import sklearn components if available
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Define placeholder type hints if sklearn is not available
    TfidfVectorizer = Any
    ENGLISH_STOP_WORDS: List[str] = [] # Use an empty list as a safe placeholder
    cosine_similarity = Any

def setup_logger(name: str = "mcts_default_logger", level: int = logging.INFO) -> logging.Logger:
    """
    Set up a configurable logger with proper formatting and handlers.

    Args:
        name: The name of the logger instance
        level: The logging level (e.g., logging.INFO, logging.DEBUG)

    Returns:
        Configured logger instance ready for use

    Note:
        Avoids duplicate handlers if logger already configured
        Sets propagate=False to prevent duplicate messages in child loggers
    """
    log = logging.getLogger(name) # Use 'log' to avoid conflict with module-level 'logger'
    log.setLevel(level)
    # Avoid adding handlers if already configured by a higher-level setup
    if not log.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)
        log.propagate = False # Don't duplicate to root logger if this is a child
    else:
        # If handlers exist, just ensure level is set (e.g. if root logger configured)
        for handler_item in log.handlers: # Renamed to avoid conflict
            handler_item.setLevel(level)
    return log

def truncate_text(text: Any, max_length: int = 200) -> str:
    """
    Truncate text for display purposes with smart word boundary handling.

    Args:
        text: Text to truncate (will be converted to string)
        max_length: Maximum length before truncation

    Returns:
        Truncated text with "..." suffix if truncated, cleaned of markdown artifacts

    Note:
        Attempts to break at word boundaries and removes common markup prefixes
    """
    if not text:
        return ""
    text_str = str(text).strip() # Ensure text is string
    text_str = re.sub(r"^(json|markdown)?\s*", "", text_str, flags=re.IGNORECASE | re.MULTILINE)
    text_str = re.sub(r"\s*$", "", text_str, flags=re.MULTILINE).strip()
    if len(text_str) <= max_length:
        return text_str
    last_space = text_str.rfind(" ", 0, max_length)
    return text_str[:last_space] + "..." if last_space != -1 else text_str[:max_length] + "..."
def calculate_semantic_distance(text1: Any, text2: Any, use_tfidf: bool = True) -> float:
    """
    Calculate semantic distance between two texts using TF-IDF or Jaccard similarity.

    Args:
        text1: First text for comparison (will be converted to string)
        text2: Second text for comparison (will be converted to string)
        use_tfidf: Whether to use TF-IDF vectorization (requires sklearn)

    Returns:
        Distance value where 0.0 = identical texts, 1.0 = completely different

    Algorithm:
        1. If sklearn available and use_tfidf=True: Uses TF-IDF cosine similarity
        2. Falls back to Jaccard similarity on word sets
        3. Handles edge cases (empty texts, vectorization failures)

    Note:
        TF-IDF method is more semantically aware but requires sklearn
        Jaccard fallback works on word overlap and is always available
    """
    if not text1 or not text2:
        return 1.0
    s_text1, s_text2 = str(text1), str(text2) # Ensure strings

    if SKLEARN_AVAILABLE and use_tfidf:
        try:
            # Ensure ENGLISH_STOP_WORDS is a list if SKLEARN_AVAILABLE is True
            custom_stop_words = list(ENGLISH_STOP_WORDS) + ["analysis", "however", "therefore", "furthermore", "perspective"]
            from sklearn.feature_extraction.text import TfidfVectorizer as ActualTfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as actual_cosine_similarity
            vectorizer = ActualTfidfVectorizer(stop_words=custom_stop_words, max_df=0.9, min_df=1)
            tfidf_matrix = vectorizer.fit_transform([s_text1, s_text2])
            if tfidf_matrix.shape[0] < 2 or tfidf_matrix.shape[1] == 0:
                logger.debug(f"TF-IDF matrix issue (shape: {tfidf_matrix.shape}) for texts. Falling back to Jaccard.")
            else:
                similarity = actual_cosine_similarity(tfidf_matrix.getrow(0).toarray(), tfidf_matrix.getrow(1).toarray())[0][0]
                similarity = max(0.0, min(1.0, similarity)) # Clamp similarity
                return 1.0 - similarity
        except ValueError as ve:
            logger.warning(f"TF-IDF ValueError: {ve} for texts '{truncate_text(s_text1, 50)}' vs '{truncate_text(s_text2, 50)}'. Falling back to Jaccard.")
        except Exception as e:
            logger.warning(f"TF-IDF semantic distance error: {e}. Falling back to Jaccard.")

    try:
        words1 = set(re.findall(r'\w+', s_text1.lower()))
        words2 = set(re.findall(r'\w+', s_text2.lower()))
        if not words1 and not words2:
            return 0.0
        if not words1 or not words2:
            return 1.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        if union == 0:
            return 0.0
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity
    except Exception as fallback_e:
        logger.error(f"Jaccard similarity fallback failed for '{truncate_text(s_text1,50)}' vs '{truncate_text(s_text2,50)}': {fallback_e}")
        return 1.0
def _summarize_text(text: str, max_words: int = 50) -> str:
    """
    Summarize text using TF-IDF sentence scoring or simple truncation.

    Args:
        text: Text to summarize
        max_words: Target maximum number of words in summary

    Returns:
        Summarized text, truncated with "..." if needed

    Algorithm:
        1. If text <= max_words: return as-is
        2. If sklearn available: Use TF-IDF to score sentences, select top-scoring
        3. Fallback: Simple word truncation with "..." suffix

    Note:
        TF-IDF method preserves most important sentences based on term frequency
        Originally moved from mcts_core.py for better organization
    """
    if not text:
        return "N/A"

    words = re.findall(r'\w+', text)
    if len(words) <= max_words:
        return text.strip()

    if SKLEARN_AVAILABLE:
        try:
            sentences = re.split(r'[.!?]+\s*', text)
            sentences = [s for s in sentences if len(s.split()) > 3]
            if not sentences:
                return ' '.join(words[:max_words]) + '...'

            from sklearn.feature_extraction.text import TfidfVectorizer as ActualTfidfVectorizer
            vectorizer = ActualTfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sparse_sums = cast(Any, tfidf_matrix).sum(axis=1)
            sentence_scores = sparse_sums.toarray().flatten()

            num_summary_sentences = max(1, min(3, len(sentences) // 5))
            top_sentence_indices = sentence_scores.argsort()[-num_summary_sentences:][::-1]
            top_sentence_indices.sort()

            summary = ' '.join([sentences[i] for i in top_sentence_indices])
            summary_words = summary.split()

            if len(summary_words) > max_words * 1.2:
                summary = ' '.join(summary_words[:max_words]) + '...'
            elif len(words) > len(summary_words):
                summary += '...'
            return summary.strip()

        except Exception as e:
            logger.warning(f"TF-IDF summary failed ({e}). Truncating.")

    return ' '.join(words[:max_words]) + '...'

def validate_config_dict(config: dict, required_keys: List[str]) -> bool:
    """
    Validate that a configuration dictionary contains required keys.

    Args:
        config: Configuration dictionary to validate
        required_keys: List of keys that must be present

    Returns:
        True if all required keys present with non-None values, False otherwise

    Note:
        Useful for validating MCTS configuration before initialization
    """
    if not isinstance(config, dict):
        return False

    for key in required_keys:
        if key not in config or config[key] is None:
            logger.warning(f"Missing required config key: {key}")
            return False

    return True

def safe_json_serialize(obj: Any) -> str:
    """
    Safely serialize an object to JSON with fallback for non-serializable objects.

    Args:
        obj: Object to serialize

    Returns:
        JSON string representation, with fallbacks for problematic objects

    Note:
        Handles common serialization issues like datetime objects, numpy types
    """
    def json_serializer(obj: Any) -> Any:
        """Custom serializer for JSON encoding."""
        if hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, '__dict__'):  # custom objects
            return obj.__dict__
        else:
            return str(obj)

    try:
        return json.dumps(obj, default=json_serializer, indent=2)
    except Exception as e:
        logger.warning(f"JSON serialization failed: {e}")
        return json.dumps({"error": "Serialization failed", "object_type": str(type(obj))})

def extract_numeric_value(text: str, default: float = 0.0) -> float:
    """
    Extract the first numeric value from a text string.

    Args:
        text: Text to search for numeric values
        default: Default value if no number found

    Returns:
        First numeric value found, or default if none found

    Note:
        Useful for parsing LLM responses that should contain numeric scores
    """
    if not text:
        return default

    # Look for integers and floats
    matches = re.findall(r'-?\d+\.?\d*', text)

    if matches:
        try:
            return float(matches[0])
        except ValueError:
            pass

    return default

