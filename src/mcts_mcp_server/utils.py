#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility Functions for MCTS
==========================

This module provides various utility functions used across the MCTS package.
"""
import logging
import re
from typing import List, Any # For type hints if sklearn is not available
import numpy as np # Added for _summarize_text

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
    """Sets up a configurable logger.

    Args:
        name (str): The name of the logger.
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    l = logging.getLogger(name) # Use 'l' to avoid conflict with module-level 'logger'
    l.setLevel(level)
    # Avoid adding handlers if already configured by a higher-level setup
    if not l.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        l.addHandler(handler)
        l.propagate = False # Don't duplicate to root logger if this is a child
    else:
        # If handlers exist, just ensure level is set (e.g. if root logger configured)
        for handler_item in l.handlers: # Renamed to avoid conflict
            handler_item.setLevel(level)
    return l

def truncate_text(text: Any, max_length: int = 200) -> str:
    """Truncates text for display purposes, ensuring it's a string first."""
    if not text: return ""
    text_str = str(text).strip() # Ensure text is string
    text_str = re.sub(r"^```(json|markdown)?\s*", "", text_str, flags=re.IGNORECASE | re.MULTILINE)
    text_str = re.sub(r"\s*```$", "", text_str, flags=re.MULTILINE).strip()
    if len(text_str) <= max_length: return text_str
    last_space = text_str.rfind(" ", 0, max_length)
    return text_str[:last_space] + "..." if last_space != -1 else text_str[:max_length] + "..."

def calculate_semantic_distance(text1: Any, text2: Any, use_tfidf: bool = True) -> float:
    """
    Calculates semantic distance (0=identical, 1=different).
    Uses TF-IDF if available and enabled, otherwise Jaccard.
    Ensures inputs are strings.
    """
    if not text1 or not text2: return 1.0
    s_text1, s_text2 = str(text1), str(text2) # Ensure strings

    if SKLEARN_AVAILABLE and use_tfidf:
        try:
            # Ensure ENGLISH_STOP_WORDS is a list if SKLEARN_AVAILABLE is True
            custom_stop_words = list(ENGLISH_STOP_WORDS) + ["analysis", "however", "therefore", "furthermore", "perspective"]
            vectorizer = TfidfVectorizer(stop_words=custom_stop_words, max_df=0.9, min_df=1)
            tfidf_matrix = vectorizer.fit_transform([s_text1, s_text2])
            if tfidf_matrix.shape[0] < 2 or tfidf_matrix.shape[1] == 0:
                logger.debug(f"TF-IDF matrix issue (shape: {tfidf_matrix.shape}) for texts. Falling back to Jaccard.")
            else:
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                similarity = max(0.0, min(1.0, similarity)) # Clamp similarity
                return 1.0 - similarity
        except ValueError as ve:
            logger.warning(f"TF-IDF ValueError: {ve} for texts '{truncate_text(s_text1, 50)}' vs '{truncate_text(s_text2, 50)}'. Falling back to Jaccard.")
        except Exception as e:
            logger.warning(f"TF-IDF semantic distance error: {e}. Falling back to Jaccard.")

    try:
        words1 = set(re.findall(r'\w+', s_text1.lower()))
        words2 = set(re.findall(r'\w+', s_text2.lower()))
        if not words1 and not words2: return 0.0
        if not words1 or not words2: return 1.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        if union == 0: return 0.0
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity
    except Exception as fallback_e:
        logger.error(f"Jaccard similarity fallback failed for '{truncate_text(s_text1,50)}' vs '{truncate_text(s_text2,50)}': {fallback_e}")
        return 1.0

def _summarize_text(text: str, max_words: int = 50) -> str:
    """
    Summarizes text using TF-IDF (if available) or simple truncation.
    Moved from mcts_core.py.
    """
    if not text: return "N/A"
    words = re.findall(r'\w+', text)
    if len(words) <= max_words: return text.strip()

    if SKLEARN_AVAILABLE:
         try:
            sentences = re.split(r'[.!?]+\s*', text)
            sentences = [s for s in sentences if len(s.split()) > 3]
            if not sentences: return ' '.join(words[:max_words]) + '...'

            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

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
