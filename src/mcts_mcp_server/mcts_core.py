#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core MCTS Implementation
=======================

This module implements the Monte Carlo Tree Search (MCTS) algorithm
for advanced analysis and reasoning.
"""
import logging
import random
import math
import asyncio
import json
import re
import os
import sqlite3
from datetime import datetime
from collections import Counter, namedtuple
from typing import (
    List, Optional, Dict, Any, Tuple, Set, Union, Protocol, Generator, AsyncGenerator
)

import numpy as np

# Try to import sklearn components if available
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    TfidfVectorizer, ENGLISH_STOP_WORDS, cosine_similarity = None, None, None

# ==============================================================================
# Default Configuration
# ==============================================================================

DEFAULT_CONFIG = {
    "max_children": 6,               # Reduced from 10 to speed up processing
    "exploration_weight": 3.0,
    "max_iterations": 1,
    "simulations_per_iteration": 5,  # Reduced from 10 to speed up processing
    "surprise_threshold": 0.66,
    "use_semantic_distance": True,
    "relative_evaluation": False,
    "score_diversity_bonus": 0.7,
    "force_exploration_interval": 4,
    "debug_logging": False,
    "global_context_in_prompts": True,
    "track_explored_approaches": True,
    "sibling_awareness": True,
    "memory_cutoff": 20,             # Reduced from 50 to use less memory
    "early_stopping": True,
    "early_stopping_threshold": 8.0,  # Reduced from 10.0 to stop earlier with good results
    "early_stopping_stability": 1,    # Reduced from 2 to stop faster when a good result is found
    "surprise_semantic_weight": 0.4,
    "surprise_philosophical_shift_weight": 0.3,
    "surprise_novelty_weight": 0.3,
    "surprise_overall_threshold": 0.7,
    "use_bayesian_evaluation": True,
    "use_thompson_sampling": True,
    "beta_prior_alpha": 1.0,
    "beta_prior_beta": 1.0,
    "unfit_score_threshold": 5.0,
    "unfit_visit_threshold": 3,
    "enable_state_persistence": True,
}

# ==============================================================================
# Approach Taxonomy & Metadata
# ==============================================================================
APPROACH_TAXONOMY = {
    "empirical": ["evidence", "data", "observation", "experiment"],
    "rational": ["logic", "reason", "deduction", "principle"],
    "phenomenological": ["experience", "perception", "consciousness"],
    "hermeneutic": ["interpret", "meaning", "context", "understanding"],
    "reductionist": ["reduce", "component", "fundamental", "elemental"],
    "holistic": ["whole", "system", "emergent", "interconnected"],
    "materialist": ["physical", "concrete", "mechanism"],
    "idealist": ["concept", "ideal", "abstract", "mental"],
    "analytical": ["analyze", "dissect", "examine", "scrutinize"],
    "synthetic": ["synthesize", "integrate", "combine", "unify"],
    "dialectical": ["thesis", "antithesis", "contradiction"],
    "comparative": ["compare", "contrast", "analogy"],
    "critical": ["critique", "challenge", "question", "flaw"],
    "constructive": ["build", "develop", "formulate"],
    "pragmatic": ["practical", "useful", "effective"],
    "normative": ["should", "ought", "value", "ethical"],
    "structural": ["structure", "organize", "framework"],
    "alternative": ["alternative", "different", "another way"],
    "complementary": ["missing", "supplement", "add"],
    "variant": [],
    "initial": [],
}

APPROACH_METADATA = {
    "empirical": {"family": "epistemology"},
    "rational": {"family": "epistemology"},
    "phenomenological": {"family": "epistemology"},
    "hermeneutic": {"family": "epistemology"},
    "reductionist": {"family": "ontology"},
    "holistic": {"family": "ontology"},
    "materialist": {"family": "ontology"},
    "idealist": {"family": "ontology"},
    "analytical": {"family": "methodology"},
    "synthetic": {"family": "methodology"},
    "dialectical": {"family": "methodology"},
    "comparative": {"family": "methodology"},
    "critical": {"family": "perspective"},
    "constructive": {"family": "perspective"},
    "pragmatic": {"family": "perspective"},
    "normative": {"family": "perspective"},
    "structural": {"family": "general"},
    "alternative": {"family": "general"},
    "complementary": {"family": "general"},
    "variant": {"family": "general"},
    "initial": {"family": "general"},
}

# ==============================================================================
# Prompts
# ==============================================================================

INITIAL_PROMPT = """<instruction>Provide an initial analysis and interpretation of the core themes, arguments, and potential implications presented. Identify key concepts. Respond with clear, natural language text ONLY.</instruction><question>{question}</question>"""

THOUGHTS_PROMPT = """<instruction>Critically examine the current analysis below. Suggest a SIGNIFICANTLY DIFFERENT interpretation, identify a MAJOR underlying assumption or weakness, or propose a novel connection to another domain or concept. Push the thinking in a new direction.

**Previous Run Context (If Available):**
- Previous Best Analysis Summary: {previous_best_summary}
- Previously Marked Unfit Concepts/Areas: {unfit_markers_summary}
- Learned Approach Preferences: {learned_approach_summary}

Consider this context. Avoid repeating unfit areas unless you have a novel mutation. Build upon previous success if appropriate, or diverge strongly if needed.</instruction>
<context>Original Text Summary: {question_summary}\nBest Overall Analysis (Score {best_score}/10): {best_answer}\nCurrent Analysis (Node {current_sequence}): {current_answer}\nCurrent Analysis Tags: {current_tags}</context>
Generate your critique or alternative direction.</instruction>"""

UPDATE_PROMPT = """<instruction>Substantially revise the draft analysis below to incorporate the core idea from the critique. Develop the analysis further based on this new direction.

**Previous Run Context (If Available):**
- Previous Best Analysis Summary: {previous_best_summary}
- Previously Marked Unfit Concepts/Areas: {unfit_markers_summary}

Ensure the revision considers past findings and avoids known unproductive paths unless the critique justifies revisiting them.</instruction>
<context>Original Text Summary: {question_summary}\nBest Overall Analysis (Score {best_score}/10): {best_answer}\nCurrent Analysis Tags: {current_tags}</context>
<draft>{answer}</draft>
<critique>{improvements}</critique>
Write the new, revised analysis text."""

EVAL_ANSWER_PROMPT = """<instruction>Evaluate the intellectual quality and insightfulness of the analysis below (1-10) concerning the original input. Higher scores for depth, novelty, and relevance. Use the full 1-10 scale. Reserve 9-10 for truly exceptional analyses that significantly surpass previous best analysis ({best_score}/10).

**Previous Run Context (If Available):**
- Previous Best Analysis Summary: {previous_best_summary}
- Previously Marked Unfit Concepts/Areas: {unfit_markers_summary}

Consider if this analysis productively builds upon or diverges from past findings.</instruction>
<context>Original Text Summary: {question_summary}\nBest Overall Analysis (Score {best_score}/10): {best_answer}\nAnalysis Tags: {current_tags}</context>
<answer_to_evaluate>{answer_to_evaluate}</answer_to_evaluate>
How insightful, deep, relevant, and well-developed is this analysis compared to the best so far? Does it offer a genuinely novel perspective or intelligently navigate known issues? Rate 1-10 based purely on merit. Respond with a logical rating from 1 to 10.</instruction>"""

TAG_GENERATION_PROMPT = """<instruction>Generate concise keyword tags summarizing the main concepts in the following text. Output the tags, separated by commas.</instruction>\n<text_to_tag>{analysis_text}</text_to_tag>"""

FINAL_SYNTHESIS_PROMPT = """<instruction>Synthesize the key insights developed along the primary path of analysis below into a conclusive statement addressing the original question. Focus on the progression of ideas represented.</instruction>
<original_question_summary>{question_summary}</original_question_summary>
<initial_analysis>{initial_analysis_summary}</initial_analysis>
<best_analysis_score>{best_score}/10</best_analysis_score>
<development_path>
{path_thoughts}
</development_path>
<final_best_analysis>{final_best_analysis_summary}</final_best_analysis>
Synthesize into a final conclusion:"""

INTENT_CLASSIFIER_PROMPT = """Classify user requests. Choose the *single best* category from the list. Respond appropriately.

Categories:
- CONTINUE_ANALYSIS: User wants to continue, refine, or build upon the previous MCTS analysis run (e.g., "elaborate", "explore X further", "what about Y?").
- ANALYZE_NEW: User wants a fresh MCTS analysis on a new provided text/task, ignoring any previous runs in this chat.
- ASK_LAST_RUN_SUMMARY: User asks about the outcome, score, or details of the previous MCTS run (e.g., "what was the score?", "summarize last run").
- ASK_PROCESS: User asks how the analysis process works (e.g., "how do you work?", "what algorithm is this?").
- ASK_CONFIG: User asks about the current MCTS parameters or settings (e.g., "show config", "what are the settings?").
- GENERAL_CONVERSATION: The input is conversational, off-topic, or a simple greeting/closing.

User Input:
"{raw_input_text}"

Classification:
"""

# ==============================================================================
# Utility Functions
# ==============================================================================

def setup_logger(name="mcts_core", level=logging.INFO):
    """Sets up a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Avoid adding handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False # Don't duplicate to root logger
    else:
        # If handlers exist, just ensure level is set
        for handler in logger.handlers:
            handler.setLevel(level)
    return logger

logger = setup_logger() # Initialize default logger

def truncate_text(text, max_length=200):
    """Truncates text for display purposes."""
    if not text: return ""
    text = str(text).strip()
    text = re.sub(r"^```(json|markdown)?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()
    if len(text) <= max_length: return text
    last_space = text.rfind(" ", 0, max_length)
    return text[:last_space] + "..." if last_space != -1 else text[:max_length] + "..."

def calculate_semantic_distance(text1, text2, use_tfidf=True):
    """
    Calculates semantic distance (0=identical, 1=different).
    Uses TF-IDF if available and enabled, otherwise Jaccard.
    """
    if not text1 or not text2: return 1.0
    text1, text2 = str(text1), str(text2)

    if SKLEARN_AVAILABLE and use_tfidf:
        try:
            custom_stop_words = list(ENGLISH_STOP_WORDS) + ["analysis", "however", "therefore", "furthermore", "perspective"]
            vectorizer = TfidfVectorizer(stop_words=custom_stop_words, max_df=0.9, min_df=1)
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            if tfidf_matrix.shape[0] < 2 or tfidf_matrix.shape[1] == 0:
                raise ValueError(f"TF-IDF matrix issue (shape: {tfidf_matrix.shape}).")
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            similarity = max(0.0, min(1.0, similarity))
            return 1.0 - similarity
        except Exception as e:
            logger.warning(f"TF-IDF semantic distance error: {e}. Falling back to Jaccard.")

    # Fallback to Jaccard Similarity
    try:
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        if not words1 or not words2: return 1.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        if union == 0: return 0.0
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity
    except Exception as fallback_e:
        logger.error(f"Jaccard similarity fallback failed: {fallback_e}")
        return 1.0

# ==============================================================================
# LLM Interface Protocol
# ==============================================================================

class LLMInterface(Protocol):
    """Defines the interface required for LLM interactions."""

    async def get_completion(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        """Gets a non-streaming completion from the LLM."""
        ...

    async def get_streaming_completion(self, model: str, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Gets a streaming completion from the LLM."""
        # This needs to be an async generator
        # Example: yield "chunk1"; yield "chunk2"
        if False: # pragma: no cover
             yield
        ...

    async def generate_thought(self, context: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generates a critical thought or new direction based on context."""
        ...

    async def update_analysis(self, critique: str, context: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Revises analysis based on critique and context."""
        ...

    async def evaluate_analysis(self, analysis_to_evaluate: str, context: Dict[str, Any], config: Dict[str, Any]) -> int:
        """Evaluates analysis quality (1-10 score)."""
        ...

    async def generate_tags(self, analysis_text: str, config: Dict[str, Any]) -> List[str]:
        """Generates keyword tags for the analysis."""
        ...

    async def synthesize_result(self, context: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generates a final synthesis based on the MCTS results."""
        ...

    async def classify_intent(self, text_to_classify: str, config: Dict[str, Any]) -> str:
        """Classifies user intent using the LLM."""
        ...

# ==============================================================================
# Node Class
# ==============================================================================

class Node:
    """Represents a node in the Monte Carlo Tree Search."""
    __slots__ = [
        'id', 'content', 'parent', 'children', 'visits', 'raw_scores',
        'sequence', 'is_surprising', 'surprise_explanation', 'approach_type',
        'approach_family', 'thought', 'max_children', 'use_bayesian_evaluation',
        'alpha', 'beta', 'value', 'descriptive_tags'
    ]

    def __init__(self,
                 content: str = "",
                 parent: Optional["Node"] = None,
                 sequence: int = 0,
                 thought: str = "",
                 approach_type: str = "initial",
                 approach_family: str = "general",
                 max_children: int = DEFAULT_CONFIG["max_children"],
                 use_bayesian_evaluation: bool = DEFAULT_CONFIG["use_bayesian_evaluation"],
                 beta_prior_alpha: float = DEFAULT_CONFIG["beta_prior_alpha"],
                 beta_prior_beta: float = DEFAULT_CONFIG["beta_prior_beta"],
                 **kwargs): # Allow arbitrary kwargs for flexibility if needed

        self.id = "node_" + "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=4))
        self.content = content
        self.parent = parent
        self.children = []
        self.visits = 0
        self.raw_scores = []
        self.sequence = sequence
        self.is_surprising = False
        self.surprise_explanation = ""
        self.approach_type = approach_type
        self.approach_family = approach_family
        self.thought = thought
        self.max_children = max_children
        self.use_bayesian_evaluation = use_bayesian_evaluation
        self.descriptive_tags = []

        if self.use_bayesian_evaluation:
            # Use passed priors if available in kwargs, otherwise defaults
            self.alpha = max(1e-9, float(kwargs.get("alpha", beta_prior_alpha)))
            self.beta = max(1e-9, float(kwargs.get("beta", beta_prior_beta)))
            self.value = None
        else:
            self.value = float(kwargs.get("value", 0.0))
            self.alpha = None
            self.beta = None

    def add_child(self, child: "Node") -> "Node":
        child.parent = self
        self.children.append(child)
        return child

    def fully_expanded(self) -> bool:
        # Check against actual children added, not just list length if Nones are possible
        valid_children_count = sum(1 for child in self.children if child is not None)
        return valid_children_count >= self.max_children

    def get_bayesian_mean(self) -> float:
        if self.use_bayesian_evaluation and self.alpha is not None and self.beta is not None:
            alpha_safe = max(1e-9, self.alpha)
            beta_safe = max(1e-9, self.beta)
            return alpha_safe / (alpha_safe + beta_safe) if (alpha_safe + beta_safe) > 1e-18 else 0.5
        return 0.5 # Default

    def get_average_score(self) -> float:
        """Returns score normalized to 0-10 scale."""
        if self.use_bayesian_evaluation:
            return self.get_bayesian_mean() * 10.0
        else:
            # Ensure value exists and visits are positive
            if self.value is not None and self.visits > 0:
                 # Average score per visit (assuming value is cumulative)
                 return self.value / self.visits
            elif self.value is not None and self.visits == 0:
                 # Default mid-point score
                 return 5.0
            else:
                 return 5.0 # Default mid-point score

    def thompson_sample(self) -> float:
        """Samples from the Beta distribution if Bayesian, else returns midpoint."""
        if self.use_bayesian_evaluation and self.alpha is not None and self.beta is not None:
            alpha_safe = max(1e-9, self.alpha)
            beta_safe = max(1e-9, self.beta)
            try:
                import numpy as np
                from numpy.random import beta as beta_sample
                return float(beta_sample(alpha_safe, beta_safe))
            except Exception as e:
                logger.warning(f"Thompson sample failed for node {self.sequence} (α={alpha_safe}, β={beta_safe}): {e}. Using mean.")
                return self.get_bayesian_mean()
        return 0.5 # Default

    def best_child(self) -> Optional["Node"]:
        """Finds best child based on visits then score."""
        if not self.children: return None
        valid_children = [c for c in self.children if c is not None]
        if not valid_children: return None

        # Prioritize visits
        most_visited_child = max(valid_children, key=lambda c: c.visits)
        max_visits = most_visited_child.visits
        # If no child has been visited, return None or random? Let's return None.
        if max_visits == 0: return None

        # Get all children with max visits (potential ties)
        tied_children = [c for c in valid_children if c.visits == max_visits]

        if len(tied_children) == 1:
            return tied_children[0]

        # Tie-breaking logic using score (normalized 0-10)
        if self.use_bayesian_evaluation:
            # Bayesian: Higher mean is better
             best_score_child = max(tied_children, key=lambda c: c.get_bayesian_mean())
             return best_score_child
        else:
            # Non-Bayesian: Higher average score is better
            best_score_child = max(tied_children, key=lambda c: c.get_average_score())
            return best_score_child


    def node_to_json(self) -> Dict:
        """Creates a dictionary representation for verbose output/debugging."""
        score = self.get_average_score()
        valid_children = [child for child in self.children if child is not None]
        base_json = {
            "id": self.id,
            "sequence": self.sequence,
            "content_summary": truncate_text(self.content, 150),
            "visits": self.visits,
            "approach_type": self.approach_type,
            "approach_family": self.approach_family,
            "is_surprising": self.is_surprising,
            "thought_summary": truncate_text(self.thought, 100),
            "descriptive_tags": self.descriptive_tags,
            "score": round(score, 2),
            "children": [child.node_to_json() for child in valid_children],
        }
        if self.use_bayesian_evaluation and self.alpha is not None and self.beta is not None:
            base_json["value_alpha"] = round(self.alpha, 3)
            base_json["value_beta"] = round(self.beta, 3)
            base_json["value_mean"] = round(self.get_bayesian_mean(), 3)
        elif not self.use_bayesian_evaluation and self.value is not None:
            base_json["value_cumulative"] = round(self.value, 2)
        return base_json

    def node_to_state_dict(self) -> Dict:
        """Creates a dictionary representation suitable for state persistence."""
        score = self.get_average_score()
        state = {
            "id": self.id,
            "sequence": self.sequence,
            "content_summary": truncate_text(self.content, 200),
            "visits": self.visits,
            "approach_type": self.approach_type,
            "approach_family": self.approach_family,
            "thought": self.thought,
            "descriptive_tags": self.descriptive_tags,
            "score": round(score, 2),
            "is_surprising": self.is_surprising,
        }
        if self.use_bayesian_evaluation and self.alpha is not None and self.beta is not None:
            state["alpha"] = round(self.alpha, 4)
            state["beta"] = round(self.beta, 4)
        elif not self.use_bayesian_evaluation and self.value is not None:
            state["value"] = round(self.value, 2)
        return state

    def __repr__(self) -> str:
        score = self.get_average_score()
        return (f"Node(Seq:{self.sequence}, Score:{score:.2f}, Visits:{self.visits}, "
                f"Approach:'{self.approach_type}', Children:{len(self.children)}, "
                f"Tags:{self.descriptive_tags})")

# ==============================================================================
# MCTS Class
# ==============================================================================

# Define a simple structure to hold MCTS results
MCTSResult = namedtuple("MCTSResult", ["best_score", "best_solution_content", "mcts_instance"])

class MCTS:
    """Implements the Monte Carlo Tree Search algorithm for analysis."""

    def __init__(self,
                 llm_interface: LLMInterface,
                 question: str,
                 initial_analysis_content: str,
                 config: Optional[Dict[str, Any]] = None,
                 initial_state: Optional[Dict[str, Any]] = None):
        """
        Initializes the MCTS instance.

        Args:
            llm_interface: An object implementing the LLMInterface protocol.
            question: The original user question or text to analyze.
            initial_analysis_content: The initial analysis generated by the LLM.
            config: MCTS configuration dictionary (uses DEFAULT_CONFIG if None).
            initial_state: Optional dictionary containing state loaded from a previous run.
        """
        self.llm = llm_interface
        self.question = question
        self.config = config if config is not None else DEFAULT_CONFIG.copy()
        self.debug_logging = self.config.get("debug_logging", False)
        # Update logger level based on config
        logger.setLevel(logging.DEBUG if self.debug_logging else logging.INFO)

        self.question_summary = self._summarize_text(self.question, max_words=50)
        self.loaded_initial_state = initial_state
        self.node_sequence = 0

        # Runtime state
        self.iterations_completed = 0
        self.simulations_completed = 0
        self.high_score_counter = 0 # For early stopping stability
        self.random_state = random.Random() # Use a dedicated random instance
        self.explored_approaches: Dict[str, List[str]] = {} # Track thoughts per approach type
        self.explored_thoughts: Set[str] = set() # Track unique thoughts generated
        self.approach_types: List[str] = ["initial"] # Track unique approach types encountered
        self.surprising_nodes: List[Node] = []
        self.memory: Dict[str, Any] = {"depth": 0, "branches": 0, "high_scoring_nodes": []}
        self.unfit_markers: List[Dict[str, Any]] = [] # Store unfit markers loaded/identified

        # --- Initialize Priors and Best Solution based on Loaded State ---
        prior_alpha = max(1e-9, self.config["beta_prior_alpha"])
        prior_beta = max(1e-9, self.config["beta_prior_beta"])

        # Approach Priors (for Bayesian mode)
        self.approach_alphas: Dict[str, float] = {}
        self.approach_betas: Dict[str, float] = {}
        initial_priors = self.loaded_initial_state.get("approach_priors") if self.loaded_initial_state else None
        if (self.config["use_bayesian_evaluation"] and initial_priors and
            isinstance(initial_priors.get("alpha"), dict) and
            isinstance(initial_priors.get("beta"), dict)):
            self.approach_alphas = {k: max(1e-9, v) for k, v in initial_priors["alpha"].items()}
            self.approach_betas = {k: max(1e-9, v) for k, v in initial_priors["beta"].items()}
            logger.info("Loaded approach priors from previous state.")
        else:
            # Initialize default priors for all known approaches + initial/variant
            all_approach_keys = list(APPROACH_TAXONOMY.keys()) + ["initial", "variant"]
            self.approach_alphas = {app: prior_alpha for app in all_approach_keys}
            self.approach_betas = {app: prior_beta for app in all_approach_keys}
            if self.config["use_bayesian_evaluation"]:
                 logger.info("Initialized default approach priors.")

        # Approach Scores (for non-Bayesian mode) - Simple average tracking
        self.approach_scores: Dict[str, float] = {} # Average score per approach

        # Best Solution Tracking
        self.best_score: float = 0.0
        self.best_solution: str = initial_analysis_content # Start with the initial analysis
        if self.loaded_initial_state:
            self.best_score = float(self.loaded_initial_state.get("best_score", 0.0))
            # Keep track of the previously best solution content if needed for context,
            # but the MCTS *starts* its search from the new initial_analysis_content.
            self.previous_best_solution_content = self.loaded_initial_state.get("best_solution_content")
            logger.info(f"Initialized best score ({self.best_score}) tracker from previous state.")
            # Load unfit markers
            self.unfit_markers = self.loaded_initial_state.get("unfit_markers", [])
            if self.unfit_markers:
                logger.info(f"Loaded {len(self.unfit_markers)} unfit markers from previous state.")


        # --- Initialize Root Node ---
        self.root = Node(
            content=initial_analysis_content,
            sequence=self.get_next_sequence(),
            parent=None,
            max_children=self.config["max_children"],
            use_bayesian_evaluation=self.config["use_bayesian_evaluation"],
            beta_prior_alpha=prior_alpha, # Root starts with default priors
            beta_prior_beta=prior_beta,
            approach_type="initial",
            approach_family="general"
        )
        # Initial simulation/backpropagation for the root node?
        # Not doing this in the original code, root starts with 0 visits/priors.

        logger.info(f"MCTS Initialized. Root Node Seq: {self.root.sequence}. Initial Best Score: {self.best_score:.2f}")
        if self.debug_logging:
             logger.debug(f"Initial Root Content: {truncate_text(self.root.content, 100)}")


    def get_next_sequence(self) -> int:
        """Gets the next sequential ID for a node."""
        self.node_sequence += 1
        return self.node_sequence

    def _summarize_text(self, text: str, max_words=50) -> str:
        """Summarizes text using TF-IDF (if available) or simple truncation."""
        if not text: return "N/A"
        words = re.findall(r'\w+', text)
        if len(words) <= max_words: return text.strip()

        if SKLEARN_AVAILABLE:
             try:
                sentences = re.split(r'[.!?]+\s*', text)
                sentences = [s for s in sentences if len(s.split()) > 3] # Filter short sentences
                if not sentences: return ' '.join(words[:max_words]) + '...' # Fallback

                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(sentences)
                sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

                # Select top N sentences, but ensure N is reasonable
                num_summary_sentences = max(1, min(3, len(sentences) // 5))
                top_sentence_indices = sentence_scores.argsort()[-num_summary_sentences:][::-1]
                top_sentence_indices.sort() # Keep original order

                summary = ' '.join([sentences[i] for i in top_sentence_indices])
                summary_words = summary.split()

                # Final length check
                if len(summary_words) > max_words * 1.2: # Allow slightly longer summary
                    summary = ' '.join(summary_words[:max_words]) + '...'
                # Add ellipsis if original text was longer than summary
                elif len(words) > len(summary_words):
                     summary += '...'
                return summary.strip()

             except Exception as e:
                logger.warning(f"TF-IDF summary failed ({e}). Truncating.")
                # Fall through to truncation

        # Fallback to simple truncation
        return ' '.join(words[:max_words]) + '...'

    def get_context_for_node(self, node: Node) -> Dict[str, str]:
        """
        Gathers context for LLM prompts based on the current MCTS state and loaded state.
        Ensures all values are strings.
        """
        cfg = self.config
        best_answer_str = str(self.best_solution) if self.best_solution else "N/A"

        # --- Base Context ---
        context = {
            "question_summary": self.question_summary,
            "best_answer": truncate_text(best_answer_str, 300),
            "best_score": f"{self.best_score:.1f}",
            "current_answer": truncate_text(node.content, 300),
            "current_sequence": str(node.sequence),
            "current_approach": node.approach_type,
            "current_tags": ", ".join(node.descriptive_tags) if node.descriptive_tags else "None",
            "tree_depth": str(self.memory.get("depth", 0)),
            "branches": str(self.memory.get("branches", 0)),
            "approach_types": ", ".join(self.approach_types),
            # Initialize context from loaded state with defaults
            "previous_best_summary": "N/A",
            "unfit_markers_summary": "None",
            "learned_approach_summary": "Default priors",
            "explored_approaches": "None yet.",
            "high_scoring_examples": "None yet.",
            "sibling_approaches": "", # Default empty, populated below if applicable
        }

        # --- Context from Loaded State ---
        if self.loaded_initial_state:
            context["previous_best_summary"] = self.loaded_initial_state.get("best_solution_summary", "N/A")

            unfit = self.loaded_initial_state.get("unfit_markers", [])
            if unfit:
                markers_str = "; ".join([
                    f"'{m.get('summary', m.get('id', 'Unknown'))}' ({m.get('reason', 'Unknown')})"
                    for m in unfit[:5] # Show first 5
                ])
                context["unfit_markers_summary"] = markers_str + ("..." if len(unfit) > 5 else "")
            else:
                context["unfit_markers_summary"] = "None recorded"

            priors = self.loaded_initial_state.get("approach_priors")
            if priors and "alpha" in priors and "beta" in priors:
                means = {}
                for app, alpha in priors["alpha"].items():
                    beta = priors["beta"].get(app, 1.0)
                    alpha, beta = max(1e-9, alpha), max(1e-9, beta)
                    if alpha + beta > 1e-9: means[app] = (alpha / (alpha + beta)) * 10
                sorted_means = sorted(means.items(), key=lambda item: item[1], reverse=True)
                top_approaches = [f"{app} ({score:.1f})" for app, score in sorted_means[:3]]
                context["learned_approach_summary"] = f"Favors: {', '.join(top_approaches)}" + ("..." if len(sorted_means) > 3 else "")
            else:
                 context["learned_approach_summary"] = "Priors not loaded or incomplete"


        # --- Context from Current MCTS Run ---
        try: # Explored Thought Types (using current run's data)
            if cfg["track_explored_approaches"] and self.explored_approaches:
                exp_app_text = []
                current_alphas = self.approach_alphas
                current_betas = self.approach_betas
                sorted_approach_keys = sorted(self.explored_approaches.keys())

                for app in sorted_approach_keys:
                    thoughts = self.explored_approaches.get(app, [])
                    if thoughts:
                        count = len(thoughts)
                        score_text = ""
                        if cfg["use_bayesian_evaluation"]:
                            alpha = current_alphas.get(app, 1)
                            beta = current_betas.get(app, 1)
                            alpha, beta = max(1e-9, alpha), max(1e-9, beta)
                            if (alpha + beta) > 1e-9:
                                score_text = f"(β-Mean: {alpha / (alpha + beta):.2f}, N={count})"
                            else: score_text = f"(N={count})"
                        else:
                            score = self.approach_scores.get(app, 0) # Use simple avg score
                            count_non_bayes = sum(1 for n in self._find_nodes_by_approach(app) if n.visits > 0) # More accurate count?
                            if count_non_bayes > 0:
                                 score_text = f"(Avg: {score:.1f}, N={count_non_bayes})" # Use avg score if tracked
                            else: score_text = f"(N={count})"


                        sample_count = min(2, len(thoughts))
                        sample = thoughts[-sample_count:]
                        exp_app_text.append(f"- {app} {score_text}: {'; '.join([f'{truncate_text(str(t), 50)}' for t in sample])}")
                if exp_app_text: context["explored_approaches"] = "\n".join(exp_app_text)

        except Exception as e:
            logger.error(f"Ctx err (approaches): {e}")
            context["explored_approaches"] = "Error generating approach context."

        try: # High Scoring Examples
            if self.memory["high_scoring_nodes"]:
                high_score_text = [
                    f"- Score {score:.1f} ({app}): {truncate_text(content, 70)}"
                    for score, content, app, thought in self.memory["high_scoring_nodes"]
                ]
                context["high_scoring_examples"] = "\n".join(["Top Examples:"] + high_score_text)
        except Exception as e:
            logger.error(f"Ctx err (high scores): {e}")
            context["high_scoring_examples"] = "Error generating high score context."

        try: # Sibling Context
            if cfg["sibling_awareness"] and node.parent and len(node.parent.children) > 1:
                siblings = [c for c in node.parent.children if c is not None and c != node and c.visits > 0] # Only visited siblings
                if siblings:
                    sib_app_text = []
                    sorted_siblings = sorted(siblings, key=lambda s: s.sequence)
                    for s in sorted_siblings:
                        if s.thought: # Only show siblings that generated a thought
                            score = s.get_average_score()
                            tags_str = f"Tags: [{', '.join(s.descriptive_tags)}]" if s.descriptive_tags else ""
                            sib_app_text.append(f'"{truncate_text(str(s.thought), 50)}" -> (Score: {score:.1f} {tags_str})')
                    if sib_app_text:
                        context["sibling_approaches"] = "\n".join(["Siblings:"] + [f"- {sa}" for sa in sib_app_text])
        except Exception as e:
            logger.error(f"Ctx err (siblings): {e}")
            context["sibling_approaches"] = "Error generating sibling context."

        # Ensure all values are strings for final formatting
        safe_context = {k: str(v) if v is not None else "" for k, v in context.items()}
        return safe_context


    def _calculate_uct(self, node: Node, parent_visits: int) -> float:
        """Calculates the UCT score for a node, considering penalties and bonuses."""
        cfg = self.config
        if node.visits == 0:
            return float('inf') # Prioritize unvisited nodes

        # 1. Exploitation Term (normalized 0-1)
        exploitation = node.get_bayesian_mean() if cfg["use_bayesian_evaluation"] else (node.get_average_score() / 10.0)

        # 2. Exploration Term
        log_parent_visits = math.log(max(1, parent_visits))
        exploration = cfg["exploration_weight"] * math.sqrt(log_parent_visits / node.visits)

        # 3. Penalty for Unfit Nodes (using loaded/identified markers)
        penalty = 0.0
        is_unfit = False
        if self.unfit_markers:
            node_summary = node.thought or node.content # Use thought if available, else content
            node_tags_set = set(node.descriptive_tags)
            for marker in self.unfit_markers:
                # Quick checks first
                if marker.get("id") == node.id or marker.get("sequence") == node.sequence:
                    is_unfit = True; break
                # Check tag overlap
                marker_tags_set = set(marker.get('tags', []))
                if node_tags_set and marker_tags_set and len(node_tags_set.intersection(marker_tags_set)) > 0:
                    is_unfit = True; break # Simple tag overlap check

            # Apply penalty if unfit and *not* surprising (allow surprise to override)
            if is_unfit and not node.is_surprising:
                penalty = -100.0 # Strong penalty to avoid selecting unfit nodes
                if self.debug_logging: logger.debug(f"Applying UCT penalty to unfit Node {node.sequence}")


        # 4. Surprise Bonus
        surprise_bonus = 0.3 if node.is_surprising else 0.0 # Simple fixed bonus

        # 5. Diversity Bonus (relative to siblings)
        diversity_bonus = 0.0
        if node.parent and len(node.parent.children) > 1 and cfg["score_diversity_bonus"] > 0:
            my_score_norm = node.get_average_score() / 10.0
            sibling_scores = [
                (sib.get_average_score() / 10.0)
                for sib in node.parent.children
                if sib is not None and sib != node and sib.visits > 0
            ]
            if sibling_scores:
                sibling_avg = sum(sibling_scores) / len(sibling_scores)
                diversity_bonus = cfg["score_diversity_bonus"] * abs(my_score_norm - sibling_avg)

        # Combine terms
        uct_value = exploitation + exploration + surprise_bonus + diversity_bonus + penalty
        # Ensure finite return, default to low value if not
        return uct_value if math.isfinite(uct_value) else -float('inf')

    def _collect_non_leaf_nodes(self, node: Node, non_leaf_nodes: List[Node], max_depth: int, current_depth: int = 0):
        """Helper to find nodes that can still be expanded within a depth limit."""
        if current_depth > max_depth or node is None:
            return
        # Node is non-leaf if it HAS children AND is not fully expanded yet
        if node.children and not node.fully_expanded():
            non_leaf_nodes.append(node)

        for child in node.children:
             # Recursive call only if child exists
            if child is not None:
                self._collect_non_leaf_nodes(child, non_leaf_nodes, max_depth, current_depth + 1)


    async def select(self) -> Node:
        """Selects a node for expansion using UCT or Thompson Sampling."""
        cfg = self.config
        node = self.root
        selection_path_ids = [node.id] # Track path by ID

        # Optional: Branch Enhancement (Force exploration of less visited branches)
        force_interval = cfg["force_exploration_interval"]
        if (force_interval > 0 and self.simulations_completed > 0 and
            self.simulations_completed % force_interval == 0 and self.memory["depth"] > 1):
            candidate_nodes = []
            # Collect expandable nodes up to half the current max depth
            self._collect_non_leaf_nodes(self.root, candidate_nodes, max_depth=max(1, self.memory["depth"] // 2))
            expandable_candidates = [n for n in candidate_nodes if not n.fully_expanded()]
            if expandable_candidates:
                forced_node = self.random_state.choice(expandable_candidates)
                if self.debug_logging: logger.debug(f"BRANCH ENHANCE: Forcing selection of Node {forced_node.sequence}")
                # Need to return the actual node selected by force
                return forced_node # Exit selection early with the forced node

        # Standard Selection Loop
        while node.children: # While the node has children listed
            valid_children = [child for child in node.children if child is not None]
            if not valid_children:
                 logger.warning(f"Node {node.sequence} has empty children list or only None entries. Stopping selection.")
                 break # Cannot proceed

            parent_visits = node.visits
            unvisited = [child for child in valid_children if child.visits == 0]

            if unvisited:
                selected_child = self.random_state.choice(unvisited)
                node = selected_child # Move to the unvisited child
                selection_path_ids.append(node.id)
                break # Stop selection here, this node will be expanded/simulated

            # If all children visited, use selection strategy
            if cfg["use_thompson_sampling"] and cfg["use_bayesian_evaluation"]:
                # Thompson Sampling
                samples = [(child, child.thompson_sample()) for child in valid_children]
                if not samples:
                    logger.warning(f"No valid Thompson samples for children of {node.sequence}. Selecting randomly.")
                    selected_child = self.random_state.choice(valid_children)
                else:
                    selected_child, _ = max(samples, key=lambda x: x[1])
                node = selected_child
            else:
                # UCT Selection
                uct_values = []
                for child in valid_children:
                     try:
                         uct = self._calculate_uct(child, parent_visits)
                         if math.isfinite(uct): uct_values.append((child, uct))
                         else: logger.warning(f"UCT for child {child.sequence} was non-finite. Skipping.")
                     except Exception as uct_err:
                         logger.error(f"UCT calculation error for node {child.sequence}: {uct_err}")

                if not uct_values:
                    logger.warning(f"No valid UCT values for children of {node.sequence}. Selecting randomly.")
                    if not valid_children: # Should not happen if loop condition met, but safety check
                         logger.error(f"Selection error: Node {node.sequence} has no valid children. Cannot proceed.")
                         return node # Return current node as selection cannot advance
                    selected_child = self.random_state.choice(valid_children)
                else:
                    uct_values.sort(key=lambda x: x[1], reverse=True) # Highest UCT wins
                    selected_child = uct_values[0][0]
                node = selected_child

            selection_path_ids.append(node.id) # Add selected node to path

            # If the newly selected node is not fully expanded, stop selection (it's the target)
            # Or if it has no children (it's a leaf node)
            if not node.fully_expanded() or not node.children:
                break

        # Update max depth seen
        current_depth = len(selection_path_ids) - 1
        self.memory["depth"] = max(self.memory.get("depth", 0), current_depth)
        if self.debug_logging:
             path_seq = [self._find_node_by_id(nid).sequence if self._find_node_by_id(nid) else '?' for nid in selection_path_ids]
             logger.debug(f"Selection path (Sequences): {' -> '.join(map(str, path_seq))}")

        return node # Return the selected leaf or expandable node


    def _classify_approach(self, thought: str) -> Tuple[str, str]:
        """Classifies the thought into an approach type and family using keywords."""
        approach_type = "variant" # Default if no keywords match
        approach_family = "general"
        if not thought or not isinstance(thought, str):
            return approach_type, approach_family

        thought_lower = thought.lower()
        approach_scores = {
            app: sum(1 for kw in kws if kw in thought_lower)
            for app, kws in APPROACH_TAXONOMY.items() if kws # Check if keywords exist
        }
        positive_scores = {app: score for app, score in approach_scores.items() if score > 0}

        if positive_scores:
            max_score = max(positive_scores.values())
            # Handle ties by random choice among best
            best_approaches = [app for app, score in positive_scores.items() if score == max_score]
            approach_type = self.random_state.choice(best_approaches)

        # Get family from metadata
        approach_family = APPROACH_METADATA.get(approach_type, {}).get("family", "general")

        if self.debug_logging:
            logger.debug(f"Classified thought '{truncate_text(thought, 50)}' as: {approach_type} ({approach_family})")
        return approach_type, approach_family

    def _check_surprise(self, parent_node: Node, new_content: str, new_approach_type: str, new_approach_family: str) -> Tuple[bool, str]:
        """Checks if the new node content/approach is surprising relative to the parent."""
        cfg = self.config
        surprise_factors = []
        is_surprising = False
        surprise_explanation = ""

        # 1. Semantic Distance Check
        if cfg["use_semantic_distance"]:
            try:
                parent_content = str(parent_node.content) if parent_node.content else ""
                new_content_str = str(new_content) if new_content else ""
                if parent_content and new_content_str:
                    dist = calculate_semantic_distance(parent_content, new_content_str, use_tfidf=True) # Can disable TFIDF here if too slow
                    if dist > cfg["surprise_threshold"]:
                        surprise_factors.append({
                            "type": "semantic", "value": dist,
                            "weight": cfg["surprise_semantic_weight"],
                            "desc": f"Semantic dist ({dist:.2f})"
                        })
            except Exception as e: logger.warning(f"Semantic distance check failed: {e}")

        # 2. Shift in Thought Approach Family
        parent_family = parent_node.approach_family
        if parent_family != new_approach_family and new_approach_family != "general":
            surprise_factors.append({
                "type": "family_shift", "value": 1.0,
                "weight": cfg["surprise_philosophical_shift_weight"],
                "desc": f"Shift '{parent_family}'->'{new_approach_family}'"
            })

        # 3. Novelty of Thought Approach Family (using BFS on current tree)
        try:
            family_counts = Counter()
            queue = [(self.root, 0)] if self.root else []
            processed_bfs = set()
            nodes_visited = 0
            MAX_BFS_NODES = 100
            MAX_BFS_DEPTH = 5

            while queue and nodes_visited < MAX_BFS_NODES:
                curr_node, depth = queue.pop(0)
                if curr_node is None or curr_node.id in processed_bfs or depth > MAX_BFS_DEPTH:
                    continue
                processed_bfs.add(curr_node.id)
                nodes_visited += 1
                family_counts[curr_node.approach_family] += 1
                if depth + 1 <= MAX_BFS_DEPTH:
                    queue.extend([(child, depth + 1) for child in curr_node.children if child is not None])

            # If the new family has been seen <= 1 times (itself) and isn't 'general'
            if family_counts.get(new_approach_family, 0) <= 1 and new_approach_family != "general":
                surprise_factors.append({
                    "type": "novelty", "value": 0.8, # Slightly less value than shift/semantic maybe?
                    "weight": cfg["surprise_novelty_weight"],
                    "desc": f"Novel approach family ('{new_approach_family}')"
                })
        except Exception as e: logger.warning(f"Novelty check BFS failed: {e}", exc_info=self.debug_logging)


        # Calculate combined weighted score
        if surprise_factors:
            total_weighted_score = sum(f["value"] * f["weight"] for f in surprise_factors)
            total_weight = sum(f["weight"] for f in surprise_factors)
            combined_score = (total_weighted_score / total_weight) if total_weight > 1e-6 else 0.0

            if combined_score >= cfg["surprise_overall_threshold"]:
                is_surprising = True
                factor_descs = [f"- {f['desc']} (Val:{f['value']:.2f}, W:{f['weight']:.1f})" for f in surprise_factors]
                surprise_explanation = (f"Combined surprise ({combined_score:.2f} >= {cfg['surprise_overall_threshold']}):\n" + "\n".join(factor_descs))
                if self.debug_logging: logger.debug(f"Surprise DETECTED for node sequence {parent_node.sequence+1}: Score={combined_score:.2f}\n{surprise_explanation}")

        return is_surprising, surprise_explanation


    async def expand(self, node: Node) -> Optional[Node]:
        """Expands a node by generating a thought and a new analysis."""
        cfg = self.config
        if node.fully_expanded():
             logger.warning(f"Attempted to expand fully expanded Node {node.sequence}. Returning None.")
             return None
        if not node.content:
             logger.warning(f"Attempted to expand Node {node.sequence} with no content. Returning None.")
             return None

        try:
            context = self.get_context_for_node(node)

            # 1. Generate Thought
            if self.debug_logging: logger.debug(f"Generating thought for Node {node.sequence}")
            thought = await self.llm.generate_thought(context, cfg)
            if not isinstance(thought, str) or not thought.strip() or "Error:" in thought:
                logger.error(f"Invalid thought generation for Node {node.sequence}: '{thought}'")
                return None # Expansion failed
            thought = thought.strip()
            if self.debug_logging: logger.debug(f"Node {node.sequence} -> Thought: '{truncate_text(thought, 80)}'")

            # Check thought against unfit markers (simple check)
            is_unfit_thought = False
            if self.unfit_markers:
                 for marker in self.unfit_markers:
                      marker_summary = marker.get('summary')
                      # if marker_summary and calculate_semantic_distance(thought, marker_summary) < 0.15: # Strict threshold
                      #      is_unfit_thought = True
                      #      logger.warning(f"Generated thought for Node {node.sequence} resembles unfit marker '{marker_summary}'. Skipping expansion.")
                      #      break
            if is_unfit_thought: return None # Skip expansion if thought seems unfit

            # Classify approach based on thought
            approach_type, approach_family = self._classify_approach(thought)
            self.explored_thoughts.add(thought)
            if approach_type not in self.approach_types: self.approach_types.append(approach_type)
            if approach_type not in self.explored_approaches: self.explored_approaches[approach_type] = []
            self.explored_approaches[approach_type].append(thought)


            # 2. Update Analysis based on Thought
            if self.debug_logging: logger.debug(f"Updating analysis for Node {node.sequence} based on thought")
            # Pass original content in context for update prompt
            context_for_update = context.copy()
            context_for_update['answer'] = node.content # Use 'answer' key as expected by UPDATE_PROMPT
            context_for_update['improvements'] = thought # Use 'improvements' key
            new_content = await self.llm.update_analysis(thought, context_for_update, cfg)

            if not isinstance(new_content, str) or not new_content.strip() or "Error:" in new_content:
                logger.error(f"Invalid new content generation for Node {node.sequence}: '{new_content}'")
                return None # Expansion failed
            new_content = new_content.strip()
            if self.debug_logging: logger.debug(f"Node {node.sequence} -> New Content: '{truncate_text(new_content, 80)}'")

            # 3. Generate Tags for New Content
            new_tags = await self.llm.generate_tags(new_content, cfg)
            if self.debug_logging: logger.debug(f"Generated Tags for new node: {new_tags}")

            # 4. Check for Surprise
            is_surprising, surprise_explanation = self._check_surprise(node, new_content, approach_type, approach_family)

            # 5. Create Child Node
            child = Node(
                content=new_content,
                parent=node,
                sequence=self.get_next_sequence(),
                thought=thought,
                approach_type=approach_type,
                approach_family=approach_family,
                max_children=cfg["max_children"],
                use_bayesian_evaluation=cfg["use_bayesian_evaluation"],
                beta_prior_alpha=cfg["beta_prior_alpha"], # Child starts with default priors
                beta_prior_beta=cfg["beta_prior_beta"]
            )
            child.descriptive_tags = new_tags
            child.is_surprising = is_surprising
            child.surprise_explanation = surprise_explanation

            # Add child to parent
            node.add_child(child)
            if is_surprising: self.surprising_nodes.append(child)

            # Update branch count if this adds a new branch
            if len(node.children) > 1: self.memory["branches"] += 1

            if self.debug_logging: logger.debug(f"Successfully expanded Node {node.sequence} -> Child {child.sequence}")
            return child

        except Exception as e:
            logger.error(f"Expand error on Node {node.sequence}: {e}", exc_info=self.debug_logging)
            return None


    async def simulate(self, node: Node) -> Optional[float]:
        """Simulates (evaluates) a node using the LLM, returns score (1-10)."""
        cfg = self.config
        if not node.content:
            logger.warning(f"Cannot simulate Node {node.sequence}: Content is empty. Returning default score 5.0")
            return 5.0

        try:
            context = self.get_context_for_node(node)
            # Ensure context has the key expected by the eval prompt
            context['answer_to_evaluate'] = node.content

            if self.debug_logging: logger.debug(f"Evaluating Node {node.sequence}")
            raw_score = await self.llm.evaluate_analysis(node.content, context, cfg)

            # Validate score is int 1-10
            if not isinstance(raw_score, int) or not (1 <= raw_score <= 10):
                logger.error(f"Evaluation for Node {node.sequence} returned invalid score: {raw_score}. Defaulting to 5.")
                raw_score = 5

            score = float(raw_score)
            node.raw_scores.append(raw_score) # Keep track of raw scores received
            approach = node.approach_type if node.approach_type else "unknown"

            # Update approach performance tracking
            if cfg["use_bayesian_evaluation"]:
                 # Use raw score for pseudo counts (scale 1-10)
                 pseudo_successes = max(0, raw_score - 1) # 10 -> 9 successes
                 pseudo_failures = max(0, 10 - raw_score) # 3 -> 7 failures
                 # Ensure approach exists in prior dicts, initializing if necessary
                 current_alpha = self.approach_alphas.setdefault(approach, cfg["beta_prior_alpha"])
                 current_beta = self.approach_betas.setdefault(approach, cfg["beta_prior_beta"])
                 # Update priors safely
                 self.approach_alphas[approach] = max(1e-9, current_alpha + pseudo_successes)
                 self.approach_betas[approach] = max(1e-9, current_beta + pseudo_failures)
            else:
                 # Non-Bayesian: Update simple average score (e.g., using EMA)
                 current_avg = self.approach_scores.get(approach, score) # Initialize with current score if first time
                 self.approach_scores[approach] = 0.7 * score + 0.3 * current_avg # EMA update

            if self.debug_logging: logger.debug(f"Node {node.sequence} evaluation result: {score:.1f}/10")

            # Update high score memory (use score 1-10)
            if score >= 7:
                entry = (score, node.content, approach, node.thought)
                self.memory["high_scoring_nodes"].append(entry)
                # Sort and trim memory
                self.memory["high_scoring_nodes"].sort(key=lambda x: x[0], reverse=True)
                self.memory["high_scoring_nodes"] = self.memory["high_scoring_nodes"][:cfg["memory_cutoff"]]

            return score

        except Exception as e:
            logger.error(f"Simulate error for Node {node.sequence}: {e}", exc_info=self.debug_logging)
            return None # Indicate simulation failure

    def backpropagate(self, node: Node, score: float):
        """Backpropagates the simulation score up the tree."""
        cfg = self.config
        if score is None or not math.isfinite(score):
             logger.warning(f"Invalid score ({score}) received for backpropagation from Node {node.sequence}. Skipping.")
             return

        if self.debug_logging: logger.debug(f"Backpropagating score {score:.2f} from Node {node.sequence}")

        # Normalize score to 0-1 for Bayesian updates if score is 1-10
        normalized_score = score / 10.0
        pseudo_successes = max(0, score - 1) # Use 1-10 score for pseudo counts
        pseudo_failures = max(0, 10 - score)

        temp_node: Optional[Node] = node
        path_len = 0
        while temp_node:
            temp_node.visits += 1
            if cfg["use_bayesian_evaluation"]:
                 if temp_node.alpha is not None and temp_node.beta is not None:
                     # Update using pseudo counts from 1-10 score
                     temp_node.alpha = max(1e-9, temp_node.alpha + pseudo_successes)
                     temp_node.beta = max(1e-9, temp_node.beta + pseudo_failures)
                 else: logger.warning(f"Node {temp_node.sequence} missing alpha/beta during backprop.")
            else: # Non-Bayesian: Add score to cumulative value
                 if temp_node.value is not None:
                     temp_node.value += score # Add the raw score (1-10)
                 else: # Initialize if missing (should only happen for root if not pre-simulated)
                      logger.warning(f"Node {temp_node.sequence} missing value during non-Bayesian backprop. Initializing.")
                      temp_node.value = score

            temp_node = temp_node.parent
            path_len += 1

        if self.debug_logging: logger.debug(f"Backpropagation complete for Node {node.sequence} (Path length: {path_len})")

    async def run_search_iterations(self, num_iterations: int, simulations_per_iteration: int) -> None:
        """Runs the main MCTS search loop."""
        cfg = self.config
        logger.info(f"Starting MCTS search: {num_iterations} iterations, {simulations_per_iteration} simulations/iter.")

        # Performance optimization - run multiple simulations concurrently
        max_concurrent = 3  # Set a reasonable limit for concurrency
        
        for i in range(num_iterations):
            self.iterations_completed = i + 1
            logger.info(f"--- Starting Iteration {self.iterations_completed}/{num_iterations} ---")
            best_score_before_iter = self.best_score

            # Process simulations in batches for better concurrency
            for batch_start in range(0, simulations_per_iteration, max_concurrent):
                batch_size = min(max_concurrent, simulations_per_iteration - batch_start)
                batch_tasks = []
                
                # Create tasks for the batch
                for j in range(batch_start, batch_start + batch_size):
                    sim_num = j + 1
                    task = asyncio.create_task(self._run_single_simulation(sim_num, simulations_per_iteration))
                    batch_tasks.append(task)
                
                # Wait for the batch to complete
                await asyncio.gather(*batch_tasks)
                
                # Check early stopping after each batch
                if (cfg["early_stopping"] and
                    self.best_score >= cfg["early_stopping_threshold"] and
                    self.high_score_counter >= cfg["early_stopping_stability"]):
                    logger.info(f"EARLY STOPPING criteria met during Iteration {self.iterations_completed}.")
                    return  # Exit early
            
            # --- End of Simulations for Iteration i ---
            logger.info(f"--- Finished Iteration {self.iterations_completed}. Current Best Score: {self.best_score:.2f} ---")

            # Re-check early stopping condition after the iteration
            if (cfg["early_stopping"] and
                self.best_score >= cfg["early_stopping_threshold"] and
                self.high_score_counter >= cfg["early_stopping_stability"]):
                logger.info(f"EARLY STOPPING criteria met at end of Iteration {self.iterations_completed}.")
                break  # Exit outer iteration loop

        logger.info("MCTS search finished.")
    
    async def _run_single_simulation(self, current_sim_num: int, total_sims: int) -> None:
        """Runs a single simulation (select-expand-simulate-backpropagate cycle)."""
        self.simulations_completed += 1
        cfg = self.config
        
        if self.debug_logging: 
            logger.debug(f"--- Sim {current_sim_num}/{total_sims} ---")

        # 1. Select
        leaf = await self.select()
        if not leaf:
            logger.error(f"Sim {current_sim_num}: Selection returned None. Skipping simulation.")
            return

        # 2. Expand (if not terminal and not fully expanded)
        node_to_simulate = leaf
        if not leaf.fully_expanded() and leaf.content:  # Check content exists
            if self.debug_logging: 
                logger.debug(f"Sim {current_sim_num}: Attempting expansion from Node {leaf.sequence}")
            expanded_node = await self.expand(leaf)
            if expanded_node:
                node_to_simulate = expanded_node  # Simulate the newly expanded node
                if self.debug_logging: 
                    logger.debug(f"Sim {current_sim_num}: Expanded {leaf.sequence} -> {node_to_simulate.sequence}")
            else:
                if self.debug_logging: 
                    logger.warning(f"Sim {current_sim_num}: Expansion failed for {leaf.sequence}. Simulating original leaf.")
                node_to_simulate = leaf  # Simulate original leaf if expansion failed
        elif self.debug_logging:
            logger.debug(f"Sim {current_sim_num}: Node {leaf.sequence} is fully expanded or has no content. Simulating directly.")

        # 3. Simulate
        score = None
        if node_to_simulate and node_to_simulate.content:
            score = await self.simulate(node_to_simulate)
        elif node_to_simulate:
            logger.warning(f"Sim {current_sim_num}: Skipping simulation for {node_to_simulate.sequence} (no content).")
            score = 5.0  # Assign default score
        else:  # Should not happen if selection worked
            logger.error(f"Sim {current_sim_num}: node_to_simulate is None after select/expand. Skipping simulation.")
            return  # Skip backprop

        # 4. Backpropagate
        if score is not None:
            self.backpropagate(node_to_simulate, score)

            # Update overall best score/solution found so far
            if score > self.best_score:
                logger.info(f"Sim {current_sim_num}: ✨ New best! Score: {score:.1f} (Node {node_to_simulate.sequence})")
                self.best_score = score
                self.best_solution = str(node_to_simulate.content)
                self.high_score_counter = 0  # Reset stability counter
            elif score == self.best_score:
                # If score matches best, don't reset counter
                pass
            else:  # Score is lower than best
                self.high_score_counter = 0  # Reset stability counter if score drops

            # Check early stopping (threshold) - based on overall best score
            if cfg["early_stopping"] and self.best_score >= cfg["early_stopping_threshold"]:
                self.high_score_counter += 1  # Increment counter only if score >= threshold
                if self.debug_logging: 
                    logger.debug(f"Sim {current_sim_num}: Best score ({self.best_score:.1f}) >= threshold. Stability: {self.high_score_counter}/{cfg['early_stopping_stability']}")
        else:  # Simulation failed (score is None)
            if node_to_simulate: 
                logger.warning(f"Sim {current_sim_num}: Simulation failed for Node {node_to_simulate.sequence}. No score obtained.")
            self.high_score_counter = 0  # Reset stability counter if sim fails


    def get_final_results(self) -> MCTSResult:
        """Returns the best score and solution found."""
        # Clean the best solution content of <think> tags if present
        cleaned_solution = self.best_solution
        if cleaned_solution and isinstance(cleaned_solution, str):
            # First try to remove the entire <think> block if it's a pure think block
            clean_attempt = re.sub(r'<think>.*?</think>', '', cleaned_solution, flags=re.DOTALL)
            # If that removes everything, keep the original but strip just the tags
            if not clean_attempt.strip() and ("<think>" in cleaned_solution or "</think>" in cleaned_solution):
                cleaned_solution = re.sub(r'</?think>', '', cleaned_solution)
            else:
                cleaned_solution = clean_attempt
            
        # In a real app, you might want more detailed results (e.g., best node path)
        return MCTSResult(
            best_score=self.best_score,
            best_solution_content=cleaned_solution.strip() if isinstance(cleaned_solution, str) else cleaned_solution,
            mcts_instance=self # Return self for further analysis if needed
        )

    def find_best_final_node(self) -> Optional[Node]:
        """Finds the node object corresponding to the best solution content."""
        if not self.best_solution or not self.root: return None

        queue = [self.root]
        visited_ids = {self.root.id}
        best_match_node = None
        min_score_diff = float('inf')

        # Clean target solution content once
        target_content = str(self.best_solution).strip()
        target_content = re.sub(r"^```(json|markdown)?\s*", "", target_content, flags=re.IGNORECASE | re.MULTILINE)
        target_content = re.sub(r"\s*```$", "", target_content, flags=re.MULTILINE).strip()

        while queue:
            current = queue.pop(0)
            if current is None: continue

            # Clean node content for comparison
            node_content = str(current.content).strip()
            node_content = re.sub(r"^```(json|markdown)?\s*", "", node_content, flags=re.IGNORECASE | re.MULTILINE)
            node_content = re.sub(r"\s*```$", "", node_content, flags=re.MULTILINE).strip()

            # Check for exact content match (after cleaning)
            if node_content == target_content:
                score_diff = abs(current.get_average_score() - self.best_score)
                # Prefer node with score closest to the recorded best score
                if best_match_node is None or score_diff < min_score_diff:
                    best_match_node = current
                    min_score_diff = score_diff
                    # Optimization: If perfect match found, can stop early? Only if content is guaranteed unique.
                    # Let's keep searching to ensure we find the one with the closest score if duplicates exist.

            # Add valid children to queue
            for child in current.children:
                if child and child.id not in visited_ids:
                    visited_ids.add(child.id)
                    queue.append(child)

        if not best_match_node:
            logger.warning("Could not find node object exactly matching best solution content. Best score might be from a pruned or non-existent node state.")
            # Fallback: Find node with highest score overall?
            # best_overall_node = self._find_node_with_highest_score()
            # return best_overall_node
            # For now, return None if exact content match fails.
            return None

        return best_match_node

    def _find_node_by_id(self, node_id: str) -> Optional[Node]:
        """Finds a node by its ID using BFS."""
        if not self.root: return None
        queue = [self.root]
        visited = {self.root.id}
        while queue:
            current = queue.pop(0)
            if current.id == node_id:
                return current
            for child in current.children:
                if child and child.id not in visited:
                     visited.add(child.id)
                     queue.append(child)
        return None

    def _find_nodes_by_approach(self, approach_type: str) -> List[Node]:
        """Finds all nodes with a specific approach type using BFS."""
        nodes = []
        if not self.root: return nodes
        queue = [self.root]
        visited = {self.root.id}
        while queue:
             current = queue.pop(0)
             if current.approach_type == approach_type:
                  nodes.append(current)
             for child in current.children:
                  if child and child.id not in visited:
                       visited.add(child.id)
                       queue.append(child)
        return nodes


    def export_tree_summary(self) -> Dict:
         """Exports a summary of the tree structure and key nodes."""
         if not self.root: return {"error": "No root node"}
         return self.root.node_to_json() # Use the recursive JSON export

    def get_best_path_nodes(self) -> List[Node]:
         """Traces the path from the root to the best scoring node found."""
         best_node = self.find_best_final_node()
         if not best_node: return []
         path = []
         current = best_node
         while current:
              path.append(current)
              current = current.parent
         return path[::-1] # Reverse to get root -> best order

# ==============================================================================
# State Management
# ==============================================================================

class StateManager:
    """Handles saving and loading MCTS state to/from a SQLite database."""

    def __init__(self, db_file_path: str):
        self.db_file = db_file_path
        self._ensure_db_and_table()

    def _get_db_connection(self) -> Optional[sqlite3.Connection]:
        """Establishes connection to the SQLite DB."""
        conn = None
        try:
            db_dir = os.path.dirname(self.db_file)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
                logger.info(f"Created directory for database: {db_dir}")
            conn = sqlite3.connect(self.db_file, timeout=10) # Add timeout
            conn.execute("PRAGMA journal_mode=WAL;")  # Improve concurrency
            conn.execute("PRAGMA busy_timeout = 5000;") # Wait 5s if locked
            return conn
        except sqlite3.Error as e:
            logger.error(f"SQLite error connecting to {self.db_file}: {e}", exc_info=True)
            if conn: conn.close()
            return None
        except Exception as e:
            logger.error(f"Unexpected error during DB connection setup {self.db_file}: {e}", exc_info=True)
            if conn: conn.close()
            return None

    def _ensure_db_and_table(self):
        """Ensures the database file and the state table exist."""
        conn = self._get_db_connection()
        if not conn:
            logger.error("Cannot ensure DB table exists: Failed to get DB connection.")
            return
        try:
            with conn: # Use context manager for commit/rollback
                 conn.execute("""
                    CREATE TABLE IF NOT EXISTS mcts_state (
                        chat_id TEXT PRIMARY KEY,
                        last_state_json TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                 """)
                 # Optional: Index for faster lookups
                 conn.execute("CREATE INDEX IF NOT EXISTS idx_mcts_state_timestamp ON mcts_state (timestamp);")
            logger.debug(f"Database table 'mcts_state' ensured in {self.db_file}")
        except sqlite3.Error as e:
            logger.error(f"SQLite error creating table in {self.db_file}: {e}", exc_info=True)
        finally:
            if conn: conn.close()

    def save_state(self, chat_id: str, mcts_instance: MCTS):
        """Serializes MCTS state and saves it to the database."""
        if not chat_id:
            logger.warning("Cannot save state: chat_id is missing.")
            return
        if not mcts_instance:
            logger.warning("Cannot save state: MCTS instance is invalid.")
            return

        state_json = self._serialize_mcts_state(mcts_instance)
        if state_json == "{}":
             logger.warning("Serialization produced empty state, not saving.")
             return

        conn = self._get_db_connection()
        if not conn:
            logger.error("Cannot save state: Failed to get DB connection.")
            return
        try:
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO mcts_state (chat_id, last_state_json, timestamp) VALUES (?, ?, ?)",
                    (chat_id, state_json, datetime.now())
                )
            logger.info(f"Saved MCTS state for chat_id: {chat_id}")
        except sqlite3.Error as e:
            logger.error(f"SQLite error saving state for chat_id {chat_id}: {e}", exc_info=True)
        finally:
            if conn: conn.close()

    def load_state(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Loads and deserializes the latest MCTS state for a chat_id."""
        if not chat_id:
            logger.warning("Cannot load state: chat_id is missing.")
            return None

        conn = self._get_db_connection()
        if not conn:
            logger.error("Cannot load state: Failed to get DB connection.")
            return None

        state_dict = None
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT last_state_json FROM mcts_state WHERE chat_id = ? ORDER BY timestamp DESC LIMIT 1",
                    (chat_id,)
                )
                result = cursor.fetchone()
            if result and result[0]:
                try:
                    loaded_data = json.loads(result[0])
                    # Basic validation (can be extended)
                    if isinstance(loaded_data, dict) and loaded_data.get("version") == "0.8.0":
                         state_dict = loaded_data
                         logger.info(f"Loaded and validated MCTS state for chat_id: {chat_id}")
                    else:
                         logger.warning(f"Loaded state for {chat_id} is invalid or wrong version ({loaded_data.get('version')}). Discarding.")
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding loaded state JSON for chat_id {chat_id}: {e}")
            else:
                 logger.info(f"No previous MCTS state found for chat_id: {chat_id}")
        except sqlite3.Error as e:
            logger.error(f"SQLite error loading state for chat_id {chat_id}: {e}", exc_info=True)
        finally:
            if conn: conn.close()
        return state_dict

    def _serialize_mcts_state(self, mcts_instance: MCTS) -> str:
        """Extracts selective state from MCTS and returns as JSON string."""
        if not mcts_instance or not mcts_instance.root:
            logger.warning("Attempted to serialize empty or invalid MCTS instance.")
            return "{}"

        state = {"version": "0.8.0"} # Version state format

        try:
            cfg = mcts_instance.config
            # Basic info
            state["best_score"] = mcts_instance.best_score
            best_node = mcts_instance.find_best_final_node()
            state["best_solution_summary"] = truncate_text(mcts_instance.best_solution, 400)
            state["best_solution_content"] = str(mcts_instance.best_solution) # Save full best content
            state["best_node_tags"] = best_node.descriptive_tags if best_node else []

            # Learned Priors (if Bayesian)
            if cfg.get("use_bayesian_evaluation"):
                alphas = getattr(mcts_instance, "approach_alphas", {})
                betas = getattr(mcts_instance, "approach_betas", {})
                if isinstance(alphas, dict) and isinstance(betas, dict):
                    state["approach_priors"] = {
                        "alpha": {k: round(v, 4) for k, v in alphas.items()},
                        "beta": {k: round(v, 4) for k, v in betas.items()}
                    }
                else: state["approach_priors"] = None

            # Unfit Markers (simple approach based on score/visits)
            unfit_markers = []
            score_thresh = cfg.get("unfit_score_threshold", DEFAULT_CONFIG["unfit_score_threshold"])
            visit_thresh = cfg.get("unfit_visit_threshold", DEFAULT_CONFIG["unfit_visit_threshold"])

            # Collect all nodes first (BFS)
            all_nodes = []
            queue = [mcts_instance.root] if mcts_instance.root else []
            visited_ids = {mcts_instance.root.id} if mcts_instance.root else set()
            while queue:
                 current = queue.pop(0)
                 if not current: continue
                 all_nodes.append(current) # Add node itself
                 for child in current.children:
                      if child and child.id not in visited_ids:
                           visited_ids.add(child.id)
                           queue.append(child)

            # Now check collected nodes for unfitness
            for node in all_nodes:
                 if node.visits >= visit_thresh and node.get_average_score() < score_thresh:
                      marker = {
                          "id": node.id,
                          "sequence": node.sequence,
                          "summary": truncate_text(node.thought or node.content, 80),
                          "reason": f"Low score ({node.get_average_score():.1f} < {score_thresh}) after {node.visits} visits",
                          "tags": node.descriptive_tags,
                      }
                      unfit_markers.append(marker)
            state["unfit_markers"] = unfit_markers[:10] # Limit number saved

            # Optional: Save top N nodes' state_dict for potential warm start?
            # sorted_nodes = sorted(all_nodes, key=lambda n: n.get_average_score(), reverse=True)
            # state["top_nodes_state"] = [node.node_to_state_dict() for node in sorted_nodes[:3]]

            return json.dumps(state)

        except Exception as e:
            logger.error(f"Error during MCTS state serialization: {e}", exc_info=True)
            return "{}" # Return empty JSON on error


# ==============================================================================
# Intent Handling
# ==============================================================================

# Define result structures for handlers
IntentResult = namedtuple("IntentResult", ["type", "data"]) # type = 'message', 'error', 'mcts_params'

class IntentHandler:
    """Handles different user intents based on classification."""

    def __init__(self, llm_interface: LLMInterface, state_manager: StateManager, config: Dict[str, Any]):
        self.llm = llm_interface
        self.state_manager = state_manager
        self.config = config # Store current config

    async def classify_intent(self, user_input: str) -> str:
        """Classifies user intent using the LLM."""
        # Use the LLM interface's classification method
        try:
            classification_result = await self.llm.classify_intent(user_input, self.config)
            # Basic validation against known intents
            valid_intents = [
                "ANALYZE_NEW", "CONTINUE_ANALYSIS", "ASK_LAST_RUN_SUMMARY",
                "ASK_PROCESS", "ASK_CONFIG", "GENERAL_CONVERSATION"
            ]
            clean_result = classification_result.strip().upper().split()[0] if classification_result else ""
            clean_result = re.sub(r"[.,!?;:]$", "", clean_result) # Remove trailing punctuation

            if clean_result in valid_intents:
                logger.info(f"Intent classified as: {clean_result}")
                return clean_result
            else:
                logger.warning(f"Intent classification returned unexpected result: '{classification_result}'. Defaulting to ANALYZE_NEW.")
                # Simple keyword check as fallback
                if any(keyword in user_input.lower() for keyword in ["continue", "elaborate", "further", "what about", "refine"]):
                    logger.info("Input text suggests continuation despite classification. Setting intent to CONTINUE_ANALYSIS.")
                    return "CONTINUE_ANALYSIS"
                return "ANALYZE_NEW"
        except Exception as e:
             logger.error(f"Intent classification LLM call failed: {e}", exc_info=True)
             return "ANALYZE_NEW" # Default on error

    async def handle_intent(self, intent: str, user_input: str, chat_id: Optional[str]) -> IntentResult:
        """Dispatches to the appropriate handler based on intent."""

        loaded_state = None
        if chat_id and intent in ["CONTINUE_ANALYSIS", "ASK_LAST_RUN_SUMMARY"]:
             loaded_state = self.state_manager.load_state(chat_id)
             # If trying to continue but no state exists, switch intent to ANALYZE_NEW
             if intent == "CONTINUE_ANALYSIS" and not loaded_state:
                  logger.info(f"Cannot continue analysis for chat {chat_id}: No state found. Switching to ANALYZE_NEW.")
                  intent = "ANALYZE_NEW"
                  # Return an info message?
                  # return IntentResult(type='message', data="No previous analysis state found. Starting a new analysis.")


        if intent == "ASK_PROCESS":
            return await self.handle_ask_process()
        elif intent == "ASK_CONFIG":
            return await self.handle_ask_config()
        elif intent == "ASK_LAST_RUN_SUMMARY":
             # Ensure state was loaded or handle gracefully
             if not loaded_state:
                  return IntentResult(type='message', data="I don't have any saved results from a previous analysis run in this chat session.")
             return await self.handle_ask_last_run_summary(loaded_state)
        elif intent == "GENERAL_CONVERSATION":
            return await self.handle_general_conversation()
        elif intent == "ANALYZE_NEW":
             # Signal to start MCTS, pass user input as the question
             return IntentResult(type='mcts_params', data={'question': user_input, 'initial_state': None})
        elif intent == "CONTINUE_ANALYSIS":
            # Signal to start MCTS, pass user input and loaded state
            # The user_input might be a follow-up instruction, not necessarily the *original* question.
            # The MCTS needs the *original* question summary from the loaded state ideally.
            # For now, pass the new input. MCTS uses loaded state context.
            if not loaded_state: # Should have been caught above, but double check
                 logger.error("CONTINUE_ANALYSIS intent but no loaded state. This shouldn't happen.")
                 return IntentResult(type='error', data="Internal error: Cannot continue without loaded state.")
            return IntentResult(type='mcts_params', data={'question': user_input, 'initial_state': loaded_state})
        else:
            logger.error(f"Unhandled intent: {intent}")
            return IntentResult(type='error', data=f"Unknown intent: {intent}")


    async def handle_ask_process(self) -> IntentResult:
        logger.info("Handling intent: ASK_PROCESS")
        db_file_name = os.path.basename(self.state_manager.db_file) if self.state_manager.db_file else "N/A"
        explanation = f"""I use an Advanced Bayesian Monte Carlo Tree Search (MCTS) algorithm. Key aspects include:
- **Exploration vs. Exploitation:** Balancing trying new ideas (exploration) with focusing on promising ones (exploitation) using UCT or Thompson Sampling.
- **Bayesian Evaluation:** (Optional) Using Beta distributions to represent score uncertainty for more informed decisions.
- **Node Expansion:** Generating new 'thoughts' (critiques/alternatives/connections) via LLM calls to expand the analysis tree.
- **Simulation:** Evaluating the quality of analysis nodes using LLM calls based on criteria like insight, novelty, relevance, and depth.
- **Backpropagation:** Updating scores/priors and visit counts up the tree path.
- **State Persistence:** (Optional) Saving key results (best analysis, score, priors) and unfit markers between turns within a chat session using a local database (`{db_file_name}`).
- **Intent Handling:** Trying to understand if you want a new analysis, to continue the last one, or ask about results/process/config.
You can adjust parameters via the configuration."""
        return IntentResult(type='message', data=f"**About My Process:**\n{explanation}")

    async def handle_ask_config(self) -> IntentResult:
        logger.info("Handling intent: ASK_CONFIG")
        try:
            # Maybe filter out less relevant config items for display?
            config_to_display = self.config.copy()
            # Remove complex objects if any, or format them nicely
            config_str = json.dumps(config_to_display, indent=2, default=str) # Use default=str for non-serializable
            return IntentResult(type='message', data=f"**Current Run Configuration:**\n```json\n{config_str}\n```")
        except Exception as e:
            logger.error(f"Failed to format/emit config: {e}")
            return IntentResult(type='error', data="Could not display configuration.")


    async def handle_ask_last_run_summary(self, loaded_state: Dict[str, Any]) -> IntentResult:
         logger.info("Handling intent: ASK_LAST_RUN_SUMMARY")
         try:
             summary = "**Summary of Last Analysis Run:**\n"
             summary += f"- **Best Score:** {loaded_state.get('best_score', 'N/A'):.1f}/10\n"
             summary += f"- **Best Analysis Tags:** {', '.join(loaded_state.get('best_node_tags', [])) or 'N/A'}\n"
             summary += f"- **Best Analysis Summary:** {loaded_state.get('best_solution_summary', 'N/A')}\n"

             priors = loaded_state.get("approach_priors")
             if priors and "alpha" in priors and "beta" in priors:
                 means = {}
                 alphas, betas = priors.get("alpha", {}), priors.get("beta", {})
                 for app, alpha in alphas.items():
                     beta = betas.get(app, 1.0)
                     alpha, beta = max(1e-9, alpha), max(1e-9, beta)
                     if alpha + beta > 1e-9: means[app] = (alpha / (alpha + beta)) * 10
                 if means:
                     sorted_means = sorted(means.items(), key=lambda item: item[1], reverse=True)
                     top = [f"{app} ({score:.1f})" for app, score in sorted_means[:3]]
                     summary += f"- **Learned Approach Preferences:** Favored {', '.join(top)}...\n"
                 else: summary += "- **Learned Approach Preferences:** (No valid priors loaded)\n"

             unfit = loaded_state.get("unfit_markers", [])
             if unfit:
                  first_unfit = unfit[0]
                  summary += f"- **Potential Unfit Areas Noted:** {len(unfit)} (e.g., '{first_unfit.get('summary','...')}' due to {first_unfit.get('reason','...')})\n"
             else: summary += "- **Potential Unfit Areas Noted:** None\n"

             return IntentResult(type='message', data=summary)
         except Exception as e:
             logger.error(f"Error formatting last run summary: {e}")
             return IntentResult(type='error', data="Could not display summary of last run.")


    async def handle_general_conversation(self) -> IntentResult:
        logger.info("Handling intent: GENERAL_CONVERSATION")
        response = "Respond to the user's message."
        return IntentResult(type='message', data=response)
