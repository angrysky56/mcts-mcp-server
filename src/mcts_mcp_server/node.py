#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Node Class for MCTS
====================

This module defines the Node class used in the Monte Carlo Tree Search.
"""
import random
import math
import logging
from typing import Optional, List, Dict, Any

# DEFAULT_CONFIG is now in mcts_config
from .mcts_config import DEFAULT_CONFIG
# truncate_text is now in utils
from .utils import truncate_text

# Setup logger for this module
logger = logging.getLogger(__name__)

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
        self.children: List["Node"] = [] # Explicitly type children
        self.visits = 0
        self.raw_scores: List[float] = [] # Explicitly type raw_scores
        self.sequence = sequence
        self.is_surprising = False
        self.surprise_explanation = ""
        self.approach_type = approach_type
        self.approach_family = approach_family
        self.thought = thought
        self.max_children = max_children
        self.use_bayesian_evaluation = use_bayesian_evaluation
        self.descriptive_tags: List[str] = [] # Explicitly type descriptive_tags

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
                 # Default mid-point score if value is set but no visits (e.g. root node pre-sim)
                 return self.value # Or 5.0 if value represents a single pre-evaluation
            else:
                 return 5.0 # Default mid-point score

    def thompson_sample(self) -> float:
        """Samples from the Beta distribution if Bayesian, else returns midpoint."""
        if self.use_bayesian_evaluation and self.alpha is not None and self.beta is not None:
            alpha_safe = max(1e-9, self.alpha)
            beta_safe = max(1e-9, self.beta)
            try:
                # Try to import numpy locally for beta sampling
                import numpy as np
                return float(np.random.beta(alpha_safe, beta_safe))
            except ImportError:
                logger.warning("Numpy not available for Thompson sampling. Using Bayesian mean.")
                return self.get_bayesian_mean()
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
