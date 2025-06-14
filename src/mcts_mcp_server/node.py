#!/usr/bin/env python3
"""
Node Class for MCTS
====================

This module defines the Node class used in the Monte Carlo Tree Search.
"""
import logging
import random
from typing import Any, Optional

# DEFAULT_CONFIG is now in mcts_config
from .mcts_config import DEFAULT_CONFIG

# truncate_text is now in utils
from .utils import truncate_text

# Setup logger for this module
logger = logging.getLogger(__name__)

class Node:
    """
    Represents a node in the Monte Carlo Tree Search tree.

    Each node contains analysis content, maintains tree relationships, tracks visit statistics,
    and supports both Bayesian and non-Bayesian evaluation modes.
    """
    __slots__ = [
        'alpha',
        'approach_family',
        'approach_type',
        'beta',
        'children',
        'content',
        'descriptive_tags',
        'id',
        'is_surprising',
        'max_children',
        'parent',
        'raw_scores',
        'sequence',
        'surprise_explanation',
        'thought',
        'use_bayesian_evaluation',
        'value',
        'visits'
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
                 **kwargs) -> None:
        """
        Initialize a new MCTS node.

        Args:
            content: The analysis content stored in this node
            parent: Parent node in the tree (None for root)
            sequence: Unique sequence number for node identification
            thought: The thought that generated this node's content
            approach_type: Classification of the analytical approach used
            approach_family: Higher-level grouping of the approach
            max_children: Maximum number of children this node can have
            use_bayesian_evaluation: Whether to use Bayesian or simple averaging
            beta_prior_alpha: Alpha parameter for Beta prior distribution
            beta_prior_beta: Beta parameter for Beta prior distribution
            **kwargs: Additional arguments for alpha, beta, or value initialization
        """
        self.id = "node_" + "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=4))
        self.content = content
        self.parent = parent
        self.children: list[Node] = [] # Explicitly type children
        self.visits = 0
        self.raw_scores: list[float] = [] # Explicitly type raw_scores
        self.sequence = sequence
        self.is_surprising = False
        self.surprise_explanation = ""
        self.approach_type = approach_type
        self.approach_family = approach_family
        self.thought = thought
        self.max_children = max_children
        self.use_bayesian_evaluation = use_bayesian_evaluation
        self.descriptive_tags: list[str] = [] # Explicitly type descriptive_tags

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
        """
        Add a child node to this node.

        Args:
            child: The child node to add

        Returns:
            The added child node (for method chaining)

        Note:
            Automatically sets the child's parent reference to this node
        """
        child.parent = self
        self.children.append(child)
        return child

    def fully_expanded(self) -> bool:
        """
        Check if this node has reached its maximum number of children.

        Returns:
            True if the node cannot have more children, False otherwise

        Note:
            Counts only non-None children in case of sparse child lists
        """
        # Check against actual children added, not just list length if Nones are possible
        valid_children_count = sum(1 for child in self.children if child is not None)
        return valid_children_count >= self.max_children

    def get_bayesian_mean(self) -> float:
        """
        Calculate the Bayesian mean estimate for this node's quality.

        Returns:
            Mean of the Beta distribution (alpha / (alpha + beta)) in range [0, 1]

        Note:
            Only meaningful when use_bayesian_evaluation is True
        """
        if self.use_bayesian_evaluation and self.alpha is not None and self.beta is not None:
            alpha_safe = max(1e-9, self.alpha)
            beta_safe = max(1e-9, self.beta)
            return alpha_safe / (alpha_safe + beta_safe) if (alpha_safe + beta_safe) > 1e-18 else 0.5
        return 0.5 # Default

    def get_average_score(self) -> float:
        """
        Get the average quality score for this node on a 0-10 scale.

        Returns:
            Average score normalized to 0-10 range

        Note:
            Uses Bayesian mean * 10 if Bayesian mode, otherwise cumulative value / visits
        """
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
        """
        Generate a Thompson sampling value from the node's Beta distribution.

        Returns:
            Random sample from Beta(alpha, beta) distribution in range [0, 1]

        Note:
            Falls back to Bayesian mean if numpy is unavailable or sampling fails
        """
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
                logger.warning(f"Thompson sample failed for node {self.sequence} (alpha={alpha_safe}, beta={beta_safe}): {e}. Using mean.")
                return self.get_bayesian_mean()
        return 0.5 # Default

    def best_child(self) -> Optional["Node"]:
        """
        Find the best child node based on visit count and quality score.

        Returns:
            The child node with highest visits (tie-broken by score), or None if no children

        Algorithm:
            1. Find children with maximum visit count
            2. If tied, select highest scoring child
            3. Return None if no children have been visited
        """
        if not self.children:
            return None
        valid_children = [c for c in self.children if c is not None]
        if not valid_children:
            return None

        # Prioritize visits
        most_visited_child = max(valid_children, key=lambda c: c.visits)
        max_visits = most_visited_child.visits
        # If no child has been visited, return None or random? Let's return None.
        if max_visits == 0:
            return None

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
    def node_to_json(self) -> dict[str, Any]:
        """
        Create a comprehensive dictionary representation for debugging and output.

        Returns:
            Dictionary containing all node information including recursive children

        Note:
            Includes full tree structure - use with caution for large trees
        """
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

    def node_to_state_dict(self) -> dict[str, Any]:
        """
        Create a dictionary representation suitable for state persistence.

        Returns:
            Dictionary containing essential node state without recursive children

        Note:
            Optimized for serialization and state saving/loading
        """
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
        """
        Create a concise string representation of the node.

        Returns:
            String containing key node information for debugging
        """
        score = self.get_average_score()
        return (f"Node(Seq:{self.sequence}, Score:{score:.2f}, Visits:{self.visits}, "
                f"Approach:'{self.approach_type}', Children:{len(self.children)}, "
                f"Tags:{self.descriptive_tags})")
