#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCTS Configurations
===================

This module stores default configurations, taxonomies, and metadata for the MCTS package.
"""
from typing import Dict, Any, List

DEFAULT_CONFIG: Dict[str, Any] = {
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

APPROACH_TAXONOMY: Dict[str, List[str]] = {
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

APPROACH_METADATA: Dict[str, Dict[str, str]] = {
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
