"""
MoBayes World Model — Bayesian inference engine for diagnostic dialogue.

Submodules:
    core/      Engine, belief state, priors, likelihoods, evidence
    stopping/  Stopping rules
    knowledge/ Taxonomy, data loader
"""

# Re-export main classes for backward-compatible imports
from world_model.core.engine import WorldModel
from world_model.core.belief_state import BeliefState
from world_model.core.likelihood_model import (
    LikelihoodModel, DirichletLikelihoodModel, FeatureSchema
)
from config import resolve_confidence

from world_model.stopping.stopping_rules import StoppingRules
from world_model.stopping.stop_reason import StopReason

from world_model.knowledge.data_loader import load_world_model_from_schema, load_disease_names, load_icd_map
from world_model.knowledge.taxonomy import DiseaseTaxonomy
