from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union

class FeatureSchema:
    """
    Defines the structure and allowed values for clinical features.
    """
    def __init__(self, feature_id: str, ftype: str, values: List[Any], metadata: Dict[str, Any] = None):
        self.feature_id = feature_id
        self.ftype = ftype  # 'binary', 'categorical', 'numeric'
        self.values = values # list of categories or bin edges
        self.metadata = metadata or {}

class LikelihoodModel(ABC):
    """
    Interface for likelihood models providing P(feature=value | disease).
    """
    def __init__(self):
        self.schema: Dict[str, FeatureSchema] = {}

    @abstractmethod
    def get_likelihood(self, feature_id: str, value: Any, confidence: float = 1.0) -> np.ndarray:
        pass

class DirichletLikelihoodModel(LikelihoodModel):
    """
    Hierarchical Likelihood Model treating P(X_i | D) ~ Dirichlet.
    Propagates uncertainty in the clinical knowledge base itself.
    Instead of fixed probabilities, it uses pseudo-counts (alpha).
    """
    def __init__(self,
                 alpha_table: Dict[str, Dict[str, List[float]]],
                 num_diseases: int,
                 schema: Dict[str, FeatureSchema],
                 smoothing: float = 1e-6):
        """
        Args:
            alpha_table: Pseudo-counts mapping. { feature: { value: [alpha_1, ..., alpha_N] } }
                         For a feature with K values, Sum_k alpha_k = total evidence weight.
            smoothing: A small constant added to expected likelihoods to prevent exact zeros.
        """
        super().__init__()
        self.alpha_table = alpha_table
        self.num_diseases = num_diseases
        self.schema = schema
        self.smoothing = smoothing

        # Precompute the expected likelihood E[P(X=v | D)] = alpha_v / Sum(alpha_k)
        # We store this to compute the marginal likelihood during inference.
        self.expected_likelihood = self._compute_expected_likelihoods()

    def _compute_expected_likelihoods(self):
        """Computes the mean of the Dirichlet distribution for each feature/disease."""
        expected = {}
        for feature_id, values_dict in self.alpha_table.items():
            expected[feature_id] = {}
            
            # Aggregate alpha across all values for normalization
            # sum_alphas shape: (num_diseases,)
            sum_alphas = np.zeros(self.num_diseases)
            for val, alphas in values_dict.items():
                sum_alphas += np.array(alphas)
            
            # Avoid division by zero by setting empty sums to 1
            sum_alphas[sum_alphas == 0] = 1.0
            
            for val, alphas in values_dict.items():
                # E[P] = alpha / sum(alphas)
                expected[feature_id][val] = np.array(alphas) / sum_alphas
                
        return expected

    def get_likelihood(self, feature_id: str, value: Any, confidence: float = 1.0) -> np.ndarray:
        """
        Returns the expected likelihood E[P(X=v | D)] marginalized over the Dirichlet prior.

        When confidence < 1.0, applies Jeffrey's conditioning:
            L_eff(D) = c * P(X=v|D) + (1-c) * 1.0
        This mixes the likelihood with neutral evidence proportional to uncertainty.
        """
        if feature_id not in self.schema:
            return np.ones(self.num_diseases)

        schema = self.schema[feature_id]

        if value is None or str(value).lower() == "unknown":
            return np.ones(self.num_diseases)

        if schema.ftype == 'numeric':
            idx = np.searchsorted(schema.values, float(value))
            value = f"bin_{idx}"

        if feature_id in self.expected_likelihood and value in self.expected_likelihood[feature_id]:
            probs = self.expected_likelihood[feature_id][value].copy()
        else:
            probs = np.ones(self.num_diseases)

        if confidence < 1.0:
            # Jeffrey's conditioning: mix likelihood with uniform (neutral)
            probs = confidence * probs + (1.0 - confidence) * np.ones_like(probs)

        return probs

