import time
import numpy as np
from typing import Dict, Any, List, Optional, Set
from config import ENTROPY_EPSILON
from world_model.core.belief_state import BeliefState
from world_model.core.likelihood_model import LikelihoodModel
from world_model.knowledge.taxonomy import DiseaseTaxonomy


class EvidenceStore:
    """Tracks evidence gathered during the diagnostic session."""

    def __init__(self):
        self.evidence: Dict[str, Any] = {}
        self.evidence_confidence: Dict[str, float] = {}
        self.asked_features: Set[str] = set()
        self.audit_log: List[Dict[str, Any]] = []

    def add_evidence(self, feature_id: str, value: Any,
                     confidence: float = 1.0, provenance: str = "llm_parsing"):
        self.evidence[feature_id] = value
        self.evidence_confidence[feature_id] = confidence
        self.asked_features.add(feature_id)
        self.audit_log.append({
            "feature_id": feature_id, "value": value,
            "confidence": confidence, "provenance": provenance,
            "timestamp": time.time(),
        })

    def get_all_evidence(self) -> Dict[str, Any]:
        return self.evidence

    def get_evidence_with_confidence(self) -> Dict[str, tuple]:
        """Returns {feature_id: (value, confidence)} for Phase-B adapters."""
        return {fid: (val, self.evidence_confidence.get(fid, 1.0))
                for fid, val in self.evidence.items()}

    def get_audit_log(self) -> List[Dict[str, Any]]:
        return self.audit_log


class WorldModel:
    """
    Central orchestrator for the Bayesian belief engine.
    Owns all state (Belief, Evidence, Prior, Schema, Taxonomy).
    """
    def __init__(self,
                 num_diseases: int,
                 likelihood_model: LikelihoodModel,
                 taxonomy: Optional[DiseaseTaxonomy] = None,
                 base_prior: Optional[np.ndarray] = None,
                 leaf_diseases: Optional[List[str]] = None):
        self.num_diseases = num_diseases
        self.likelihood_model = likelihood_model
        self.taxonomy = taxonomy
        self.evidence_store = EvidenceStore()
        self.leaf_diseases = leaf_diseases or []

        # Normalize base prior
        if base_prior is not None:
            self.base_prior = np.array(base_prior, dtype=float)
            self.base_prior /= np.sum(self.base_prior)
        else:
            self.base_prior = np.ones(num_diseases, dtype=float) / num_diseases

        # d_other index (used externally for result reporting)
        self.d_other_idx = leaf_diseases.index("d_other") if (leaf_diseases and "d_other" in leaf_diseases) else None

        # Initialize belief with base prior
        self.belief = BeliefState(num_diseases, self.base_prior)

    def reset_belief(self):
        """Resets belief state to the base prior."""
        self.belief = BeliefState(self.num_diseases, self.base_prior)
        self.evidence_store = EvidenceStore()

    def update(self,
               feature_id: str,
               value: Any,
               confidence: float = 1.0,
               provenance: str = "llm_parsing"):
        """
        Deterministic Bayesian update with Jeffrey's conditioning.
        """
        likelihood_vector = self.likelihood_model.get_likelihood(feature_id, value, confidence)
        self.belief.update(likelihood_vector)
        self.evidence_store.add_evidence(feature_id, value, confidence, provenance)

    # --- API for Downstream / EIG ---

    def posterior(self) -> np.ndarray:
        return self.belief.get_posterior()

    def entropy(self) -> float:
        return self.belief.get_entropy()

    def compute_posterior_entropy(self, arr: np.ndarray) -> float:
        """Multinomial Shannon entropy of a posterior array."""
        p = arr[arr > ENTROPY_EPSILON]
        return -float(np.sum(p * np.log2(p)))

    def hierarchical_entropy(self, categories: List[str]) -> float:
        """Computes entropy over a set of higher-level categories."""
        if not self.taxonomy:
            return self.entropy()
        dist = self.taxonomy.get_category_distribution(self.posterior(), categories)
        p_safe = dist[dist > ENTROPY_EPSILON]
        return -np.sum(p_safe * np.log2(p_safe))

    def evidence(self) -> Dict[str, Any]:
        return self.evidence_store.get_all_evidence()

    def evidence_with_confidence(self) -> Dict[str, tuple]:
        """Returns {feature_id: (value, confidence)} for Phase-B marginal adapters."""
        return self.evidence_store.get_evidence_with_confidence()

    def unasked_features(self) -> List[str]:
        all_features = set(self.likelihood_model.schema.keys())
        asked = self.evidence_store.asked_features
        return list(all_features - asked)

    def predict_feature_distribution(self, feature_id: str) -> Dict[Any, float]:
        """
        Predicts P(x | E_t) = sum_d P(x | d) * b_t(d)
        """
        schema = self.likelihood_model.schema.get(feature_id)
        if not schema: return {}

        dist = {}
        posterior = self.posterior()
        for val in schema.values:
            likelihood = self.likelihood_model.get_likelihood(feature_id, val)
            p_val = np.sum(likelihood * posterior)
            dist[val] = p_val
        return dist

    def simulate_update(self, feature_id: str, hypothetical_value: Any) -> np.ndarray:
        """
        Counterfactual: what would posterior be IF feature_id = hypothetical_value?
        """
        likelihood = self.likelihood_model.get_likelihood(feature_id, hypothetical_value)
        return self.belief.posterior_if(likelihood)

    def soft_update(self, feature_id: str, center_value: str,
                    sigma: float = 1.5, confidence: float = 1.0):
        """
        Gaussian-weighted soft update for numeric 0-10 features.

        Instead of hard P(f=8|d), computes:
            L_soft(d) = sum_v  w(v) * P(f=v|d)
        where w(v) = Gaussian(v; center=extracted, sigma), normalized.
        Then applies Jeffrey's conditioning: L_eff = c * L_soft + (1-c) * 1.0

        This captures: "patient said 8, could be 7 or 9, but not 2".
        """
        schema = self.likelihood_model.schema[feature_id]
        center = int(center_value)

        # Build Gaussian weights over numeric values
        weights = {}
        for v in schema.values:
            if v == "absent" or not v.isdigit():
                continue
            diff = abs(int(v) - center)
            weights[v] = np.exp(-diff**2 / (2 * sigma**2))

        total_w = sum(weights.values())
        if total_w == 0:
            return

        # Compute weighted mixture likelihood
        soft_likelihood = np.zeros(self.num_diseases)
        for v, w in weights.items():
            L = self.likelihood_model.get_likelihood(feature_id, v)
            soft_likelihood += (w / total_w) * L

        # Apply Jeffrey's conditioning: dampen update by confidence
        if confidence < 1.0:
            soft_likelihood = confidence * soft_likelihood + (1.0 - confidence) * np.ones_like(soft_likelihood)

        # Update belief with soft likelihood
        self.belief.update(soft_likelihood)
        self.evidence_store.add_evidence(feature_id, center_value, confidence, "soft_numeric")

    def get_audit_log(self) -> List[Dict[str, Any]]:
        return self.evidence_store.get_audit_log()

    # --- Differential Diagnosis ---

    def get_differential(self, top_k: int = 3) -> List[Dict[str, Any]]:
        """Returns ranked top-K diseases with probabilities."""
        posterior = self.posterior()
        k = max(1, min(top_k, len(self.leaf_diseases)))
        ranked = np.argsort(posterior)[::-1][:k]

        return [
            {"rank": rank, "disease_id": self.leaf_diseases[idx],
             "probability": float(posterior[idx])}
            for rank, idx in enumerate(ranked, start=1)
        ]
