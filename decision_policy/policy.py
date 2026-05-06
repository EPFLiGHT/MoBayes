from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, List, Any
from config import ENTROPY_EPSILON
from world_model import WorldModel

class Policy(ABC):
    @abstractmethod
    def select_feature(self, world_model: WorldModel) -> Optional[str]:
        pass

class EIGPolicy(Policy):
    """
    Computes EIG. If taxonomy is loaded, it computes Hierarchical EIG 
    (reducing entropy over top-level categories first).
    """
    def __init__(
        self,
        topk_size: int = 3,
        topk_weight: float = 0.0,
        focus_conf_threshold: float = 1.01,
    ):
        self.topk_size = max(2, int(topk_size))
        self.topk_weight = max(0.0, float(topk_weight))
        self.focus_conf_threshold = float(focus_conf_threshold)
        self.last_selection_info: Dict[str, Any] = {
            "mode": "global",
            "max_posterior": 0.0,
            "topk_weight": 0.0,
            "focus_conf_threshold": self.focus_conf_threshold,
        }

    def select_feature(self, world_model: WorldModel) -> Optional[str]:
        unasked = world_model.unasked_features()
        if not unasked:
            return None
            
        candidates = unasked

        best_feature = None
        max_eig = -float('inf')
        
        # Decide if we are computing standard entropy or hierarchical entropy
        hierarchical_mode = world_model.taxonomy is not None
        
        if hierarchical_mode:
            current_entropy = world_model.hierarchical_entropy(world_model.taxonomy.roots)
        else:
            current_entropy = world_model.entropy()

        posterior = world_model.posterior()
        max_posterior = float(np.max(posterior)) if posterior.size else 0.0
        topk_context = self._build_topk_context(world_model)
        active_topk_weight = 0.0
        selection_mode = "global"
        if (
            topk_context is not None
            and self.topk_weight > 0.0
            and max_posterior >= self.focus_conf_threshold
        ):
            active_topk_weight = self.topk_weight
            selection_mode = "hybrid_topk"
        
        for feature_id in candidates:
            global_eig = self._compute_eig(world_model, feature_id, current_entropy, hierarchical_mode)
            focused_eig = 0.0
            if topk_context is not None and active_topk_weight > 0.0:
                focused_eig = self._compute_topk_eig(world_model, feature_id, topk_context)
            eig = global_eig + active_topk_weight * focused_eig
            if eig > max_eig:
                max_eig = eig
                best_feature = feature_id

        self.last_selection_info = {
            "mode": selection_mode,
            "max_posterior": max_posterior,
            "topk_weight": active_topk_weight,
            "focus_conf_threshold": self.focus_conf_threshold,
            "candidate_count": len(candidates),
            "best_eig": float(max_eig) if np.isfinite(max_eig) else None,
        }
        return best_feature

    def _compute_eig(self, wm: WorldModel, feature_id: str, current_entropy: float, hierarchical: bool) -> float:
        # P(x | E_t)
        prediction = wm.predict_feature_distribution(feature_id)
        expected_entropy = 0.0

        for val, p_val in prediction.items():
            if p_val > 1e-9:
                # Counterfactual update
                sim_posterior = wm.simulate_update(feature_id, val)

                # Compute entropy of simulated posterior
                # Uses engine's own method — works for both multinomial and Noisy-MAX
                if hierarchical and hasattr(wm, 'taxonomy') and wm.taxonomy is not None:
                    dist = wm.taxonomy.get_category_distribution(sim_posterior, wm.taxonomy.roots)
                    h_sim = wm.compute_posterior_entropy(dist)
                else:
                    h_sim = wm.compute_posterior_entropy(sim_posterior)

                expected_entropy += p_val * h_sim

        return current_entropy - expected_entropy

    def _build_topk_context(self, wm: WorldModel) -> Optional[Dict[str, np.ndarray]]:
        posterior = wm.posterior()
        if posterior.size == 0:
            return None

        k = min(self.topk_size, posterior.size)
        topk_idx = np.argsort(posterior)[::-1][:k]
        topk_mass = float(np.sum(posterior[topk_idx]))
        if topk_mass <= 1e-12:
            return None

        topk_prior = posterior[topk_idx] / topk_mass
        topk_safe = topk_prior[topk_prior > ENTROPY_EPSILON]
        topk_entropy = -np.sum(topk_safe * np.log2(topk_safe))
        return {
            "indices": topk_idx,
            "prior": topk_prior,
            "entropy": topk_entropy,
        }

    def _compute_topk_eig(self, wm: WorldModel, feature_id: str, topk_context: Dict[str, np.ndarray]) -> float:
        schema = wm.likelihood_model.schema.get(feature_id)
        if not schema:
            return 0.0

        indices = topk_context["indices"]
        focused_prior = topk_context["prior"]
        current_entropy = topk_context["entropy"]
        expected_entropy = 0.0

        for val in schema.values:
            like = wm.likelihood_model.get_likelihood(feature_id, val)[indices]
            p_val = float(np.sum(like * focused_prior))
            if p_val <= 1e-9:
                continue

            sim = focused_prior * like
            sim_sum = float(np.sum(sim))
            if sim_sum <= 1e-12:
                continue
            sim /= sim_sum

            sim_safe = sim[sim > ENTROPY_EPSILON]
            h_sim = -np.sum(sim_safe * np.log2(sim_safe))
            expected_entropy += p_val * h_sim

        return current_entropy - expected_entropy

