import numpy as np
from config import ENTROPY_EPSILON, LOG_FLOOR

class BeliefState:
    """
    Manages the posterior distribution over diseases using log-probabilities.
    Supports counterfactual inference for EIG computation.
    """
    def __init__(self, num_diseases: int, prior: np.ndarray = None):
        self.num_diseases = num_diseases
        if prior is not None:
            safe_prior = np.where(prior > 0, prior, LOG_FLOOR)
            self.log_posterior = np.log(safe_prior)
        else:
            self.log_posterior = np.full(num_diseases, -np.log(num_diseases))
        
        self.normalize_log()

    def normalize_log(self):
        max_log = np.max(self.log_posterior)
        log_sum_exp = max_log + np.log(np.sum(np.exp(self.log_posterior - max_log)))
        self.log_posterior -= log_sum_exp

    def get_entropy(self) -> float:
        p = self.get_posterior()
        p_safe = p[p > ENTROPY_EPSILON]
        return -np.sum(p_safe * np.log2(p_safe))

    def update(self, likelihood_vector: np.ndarray):
        """
        Bayesian update. If likelihood_vector is all ones (unknown), belief is unchanged.
        """
        # Linear ones check for neutral evidence policy
        if np.allclose(likelihood_vector, 1.0):
            return

        safe_likelihood = np.where(likelihood_vector > LOG_FLOOR, likelihood_vector, LOG_FLOOR)
        self.log_posterior += np.log(safe_likelihood)
        self.normalize_log()

    def posterior_if(self, likelihood_vector: np.ndarray) -> np.ndarray:
        """
        Returns what the posterior WOULD be if this likelihood was applied.
        Does NOT change internal state. Used for EIG.
        """
        temp_log = self.log_posterior + np.log(np.where(likelihood_vector > LOG_FLOOR, likelihood_vector, LOG_FLOOR))
        max_log = np.max(temp_log)
        log_sum_exp = max_log + np.log(np.sum(np.exp(temp_log - max_log)))
        return np.exp(temp_log - log_sum_exp)

    def get_posterior(self) -> np.ndarray:
        return np.exp(self.log_posterior)
    
    def get_log_posterior(self) -> np.ndarray:
        return self.log_posterior
