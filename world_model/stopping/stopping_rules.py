import numpy as np
from typing import Tuple, Optional, TYPE_CHECKING
from world_model.core.belief_state import BeliefState
from world_model.stopping.stop_reason import StopReason

if TYPE_CHECKING:
    from world_model.core.engine import WorldModel


class StoppingRules:
    """
    Evaluates whether the diagnostic process should terminate.

    Two stopping mechanisms (in priority order):
        1. MAX_QUESTIONS — budget exhausted
        2. CONFIDENT — max posterior >= confidence threshold τ
    """
    def __init__(self,
                 confidence_threshold: float = 0.9,
                 max_questions: int = 25,
                 min_questions_before_abstain: int = 5):
        self.confidence_threshold = confidence_threshold
        self.max_questions = max_questions
        self.min_questions_before_abstain = min_questions_before_abstain

    def should_stop(self,
                    belief: BeliefState,
                    num_questions: int,
                    ) -> Tuple[bool, StopReason]:
        """
        Returns (should_stop, reason).
        """
        posterior = belief.get_posterior()

        # 1. Max questions budget exhausted
        if num_questions >= self.max_questions:
            return True, StopReason.MAX_QUESTIONS

        # 2. Confidence threshold (max posterior)
        if np.max(posterior) >= self.confidence_threshold:
            return True, StopReason.CONFIDENT

        return False, StopReason.CONTINUE

    def should_stop_wm(self, wm: "WorldModel", num_questions: int) -> Tuple[bool, StopReason]:
        """
        Convenience method that extracts belief from WorldModel.
        """
        return self.should_stop(
            belief=wm.belief,
            num_questions=num_questions,
        )
