"""
Enumeration of reasons for stopping the diagnostic session.

Distinguishes between convergent stops (diagnosis reached) and
abstention stops (system cannot confidently diagnose).
"""
from enum import Enum, auto


class StopReason(Enum):
    CONTINUE = auto()
    CONFIDENT = auto()
    MAX_QUESTIONS = auto()
    FEATURES_EXHAUSTED = auto()
