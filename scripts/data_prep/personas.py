"""
Centralized persona definitions for patient simulation benchmarks.

Each persona controls 4 dimensions of patient behavior:
  - personality: plain | overanxious | distrustful | verbose
  - cefr: Language proficiency level (A = basic, B = intermediate)
  - recall_level: How accurately the patient recalls symptoms (high | low)
  - dazed_level: How confused/dazed the patient acts (normal | moderate)
"""

PERSONAS = {
    "plain": {
        "personality": "plain",
        "cefr": "B",
        "recall_level": "high",
        "dazed_level": "normal",
    },
    "overanxious": {
        "personality": "overanxious",
        "cefr": "B",
        "recall_level": "high",
        "dazed_level": "normal",
    },
    "distrustful": {
        "personality": "distrustful",
        "cefr": "B",
        "recall_level": "high",
        "dazed_level": "normal",
    },
    "dazed": {
        "personality": "plain",
        "cefr": "A",
        "recall_level": "low",
        "dazed_level": "moderate",
    },
    "verbose_lowrecall": {
        "personality": "verbose",
        "cefr": "B",
        "recall_level": "low",
        "dazed_level": "normal",
    },
}
