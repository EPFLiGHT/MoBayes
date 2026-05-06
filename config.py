from dataclasses import dataclass
from typing import Dict, Tuple, Union

# ── Numerical stability constants ──
ENTROPY_EPSILON = 1e-20   # min probability for entropy calculation
LOG_FLOOR = 1e-300        # floor for log-space operations (prevents log(0))

# ── Jeffrey's conditioning weights ──
CONFIDENCE_WEIGHTS: Dict[str, float] = {
    "very_likely":   1.0,
    "likely":        0.80,
    "uncertain":     0.50,
    "unlikely":      0.25,
    "very_unlikely": 0.05,
}

VALID_LABELS = set(CONFIDENCE_WEIGHTS.keys())


def resolve_confidence(label: Union[str, float, int]) -> float:
    """
    Resolve a confidence label or numeric value to a Jeffrey's weight.
    Returns a float in [0, 1].
    """
    if isinstance(label, (int, float)):
        return float(max(0.0, min(1.0, label)))
    text = str(label).strip().lower()
    if text in CONFIDENCE_WEIGHTS:
        return CONFIDENCE_WEIGHTS[text]
    return 1.0


@dataclass
class MoBayesConfig:
    # ── Stopping ──
    confidence_threshold: float = 0.90
    max_questions: int = 20
    min_questions_before_abstain: int = 12

    # ── Verbaliser confidence indicator ──
    # max(posterior) < low → "low", < high → "medium", else "high"
    confidence_display_low: float = 0.4
    confidence_display_high: float = 0.8

    # ── Extraction ──
    bulk_extraction_confidence_cap: float = 0.8   # max jeffrey weight for bulk intake
    numeric_gaussian_sigma: float = 1.5           # sigma for soft numeric 0-10 updates

    # ── Paths ──
    data_dir: str = "data/kbs/ddxplus"
    raw_data_dir: str = "data/raw"
    cases_dir: str = "data/cases"
    benchmark_results_dir: str = "data/benchmark_results"

    # ── LLM ──
    default_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0       # MoBayes parsing/question generation
    llm_max_tokens: int = 500
    patient_temperature: float = 0.7    # patient simulator
    patient_max_tokens: int = 200
    standalone_temperature: float = 0.3 # standalone doctor baseline
    standalone_max_tokens: int = 300

    # ── Deterministic parser pattern matching ──
    # (pattern tuples, applied in priority order)
    binary_emphatic_positive: Tuple[str, ...] = (
        "definitely", "absolutely", "for sure", "certainly",
        "without question", "no doubt", "100%",
    )
    binary_emphatic_negative: Tuple[str, ...] = (
        "absolutely not", "definitely not", "never ever", "not at all",
    )
    binary_clear_positive: Tuple[str, ...] = (
        "yes", "yeah", "yep", "i do",
    )
    binary_clear_negative: Tuple[str, ...] = (
        "no", "nope", "never", "i do not", "i don't", "i havent", "i haven't", "without",
    )
    binary_weak_positive: Tuple[str, ...] = (
        "i think so", "probably yes", "maybe yes", "guess so",
    )
    binary_weak_negative: Tuple[str, ...] = (
        "not really", "i don't think so", "probably not", "maybe not", "guess not",
    )
    uncertainty_patterns: Tuple[str, ...] = (
        "maybe", "perhaps", "not sure", "unsure", "uncertain", "i think",
        "i guess", "probably", "possibly", "hard to tell", "cannot remember",
        "can't remember", "dont remember", "don't remember", "not really sure",
        "kind of", "sort of", "i'm not sure", "i am not sure",
    )
