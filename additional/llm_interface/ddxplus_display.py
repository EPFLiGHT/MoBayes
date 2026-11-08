"""
DDXPlus-specific display helpers for LLM interaction.

Maps V_codes to human-readable labels, defines numeric scale endpoints,
and handles negated [nowhere] features — all DDXPlus schema knowledge
needed to present features to LLMs in natural language.
"""

from typing import Dict, List, Optional, Tuple

# ── V_code → human-readable label ────────────────────────────────────────────

V_CODE_LABELS = {
    # Rash colors
    "V_11": "not_applicable", "V_86": "dark", "V_107": "yellow",
    "V_138": "pale", "V_156": "pink", "V_157": "red",
    # Yes/No
    "V_10": "no", "V_12": "yes",
    # Travel destinations
    "V_0": "North Africa", "V_1": "West Africa", "V_2": "South Africa",
    "V_3": "Central America", "V_4": "North America", "V_5": "South America",
    "V_6": "Asia", "V_7": "South East Asia", "V_8": "Caribbean",
    "V_9": "Europe", "V_13": "Oceania",
}

LABEL_TO_VCODE = {label.lower(): vc for vc, label in V_CODE_LABELS.items()}

# ── Numeric 0-10 scale endpoint descriptions ─────────────────────────────────

NUMERIC_SCALE_INFO = {
    "f_e_59":  {"low": "gradual/slow onset", "high": "sudden/immediate onset"},
    "f_e_56":  {"low": "mild/no pain", "high": "severe/worst pain"},
    "f_e_58":  {"low": "diffuse/widespread", "high": "very localized/pinpoint"},
    "f_e_134": {"low": "mild/no rash pain", "high": "severe rash pain"},
    "f_e_132": {"low": "not swollen", "high": "very swollen"},
    "f_e_136": {"low": "no itching", "high": "severe itching"},
}

# ── Negated [nowhere] features ───────────────────────────────────────────────
# "Does pain radiate? [nowhere]" → patient "yes" means feature value = "no"

NEGATED_FEATURES = {
    "f_e_57_v_123",   # Does the pain radiate to another location? [nowhere]
    "f_e_133_v_123",  # Where is the affected region located? [nowhere]
    "f_e_152_v_123",  # Where is the swelling located? [nowhere]
}

NEGATED_FEATURE_QUESTIONS = {
    "f_e_57_v_123":  "Does the pain radiate to another location?",
    "f_e_133_v_123": "Is the affected region in a specific location?",
    "f_e_152_v_123": "Is the swelling in a specific location?",
}


# ── Helper functions ─────────────────────────────────────────────────────────

def is_numeric_scale(schema_values: List[str]) -> bool:
    """Check if this feature uses a 0-10 numeric scale."""
    digits = [v for v in schema_values if str(v).isdigit()]
    return len(digits) >= 10


def has_vcodes(schema_values: List[str]) -> bool:
    """Check if this feature uses V_ codes."""
    return any(str(v).startswith("V_") for v in schema_values)


def readable_values(schema_values: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Convert schema values to human-readable labels.
    Returns (readable_list, label_to_schema_map).
    """
    readable = []
    label_to_schema = {}
    for v in schema_values:
        if str(v).startswith("V_"):
            label = V_CODE_LABELS.get(v, v)
            readable.append(label)
            label_to_schema[label.lower()] = v
        else:
            readable.append(v)
            label_to_schema[str(v).lower()] = v
    return readable, label_to_schema


def resolve_negated_name(feature_id: str, feature_name: str) -> str:
    """Replace negated [nowhere] feature names with their positive-form question."""
    if feature_id in NEGATED_FEATURES:
        return NEGATED_FEATURE_QUESTIONS.get(feature_id, feature_name)
    return feature_name


def normalize_extracted_value(feature_id: str, value: str,
                              schema_values: List[str]) -> str:
    """
    Post-process an LLM-extracted value back to the DDXPlus schema domain.

    Handles three DDXPlus-specific conversions:
      1. V_code reverse mapping: human label → V_code (e.g. "red" → "V_157")
      2. Numeric clamping: ensure value is in 0-10 range
      3. Negated feature inversion: yes↔no for [nowhere] features
    """
    # 1. V_code reverse mapping (label → code)
    if has_vcodes(schema_values):
        _, label_to_schema = readable_values(schema_values)
        if value.lower() in label_to_schema:
            value = label_to_schema[value.lower()]
        else:
            # Fuzzy: check if any label is contained in the extracted value
            for label, vcode in label_to_schema.items():
                if label not in ("absent", "na") and label in value.lower():
                    value = vcode
                    break

    # 2. Numeric: clamp to 0-10 range
    if is_numeric_scale(schema_values) and value not in ("unknown", "absent", "clarification_needed"):
        try:
            num = int(float(value))
            value = str(max(0, min(10, num)))
        except (ValueError, TypeError):
            pass

    # 3. Negated features: invert yes↔no
    if feature_id in NEGATED_FEATURES and value in ("yes", "no"):
        value = "no" if value == "yes" else "yes"

    return value


def prepare_display(feature_id: str, feature_name: str,
                    schema_values: List[str]) -> Tuple[str, List[str], Optional[Dict[str, str]]]:
    """
    Prepare human-readable feature name and schema for LLM prompts.

    Enriches numeric features with scale endpoint descriptions,
    converts V_codes to human-readable labels.

    Returns (display_name, display_schema, reverse_map).
    reverse_map is label→schema_value for V_codes, None otherwise.
    """
    display_name = feature_name
    display_schema = list(schema_values)
    reverse_map = None

    # Numeric: add scale context to name
    if is_numeric_scale(schema_values):
        scale = NUMERIC_SCALE_INFO.get(feature_id, {})
        if scale:
            display_name = f"{feature_name} (Scale: 0={scale['low']}, 10={scale['high']})"

    # V_codes: convert to human-readable labels
    elif has_vcodes(schema_values):
        labels, label_to_schema = readable_values(schema_values)
        display_schema = labels
        reverse_map = label_to_schema

    return display_name, display_schema, reverse_map
