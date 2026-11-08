import re
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from config import VALID_LABELS

if TYPE_CHECKING:
    from config import MoBayesConfig


def enforce_schema(value: str, schema: List[str]) -> str:
    """Return the schema-cased value if it matches (case-insensitive), else 'unknown'."""
    v = str(value) if value is not None else ""
    for s in schema:
        if str(s).lower() == v.lower():
            return s if isinstance(s, str) else str(s)
    return "unknown"


class StructuredParser:
    def __init__(self, llm_client, templates, config: "MoBayesConfig" = None):
        self.llm = llm_client
        self.templates = templates
        if config is None:
            from config import MoBayesConfig
            config = MoBayesConfig()
        self.config = config

    def _response_is_emphatic(self, user_response: str) -> bool:
        text = (user_response or "").strip().lower()
        if not text:
            return False

        return any(pattern in text for pattern in self.config.binary_emphatic_positive)

    @staticmethod
    def _heuristic_map_to_schema(user_response: str, schema: List[str]) -> str:
        text = (user_response or "").strip().lower()
        if not text:
            return ""

        schema_lower = [str(s).lower() for s in schema]

        # Exact label mention in free text takes priority.
        for option in schema_lower:
            if not option:
                continue
            if " " in option or "_" in option or "-" in option:
                if option in text:
                    return option
            elif re.search(rf"\b{re.escape(option)}\b", text):
                return option

        return ""

    def _response_sounds_uncertain(self, user_response: str) -> bool:
        text = (user_response or "").strip().lower()
        if not text:
            return False

        return any(pattern in text for pattern in self.config.uncertainty_patterns)

    @staticmethod
    def _is_binary_schema(schema: List[str]) -> bool:
        lowered = {str(s).lower() for s in schema}
        return {"yes", "no"}.issubset(lowered)

    def _deterministic_binary_extract(self, user_response: str) -> Optional[Tuple[str, str]]:
        text = (user_response or "").strip().lower()
        if not text:
            return None

        cfg = self.config

        def _has(patterns):
            return any(re.search(rf"\b{re.escape(p)}\b", text) for p in patterns)

        has_emphatic_positive = _has(cfg.binary_emphatic_positive)
        has_emphatic_negative = _has(cfg.binary_emphatic_negative)
        has_clear_positive = _has(cfg.binary_clear_positive)
        has_clear_negative = _has(cfg.binary_clear_negative)
        has_weak_positive = _has(cfg.binary_weak_positive)
        has_weak_negative = _has(cfg.binary_weak_negative)

        # Conflict: both positive and negative signals → let LLM decide
        if (has_emphatic_positive or has_clear_positive) and (has_emphatic_negative or has_clear_negative):
            return None
        if has_weak_positive and has_weak_negative:
            return None
        if has_emphatic_positive:
            return "yes", "very_likely"
        if has_emphatic_negative:
            return "no", "very_likely"
        if has_clear_positive:
            return "yes", "very_likely"
        if has_clear_negative:
            return "no", "very_likely"
        if has_weak_positive:
            return "yes", "uncertain"
        if has_weak_negative:
            return "no", "uncertain"

        # No clear signal → let LLM parse (don't assume unknown)
        return None

    def _deterministic_numeric_extract(self, user_response: str, schema: List[str]) -> Optional[Tuple[str, str]]:
        text = (user_response or "").strip().lower()
        if not text:
            return None

        numeric_schema = {str(s) for s in schema}
        if not numeric_schema or not all(value.isdigit() for value in numeric_schema):
            return None

        match = re.search(r"\b(10|[0-9])\b", text)
        if not match:
            return None

        value = match.group(1)
        if value not in numeric_schema:
            return None

        confidence = "uncertain" if self._response_sounds_uncertain(text) else "very_likely"
        return value, confidence

    def _deterministic_extract(
        self, user_response: str, schema: List[str]
    ) -> Optional[Tuple[str, str]]:
        if self._looks_like_clarification(user_response):
            return "clarification_needed", "likely"

        if self._is_binary_schema(schema):
            binary = self._deterministic_binary_extract(user_response)
            if binary:
                return binary

        numeric = self._deterministic_numeric_extract(user_response, schema)
        if numeric:
            return numeric

        heuristic = self._heuristic_map_to_schema(user_response, schema)
        if heuristic:
            confidence = "uncertain" if self._response_sounds_uncertain(user_response) else "very_likely"
            return heuristic, confidence

        return None

    @staticmethod
    def _looks_like_clarification(user_response: str) -> bool:
        text = (user_response or "").strip().lower()
        if not text:
            return False

        hints = (
            "?", "what", "why", "how", "explain", "clarify", "mean", "meaning",
            "do you mean", "can you", "could you", "define",
            "什么", "为啥", "为什么", "怎么", "解释", "什么意思", "是指"
        )
        return any(token in text for token in hints)

    @staticmethod
    def _parse_value_confidence(raw: str) -> Tuple[str, str]:
        """
        Parse LLM output that may contain value|confidence_level.
        Returns (value, confidence_label).
        """
        raw = raw.strip().lower().replace('"', '').replace("'", "")
        raw = raw.split('\n')[0]

        if '|' in raw:
            parts = raw.split('|', 1)
            value = parts[0].strip()
            conf_label = parts[1].strip()
            if conf_label not in VALID_LABELS:
                conf_label = "likely"
            return value, conf_label

        return raw, "likely"

    def extract_feature_value_with_confidence(
        self, user_response: str, feature: str, schema: List[str]
    ) -> Tuple[str, str]:
        """
        Extracts a feature value AND confidence level from user response.

        Returns:
            (value, confidence_label) where confidence_label is one of
            VALID_LABELS or "likely" as default.
        """
        deterministic = self._deterministic_extract(user_response, schema)
        if deterministic:
            return deterministic

        prompt = self.templates.render_parsing_prompt(user_response, feature, schema)
        raw = self.llm.generate(prompt).strip()
        value, confidence = self._parse_value_confidence(raw)

        valid_schema = {s.lower() for s in schema}

        if value == "clarification_needed":
            return value, confidence

        if value in valid_schema or value == "unknown":
            if value == "unknown":
                fallback = self._deterministic_extract(user_response, schema)
                if fallback:
                    return fallback
            elif self._response_sounds_uncertain(user_response) and confidence in ("very_likely", "likely"):
                confidence = "uncertain"
            return value, confidence

        if self._looks_like_clarification(user_response):
            return "clarification_needed", confidence

        fallback = self._deterministic_extract(user_response, schema)
        if fallback:
            return fallback

        return (value if value else "unknown"), confidence

    def extract_feature_value(self, user_response: str, feature: str, schema: List[str]) -> str:
        """Backward-compatible: returns only the value string."""
        value, _ = self.extract_feature_value_with_confidence(user_response, feature, schema)
        return value

    def bulk_extract(self, user_text: str, feature_list: List[Dict[str, Any]]) -> dict:
        """
        Extracts multiple features from free text.

        The returned dict may contain per-feature confidence:
            {"f_fever": {"value": "high", "confidence": "likely"}, ...}
        or simple values for backward compatibility:
            {"f_fever": "high", ...}
        """
        import json
        prompt = self.templates.render_bulk_intake_prompt(user_text, feature_list)
        raw = self.llm.generate(prompt).strip()
        # Clean up possible markdown
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(raw)
        except Exception:
            return {}
