from typing import Any, Dict, List


# Structured output schema
def build_schema_generate_features() -> Dict[str, Any]:
    reuse_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["reuse"]
            },
            "existing_feature_id": {"type": "string"},
            "estimated_probability_present": {"type": "number"},
            "justification": {"type": "string"},
        },
        "required": [
            "action",
            "existing_feature_id",
            "estimated_probability_present",
            "justification",
        ],
        "additionalProperties": False,
    }

    new_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["new"]
            },
            "feature_type": {
                "type": "string",
                "enum": [
                    "symptom_presence",
                    "symptom_location",
                    "symptom_attribute",
                    "medical_history",
                    "lifestyle",
                    "risk_factor",
                ]
            },
            "name": {"type": "string"},
            "base_symptom": {"type": "string"},
            "type": {
                "type": "string",
                "enum": ["binary", "numerical"]
            },
            "estimated_probability_present": {"type": "number"},
            "justification": {"type": "string"},
        },
        "required": [
            "action",
            "feature_type",
            "name",
            "base_symptom",
            "type",
            "estimated_probability_present",
            "justification",
        ],
        "additionalProperties": False,
    }

    return {
        "type": "object",
        "properties": {
            "disease_id": {"type": "string"},
            "disease_name": {"type": "string"},
            "proposals": {
                "type": "array",
                "items": {
                    "anyOf": [reuse_schema, new_schema]
                }
            },
        },
        "required": ["disease_id", "disease_name", "proposals"],
        "additionalProperties": False,
    }


# Output schema
def build_schema_generate_distributions(numerical_distribution_type: str = "discrete_0_5") -> Dict[str, Any]:
    binary_feature_schema = {
        "type": "object",
        "properties": {
            "feature_id": {"type": "string"},
            "type": {"type": "string", "enum": ["binary"]},
            "prob_label": {
                "type": "string",
                "enum": ["very_common", "common", "occasional", "uncommon", "absent"],
            },
            "justification": {"type": "string"},
        },
        "required": ["feature_id", "type", "prob_label", "justification"],
        "additionalProperties": False,
    }

    if numerical_distribution_type == "discrete_0_5":
        numerical_feature_schema = {
            "type": "object",
            "properties": {
                "feature_id": {"type": "string"},
                "type": {"type": "string", "enum": ["numerical"]},
                "distribution": {
                    "type": "object",
                    "properties": {
                        "0": {"type": "number"},
                        "1": {"type": "number"},
                        "2": {"type": "number"},
                        "3": {"type": "number"},
                        "4": {"type": "number"},
                        "5": {"type": "number"},
                    },
                    "required": ["0", "1", "2", "3", "4", "5"],
                    "additionalProperties": False,
                },
                "justification": {"type": "string"},
            },
            "required": ["feature_id", "type", "distribution", "justification"],
            "additionalProperties": False,
        }

    elif numerical_distribution_type == "gaussian":
        numerical_feature_schema = {
            "type": "object",
            "properties": {
                "feature_id": {"type": "string"},
                "type": {"type": "string", "enum": ["numerical"]},
                "mean": {"type": "number"},
                "std": {"type": "number"},
                "justification": {"type": "string"},
            },
            "required": ["feature_id", "type", "mean", "std", "justification"],
            "additionalProperties": False,
        }

    elif numerical_distribution_type == "zero_inflated_gaussian":
        raise NotImplementedError

    else:
        raise ValueError(
            f"Unknown numerical_distribution_type: {numerical_distribution_type}"
        )

    return {
        "type": "object",
        "properties": {
            "disease_id": {"type": "string"},
            "disease_name": {"type": "string"},
            "feature_parameters": {
                "type": "array",
                "items": {
                    "anyOf": [binary_feature_schema, numerical_feature_schema]
                },
            },
        },
        "required": ["disease_id", "disease_name", "feature_parameters"],
        "additionalProperties": False,
    }