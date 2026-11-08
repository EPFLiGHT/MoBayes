import os
import re
import json
import hashlib
from typing import Any, Dict, List, Tuple
from openai import OpenAI

from loaders import load_diseases
from prompts import build_prompt_generate_features as build_prompt
from schemas import build_schema_generate_features as build_schema

from loaders import load_config

"""
Ask the model to generate features (i.e. symptoms/medical antecedants) relevant to the list of diseases in the json file.
"""

config = load_config("config.yml")

MODEL_NAME = os.environ.get("LLMKB_MODEL", config["model"]["name"]) # Model used to generate the features
THRESHOLD_P = config["feature_generation"]["threshold_p"] # Estimated minimal probability P(feature|disease) for a feature to be considered relevant

DISEASES_JSON_PATH = os.environ.get("LLMKB_DISEASES_JSON_PATH", config["paths"]["diseases_json"])
FEATURES_JSON_PATH = os.environ.get("LLMKB_FEATURES_JSON_PATH", config["paths"]["generated_features_json"])

MAX_EXISTING_FEATURES_IN_PROMPT = config["prompt"]["max_existing_features_in_prompt"] # Cap the number of features in a prompt to prevent it from getting too big .Set to None for no limit.


def make_client() -> OpenAI:
    """Build an OpenAI-compatible client, optionally targeting another provider."""
    api_key = os.environ.get("LLMKB_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Please set LLMKB_API_KEY or OPENAI_API_KEY in your environment.")

    kwargs = {"api_key": api_key}
    timeout = os.environ.get("LLMKB_TIMEOUT_SECONDS")
    if timeout:
        kwargs["timeout"] = float(timeout)
    base_url = os.environ.get("LLMKB_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


# Feature normalization / ID generation
def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def normalize_name(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def canonical_feature_key(feature: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    Used for Python-side deduplication.
    """
    return (
        feature["feature_type"].strip().lower(),
        feature.get("base_symptom", "").strip().lower(),
        feature["type"].strip().lower(),
        normalize_name(feature["name"]),
    )


def make_feature_id(feature: Dict[str, Any]) -> str:
    """
    Deterministic feature_id from semantic content.
    This keeps duplicate discoveries across diseases stable.
    """
    feature_type = feature["feature_type"]
    base_symptom = feature.get("base_symptom", "")
    name = feature["name"]
    value_type = feature["type"]

    key = f"{feature_type}|{base_symptom}|{name}|{value_type}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]

    stem_parts = [feature_type]
    if base_symptom:
        stem_parts.append(base_symptom)
    stem_parts.append(name)

    stem = slugify("_".join(stem_parts))[:48]
    return f"f_{stem}_{h}"



# OpenAI call
def call_model_for_disease(
    client: OpenAI,
    disease: Dict[str, str],
    threshold_p: float,
    existing_features: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prompt = build_prompt(
        disease=disease,
        threshold_p=threshold_p,
        existing_features=existing_features,
    )
    schema = build_schema()

    if os.environ.get("LLMKB_BASE_URL"):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "feature_generation_result",
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        output_text = response.choices[0].message.content
    else:
        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "feature_generation_result",
                    "strict": True,
                    "schema": schema,
                }
            }
        )
        output_text = response.output_text

    data = json.loads(output_text)

    if data["disease_id"] != disease["disease_id"]:
        raise ValueError(
            f"disease_id mismatch: expected {disease['disease_id']}, got {data['disease_id']}"
        )

    return data



# Validate and merge
def validate_proposal_structure(proposal: Dict[str, Any]) -> None:
    action = proposal["action"]

    if action == "reuse":
        if "existing_feature_id" not in proposal:
            raise ValueError("Reuse proposal missing existing_feature_id")

    elif action == "new":
        required = ["feature_type", "name", "base_symptom", "type"]
        missing = [k for k in required if k not in proposal]
        if missing:
            raise ValueError(f"New proposal missing fields: {missing}")

    else:
        raise ValueError(f"Unknown action: {action}")

    p = float(proposal["estimated_probability_present"])
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Invalid estimated_probability_present: {p}")


def merge_disease_result(
    disease_result: Dict[str, Any],
    global_features: List[Dict[str, Any]],
    disease_feature_map: Dict[str, List[str]],
) -> None:
    disease_id = disease_result["disease_id"]
    disease_feature_map.setdefault(disease_id, [])

    existing_by_id = {f["feature_id"]: f for f in global_features}
    existing_by_key = {canonical_feature_key(f): f for f in global_features}

    for proposal in disease_result["proposals"]:
        validate_proposal_structure(proposal)

        p = float(proposal["estimated_probability_present"])
        if p <= THRESHOLD_P:
            continue

        if proposal["action"] == "reuse":
            existing_id = proposal["existing_feature_id"]
            if existing_id not in existing_by_id:
                # Safety: if the model references a nonexistent feature_id, skip it.
                # Could also raise, but skipping is more convenient in long runs.
                print(
                    f"  Warning: model tried to reuse unknown feature_id '{existing_id}', skipping."
                )
                continue

            if existing_id not in disease_feature_map[disease_id]:
                disease_feature_map[disease_id].append(existing_id)

        else:
            new_feature = {
                "feature_type": proposal["feature_type"],
                "name": proposal["name"].strip(),
                "base_symptom": proposal.get("base_symptom", "").strip(),
                "type": proposal["type"],
            }

            # Python-side deduplication in case the model missed a near duplicate
            key = canonical_feature_key(new_feature)

            if key in existing_by_key:
                feature_id = existing_by_key[key]["feature_id"]
            else:
                feature_id = make_feature_id(new_feature)
                stored_feature = {
                    "feature_id": feature_id,
                    "feature_type": new_feature["feature_type"],
                    "base_symptom": new_feature["base_symptom"],
                    "name": new_feature["name"],
                    "type": new_feature["type"],
                }
                global_features.append(stored_feature)
                existing_by_id[feature_id] = stored_feature
                existing_by_key[key] = stored_feature

            if feature_id not in disease_feature_map[disease_id]:
                disease_feature_map[disease_id].append(feature_id)



def save_output(
    path: str,
    diseases: List[Dict[str, str]],
    global_features: List[Dict[str, Any]],
    disease_feature_map: Dict[str, List[str]],
) -> None:
    payload = {
        "_meta": {
            "description": "LLM-generated feature catalog with reuse/new normalization",
            "model": MODEL_NAME,
            "presence_threshold_p": THRESHOLD_P,
            "num_diseases": len(diseases),
            "num_unique_features": len(global_features),
        },
        "features": sorted(global_features, key=lambda x: x["feature_id"]),
        "disease_feature_map": {
            disease_id: sorted(feature_ids)
            for disease_id, feature_ids in sorted(disease_feature_map.items())
        },
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)



# Main
def main() -> None:
    diseases = load_diseases(DISEASES_JSON_PATH)
    client = make_client()

    global_features: List[Dict[str, Any]] = []
    disease_feature_map: Dict[str, List[str]] = {}

    for i, disease in enumerate(diseases, start=1):
        print(f"[{i}/{len(diseases)}] {disease['name']} ({disease['disease_id']})")

        existing_for_prompt = global_features
        if MAX_EXISTING_FEATURES_IN_PROMPT is not None:
            existing_for_prompt = global_features[:MAX_EXISTING_FEATURES_IN_PROMPT]

        try:
            disease_result = call_model_for_disease(
                client=client,
                disease=disease,
                threshold_p=THRESHOLD_P,
                existing_features=existing_for_prompt,
            )

            merge_disease_result(
                disease_result=disease_result,
                global_features=global_features,
                disease_feature_map=disease_feature_map,
            )

            print(
                f"  Total unique features so far: {len(global_features)}"
            )

        except Exception as e:
            print(f"  Failed for {disease['disease_id']}: {e}")

        # Incremental save after each disease — prevents data loss on timeout/crash
        save_output(
            path=FEATURES_JSON_PATH,
            diseases=diseases,
            global_features=global_features,
            disease_feature_map=disease_feature_map,
        )
        print(f"  [saved {len(disease_feature_map)}/{len(diseases)} diseases to disk]")

    print(f"Done. Wrote output to {FEATURES_JSON_PATH}")


if __name__ == "__main__":
    main()
