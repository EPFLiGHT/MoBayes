import os
import json
from typing import Any, Dict, List
from openai import OpenAI

from loaders import load_diseases, load_features
from prompts import build_prompt_generate_distributions as build_prompt
from schemas import build_schema_generate_distributions as build_schema

from loaders import load_config

"""
Generates the distributions P(feature|disease), requires a diseases JSON file and a features/symptom JSON file.
Currently models numerical distributions as Gaussian. 
"""

config = load_config("config.yml")

MODEL_NAME = os.environ.get("LLMKB_MODEL", config["model"]["name"]) # Model used to generate the distributions

DISEASES_JSON_PATH = os.environ.get("LLMKB_DISEASES_JSON_PATH", config["paths"]["diseases_json"]) # JSON file containing the disease list
FEATURES_JSON_PATH = os.environ.get("LLMKB_FEATURES_JSON_PATH", config["paths"]["generated_features_json"]) # JSON file containing the generated features
DISTRIBUTIONS_JSON_PATH = os.environ.get("LLMKB_DISTRIBUTIONS_JSON_PATH", config["paths"]["generated_distributions_json"]) # JSON file containing the generated distributions

CHUNK_SIZE = int(os.environ.get("LLMKB_CHUNK_SIZE", config["distribution_generation"]["chunk_size"])) # Number of features for which the distributions are computed for each API call. More is cheaper but probably starts losing precision at some point.
MAX_API_CALLS = config["distribution_generation"]["max_api_calls"] # To experiment without consuming all API credits, set to None for a full run
TEMPERATURE = float(os.environ.get("LLMKB_TEMPERATURE", config["distribution_generation"]["temperature"])) # Decoding temperature
MAX_OUTPUT_TOKENS = int(os.environ.get("LLMKB_MAX_OUTPUT_TOKENS", "8000")) # Cap reasoning + completion tokens per call (gpt-5.4-nano reasoning tokens count)

NUMERICAL_DISTRIBUTION_TYPE = config["distribution_generation"]["numerical_distribution_type"]
NUMERIC_SCALE_MIN, NUMERIC_SCALE_MAX = config["distribution_generation"]["numeric_scale_min"], config["distribution_generation"]["numeric_scale_max"]


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


def chunk_list(xs: List[Any], chunk_size: int) -> List[List[Any]]:
    return [xs[i:i + chunk_size] for i in range(0, len(xs), chunk_size)]


# Validation
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def validate_response(
    data: Dict[str, Any],
    disease: Dict[str, str],
    feature_chunk: List[Dict[str, Any]],
    numerical_distribution_type: str = NUMERICAL_DISTRIBUTION_TYPE
) -> Dict[str, Any]:
    if data["disease_id"] != disease["disease_id"]:
        raise ValueError(
            f"disease_id mismatch: expected {disease['disease_id']}, got {data['disease_id']}"
        )

    chunk_feature_map = {f["feature_id"]: f for f in feature_chunk}
    seen_ids = set()

    cleaned = {
        "disease_id": data["disease_id"],
        "disease_name": data["disease_name"],
        "feature_parameters": [],
    }

    for row in data["feature_parameters"]:
        feature_id = row["feature_id"]

        if feature_id not in chunk_feature_map:
            raise ValueError(f"Unknown feature_id returned: {feature_id}")
        if feature_id in seen_ids:
            raise ValueError(f"Duplicate feature_id in response: {feature_id}")

        seen_ids.add(feature_id)
        expected_type = chunk_feature_map[feature_id]["type"]

        if row["type"] != expected_type:
            raise ValueError(
                f"Type mismatch for {feature_id}: expected {expected_type}, got {row['type']}"
            )

        if expected_type == "binary":
            # Categorical → probability mapping (LLM returns prob_label, no
            # numbers in the prompt; quantize on our side).
            LABEL_TO_PROB = {
                "very_common": 0.85, "common": 0.65, "occasional": 0.30,
                "uncommon":    0.10, "absent": 0.02,
            }
            label = row.get("prob_label")
            if label not in LABEL_TO_PROB:
                raise ValueError(f"Invalid prob_label for {feature_id}: {label!r}")
            prob_yes = LABEL_TO_PROB[label]

            cleaned["feature_parameters"].append({
                "feature_id": feature_id,
                "type": "binary",
                "prob_yes": prob_yes,
                "prob_no": 1.0 - prob_yes,
                "justification": row["justification"],
            })

        elif expected_type == "numerical":
            if numerical_distribution_type == "gaussian":
                mean = clamp(float(row["mean"]), NUMERIC_SCALE_MIN, NUMERIC_SCALE_MAX)
                std = float(row["std"])

                if std <= 0.0:
                    raise ValueError(f"Invalid std for {feature_id}: {std}")

                std = clamp(std, 0.05, 3.0)

                cleaned["feature_parameters"].append({
                    "feature_id": feature_id,
                    "type": "numerical",
                    "distribution_type": "gaussian",
                    "mean": mean,
                    "std": std,
                    "scale_min": NUMERIC_SCALE_MIN,
                    "scale_max": NUMERIC_SCALE_MAX,
                    "justification": row["justification"],
                })

            elif NUMERICAL_DISTRIBUTION_TYPE == "discrete_0_5":
                dist = row["distribution"]

                required_keys = [str(i) for i in range(6)]
                for k in required_keys:
                    if k not in dist:
                        raise ValueError(f"Missing key {k} in distribution for {feature_id}")

                parsed_dist = {}
                total = 0.0
                for k in required_keys:
                    p = float(dist[k])
                    if p < 0.0 or p > 1.0:
                        raise ValueError(f"Invalid probability {p} for {feature_id}, value {k}")
                    parsed_dist[k] = p
                    total += p

                if abs(total - 1.0) > 1e-6:
                    raise ValueError(
                        f"Distribution for {feature_id} sums to {total}, not 1.0"
                    )

                cleaned["feature_parameters"].append({
                    "feature_id": feature_id,
                    "type": "numerical",
                    "distribution_type": "discrete_0_5",
                    "distribution": parsed_dist,
                    "scale_min": NUMERIC_SCALE_MIN,
                    "scale_max": NUMERIC_SCALE_MAX,
                    "justification": row["justification"],
                })

            else:
                raise ValueError(
                    f"Unsupported numerical distribution type: {NUMERICAL_DISTRIBUTION_TYPE}"
                )

        else:
            raise ValueError(f"Unsupported type: {expected_type}")

    missing = set(chunk_feature_map.keys()) - seen_ids
    if missing:
        raise ValueError(f"Missing feature_ids in response: {sorted(missing)}")

    return cleaned



# API Call
def call_model(
    client: OpenAI,
    disease: Dict[str, str],
    feature_chunk: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prompt = build_prompt(disease, feature_chunk, numerical_distribution_type = NUMERICAL_DISTRIBUTION_TYPE)
    schema = build_schema(numerical_distribution_type = NUMERICAL_DISTRIBUTION_TYPE)

    if os.environ.get("LLMKB_BASE_URL"):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "likelihood_chunk",
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
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "likelihood_chunk",
                    "strict": True,
                    "schema": schema,
                }
            }
        )
        output_text = response.output_text

    data = json.loads(output_text)
    return validate_response(data, disease, feature_chunk)



# Save
def save_output(
    path: str,
    diseases: List[Dict[str, str]],
    features: List[Dict[str, Any]],
    all_results: List[Dict[str, Any]],
    total_api_calls: int,
) -> None:
    payload = {
        "_meta": {
            "description": "LLM-generated likelihood parameters by disease",
            "model": MODEL_NAME,
            "chunk_size": CHUNK_SIZE,
            "max_api_calls": MAX_API_CALLS,
            "temperature": TEMPERATURE,
            "numerical_distribution_type": NUMERICAL_DISTRIBUTION_TYPE,
            "numerical_scale": {
                "min": NUMERIC_SCALE_MIN,
                "max": NUMERIC_SCALE_MAX,
                "zero_means_absent": True,
            },
            "num_diseases": len(diseases),
            "num_features": len(features),
            "num_api_calls_made": total_api_calls,
        },
        "results": all_results,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)



# Main
def main() -> None:
    diseases = load_diseases(DISEASES_JSON_PATH)
    features = load_features(FEATURES_JSON_PATH)
    feature_chunks = chunk_list(features, CHUNK_SIZE)

    client = make_client()

    all_results: List[Dict[str, Any]] = []
    api_calls_made = 0
    stop = False

    for disease_idx, disease in enumerate(diseases, start=1):
        if stop:
            break

        print(f"[Disease {disease_idx}/{len(diseases)}] {disease['name']}")

        # Final flat structure for this disease
        disease_entry = {
            "disease_id": disease["disease_id"],
            "disease_name": disease["name"],
            "features": {}
        }

        for chunk_idx, feature_chunk in enumerate(feature_chunks, start=1):
            if MAX_API_CALLS is not None and api_calls_made >= MAX_API_CALLS:
                print("Reached MAX_API_CALLS, stopping early.")
                stop = True
                break

            print(f"  - Chunk {chunk_idx}/{len(feature_chunks)} "
                  f"({len(feature_chunk)} features)")

            try:
                chunk_result = call_model(
                    client=client,
                    disease=disease,
                    feature_chunk=feature_chunk,
                )

                for feat in chunk_result["feature_parameters"]:
                    feature_id = feat["feature_id"]

                    # Store everything except feature_id itself as the value
                    disease_entry["features"][feature_id] = {
                        k: v for k, v in feat.items() if k != "feature_id"
                    }

                api_calls_made += 1

            except Exception as e:
                print(f"    Failed on disease {disease['disease_id']} chunk {chunk_idx}: {e}")
                api_calls_made += 1

        if disease_entry["features"]:
            all_results.append(disease_entry)

        # Incremental save after each disease — prevents data loss on timeout/crash
        save_output(
            path=DISTRIBUTIONS_JSON_PATH,
            diseases=diseases,
            features=features,
            all_results=all_results,
            total_api_calls=api_calls_made,
        )
        print(f"  [saved {len(all_results)}/{len(diseases)} diseases to disk]")

    print(f"Done. Wrote output to {DISTRIBUTIONS_JSON_PATH}")


if __name__ == "__main__":
    main()
