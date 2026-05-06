from typing import Any, Dict, List


# Prompt construction
def format_existing_catalog(existing_features: List[Dict[str, Any]]) -> str:
    if not existing_features:
        return "None yet."

    lines = []
    for feat in existing_features:
        lines.append(
            f'- feature_id: {feat["feature_id"]}\n'
            f'  name: {feat["name"]}\n'
            f'  feature_type: {feat["feature_type"]}\n'
            f'  base_symptom: {feat.get("base_symptom", "")}\n'
            f'  type: {feat["type"]}'
        )
    return "\n\n".join(lines)


def build_prompt_generate_features(
    disease: Dict[str, str],
    threshold_p: float,
    existing_features: List[Dict[str, Any]],
) -> str:
    existing_catalog_text = format_existing_catalog(existing_features)

    return f"""
You are helping build a synthetic feature catalog for a conversational diagnostic system.

Task:
For the disease below, generate plausible patient features that are likely to be present in patients with this disease.

Disease:
- disease_id: {disease["disease_id"]}
- name: {disease["name"]}

Inclusion rule:
Only include features whose probability of being present in a patient with this disease is greater than {threshold_p}.

Allowed feature kinds:
- symptom_presence: binary presence/absence of a symptom
- symptom_location: binary presence/absence of a symptom at a body location
- symptom_attribute: usually graded or numerical attribute of a symptom, such as intensity or suddenness
- medical_history: binary patient history item
- lifestyle: binary exposure/occupation/habit
- risk_factor: binary risk factor

Important:
- Some features are symptoms or clinical manifestations of the disease.
- Others correspond to patient history, lifestyle, or risk factors.
- For non-symptom features, the disease usually does not cause the feature, so include them only when meaningfully associated with the disease.
- Before proposing a new feature, compare it to the existing catalog below.
- If an existing feature is semantically equivalent or nearly equivalent, reuse that existing feature instead of creating a new one.
- Prefer reusing existing wording to keep the catalog consistent.
- Example: if "difficulty breathing" already exists, do not create "shortness of breath"; reuse the existing feature.
- Example: if "chest pain" already exists, do not create "pain in the chest"; reuse the existing feature.

Existing feature catalog:
{existing_catalog_text}

Output rules:
For each proposed item, return either:
1. action = "reuse"
   - existing_feature_id
   - estimated_probability_present
   - justification

or

2. action = "new"
   - feature_type
   - name
   - base_symptom (empty string if not applicable)
   - type: one of ["binary", "numerical"]
   - estimated_probability_present
   - justification

Additional guidance:
- Prefer clinically useful and distinct features.
- Do not generate generic duplicates.
- Use "numerical" when a symptom is naturally graded in severity, intensity, or duration.
- If a numerical symptom feature uses 0 to represent absence, do not also create a separate binary symptom_presence feature for the same underlying symptom.
- In those cases, prefer the numerical feature.
- Use binary features for symptoms that are not naturally represented by a meaningful graded scale.
- Do not generate feature_id for new features.
- Return only the requested structured data.
""".strip() # Note: estimated_probability_present is used to decide whether to include the feature
#- Use "numerical" only when a graded variable is natural.


# Prompt and schema
def format_feature_block(features: List[Dict[str, Any]]) -> str:
    lines = []
    for feat in features:
        block = [
            f'- feature_id: {feat["feature_id"]}',
            f'  name: {feat["name"]}',
            f'  type: {feat["type"]}',
        ]

        if "feature_type" in feat:
            block.append(f'  feature_type: {feat["feature_type"]}')
        if "base_symptom" in feat:
            block.append(f'  base_symptom: {feat["base_symptom"]}')

        if feat["type"] == "numerical":
            block.append("  scale: 0 to 5, where 0 means symptom absent")

        lines.append("\n".join(block))

    return "\n\n".join(lines)


def build_prompt_generate_distributions(disease: Dict[str, str], feature_chunk: List[Dict[str, Any]], numerical_distribution_type: str = "discrete_0_5") -> str:
    feature_block = format_feature_block(feature_chunk)

    if numerical_distribution_type == "discrete_0_5":
      output_format_rules = """
      - For binary features, return:
        - prob_label: ONE of {very_common, common, occasional, uncommon, absent}
          - very_common : essentially every patient with this disease presents with this feature
          - common      : typical / expected, present in most patients
          - occasional  : sometimes present, not typical
          - uncommon    : rarely associated with this disease
          - absent      : essentially never reported with this disease
          (Choose the label clinically; do not output any number.)
      - For numerical features, return:
        - probabilities for each value on the discrete 0..5 scale
      - For numerical features, 0 means absence of the symptom.
      - The probabilities over values 0, 1, 2, 3, 4, 5 must sum to 1.
      - Use higher probability on 0 when the symptom is often absent.
      - When the symptom is present, distribute probability plausibly across the non-zero severity levels.
      """

    elif numerical_distribution_type == "gaussian":
      output_format_rules = """
      - For binary features, return:
        - prob_yes: probability that the feature value is "yes" given the disease
      - For numerical features, return:
        - mean: expected value on the 0..5 scale
        - std: standard deviation on the 0..5 scale
      - For numerical features, 0 means absence of the symptom.
      - Keep mean within [0, 5].
      - Keep std > 0 and reasonably bounded.
      """
      
    elif numerical_distribution_type == "zero_inflated_gaussian":
      raise NotImplementedError
    
    else:
      raise ValueError(f"Unknown distribution type : {numerical_distribution_type}")

    return f"""
You are generating synthetic likelihood parameters for a Bayesian diagnostic model.

Task:
For the disease below, estimate P(feature | disease) for each listed feature.

Disease:
- disease_id: {disease["disease_id"]}
- name: {disease["name"]}

Features:
{feature_block}

Rules:
- Treat all listed features jointly and keep the disease profile coherent.
- Some features are symptoms, while others may be history, lifestyle, or risk factors.
- For non-symptom features, the disease usually does not cause the feature, so probabilities should stay near baseline prevalence unless strongly associated with the disease.
- Use medically plausible values.
- Avoid extreme certainty unless strongly justified.

Output format rules:
{output_format_rules}

Return only structured data.
""".strip()