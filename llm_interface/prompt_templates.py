from typing import Dict, Any, List, Optional


# ── Patient Persona Configurations ────────────────────────────────────────────

PERSONA_INSTRUCTIONS = {
    "plain": "Respond clearly and directly to the doctor's questions.",
    "verbose": (
        "You tend to give long, detailed answers. You often provide extra context "
        "and tell stories around your symptoms. Sometimes you go off on tangents."
    ),
    "impatient": (
        "You are impatient and want the visit to be over quickly. Give short, "
        "sometimes curt answers. You may express frustration if asked too many questions."
    ),
    "overanxious": (
        "You are very worried about your health. You tend to over-report symptoms "
        "and ask the doctor if each symptom could be serious. You sometimes confuse "
        "minor sensations with symptoms."
    ),
    "distrustful": (
        "You are skeptical of the doctor. You give minimal information and may "
        "question why certain questions are being asked. You might withhold "
        "information initially."
    ),
    "pleasing": (
        "You want to please the doctor and be a 'good patient'. You may agree "
        "with the doctor's suggestions too readily and sometimes say what you "
        "think the doctor wants to hear rather than the truth."
    ),
}

CEFR_INSTRUCTIONS = {
    "A": (
        "You have basic English proficiency (CEFR A-level). Use simple words "
        "and short sentences. You may struggle to express complex symptoms and "
        "sometimes use the wrong medical term. You may misunderstand complex questions."
    ),
    "B": (
        "You have intermediate English proficiency (CEFR B-level). You can describe "
        "your symptoms adequately but may occasionally use informal or imprecise language."
    ),
    "C": (
        "You have advanced English proficiency (CEFR C-level). You can articulate "
        "your symptoms clearly and use medical terminology when appropriate."
    ),
}

RECALL_INSTRUCTIONS = {
    "low": (
        "You have poor memory recall. You sometimes forget details about when symptoms "
        "started, what medications you take, or your medical history. You may contradict "
        "yourself or say 'I'm not sure' frequently."
    ),
    "high": "You have good recall and can accurately describe your medical history and symptoms.",
}

DAZED_INSTRUCTIONS = {
    "normal": "You are alert and coherent.",
    "moderate": (
        "You are somewhat confused and dazed. You may lose track of the conversation, "
        "repeat yourself, or need questions repeated. Your answers may be slightly "
        "disorganized."
    ),
    "high": (
        "You are very dazed and confused. You struggle to follow the conversation, "
        "give disjointed answers, and may not understand questions on the first try. "
        "You might provide irrelevant information."
    ),
}


# ── Prompt Builders ───────────────────────────────────────────────────────────

def build_patient_system_prompt(profile: Dict) -> str:
    """Build a system prompt for the patient simulator from a converted profile."""
    diagnosis = profile.get("_ddxplus_meta", {}).get("diagnosis_name",
                profile.get("Current Visit", {}).get("diagnosis", "Unknown"))
    positive = profile.get("Present Illness", {}).get("present_illness_positive", "")
    chief = profile.get("Current Visit", {}).get("chiefcomplaint", "")
    social_history = profile.get("Social History", {})
    social_summary = ", ".join(
        f"{key.replace('_', ' ')}: {value}"
        for key, value in social_history.items()
        if value and value != "Unknown"
    ) or "Unknown"
    family_history = profile.get("Medical History", {}).get("family_medical_history", "Unknown")
    med_hist = profile.get("Medical History", {}).get("medical_history", "")
    age = profile.get("age", 30)
    gender = profile.get("gender", "Unknown")
    pain = profile.get("Current Visit", {}).get("pain", 0)
    cat_findings = profile.get("_ddxplus_meta", {}).get("categorical_findings", "")
    # Support both _ddxplus_meta and _llmkb_meta for feature_map
    meta = profile.get("_ddxplus_meta") or profile.get("_llmkb_meta") or {}
    feature_map = meta.get("feature_map", {})

    # Persona
    persona = profile.get("Persona Settings", {}) or profile.get("persona", {})
    personality = persona.get("personality", "plain")
    cefr = profile.get("Language Proficiency (CEFR Levels)", {}).get("cefr", "B")
    recall = persona.get("recall_level", "high")
    dazed = persona.get("dazed_level", "normal")

    # Build sparse observed findings reference
    if feature_map:
        feature_lines = "\n".join(f"  - {v}" for v in feature_map.values())
        feature_section = f"""
KNOWN OBSERVED FINDINGS (use these to answer accurately; do not assume unlisted findings):
{feature_lines}
"""
    else:
        feature_section = ""

    prompt = f"""You are a patient in an Emergency Department consultation. Role-play as this patient.

PATIENT PROFILE:
- Age: {age}, Gender: {gender}
- Chief Complaint: {chief}
- Symptoms Present: {positive}
- Social History: {social_summary}
- Family History: {family_history}
- Medical History: {med_hist}
- Pain Level: {pain}/10
{f'- Additional Findings: {cat_findings}' if cat_findings else ''}
{feature_section}
YOUR TRUE DIAGNOSIS (hidden from doctor): {diagnosis}

RULES:
1. Answer based on the KNOWN OBSERVED FINDINGS above and the patient profile summary.
2. If the doctor asks about something explicitly listed in your observed findings, answer faithfully according to it.
3. If asked about something NOT listed in your observed findings or summary, say no — you do not have that symptom. Do not invent or assume unlisted symptoms.
4. NEVER reveal your diagnosis directly. The doctor must figure it out.
5. DO NOT diagnose yourself or suggest diagnoses.
6. Keep responses concise (1-3 sentences typically).

PERSONA:
- Language: {CEFR_INSTRUCTIONS.get(cefr, CEFR_INSTRUCTIONS['B'])}
- Personality: {PERSONA_INSTRUCTIONS.get(personality, PERSONA_INSTRUCTIONS['plain'])}
- Memory: {RECALL_INSTRUCTIONS.get(recall, RECALL_INSTRUCTIONS['high'])}
- Alertness: {DAZED_INSTRUCTIONS.get(dazed, DAZED_INSTRUCTIONS['normal'])}"""

    return prompt


def build_standalone_doctor_prompt(max_turns: int = 20) -> str:
    """Build a system prompt for the standalone (pure LLM) doctor baseline."""
    base = (
        "You are an emergency department physician conducting a diagnostic interview. "
        "Ask focused, one-at-a-time questions to narrow down the diagnosis. "
        "After gathering enough information, either provide your differential diagnosis "
        "or abstain if the evidence is insufficient. "
    )

    ddx_instr = (
        "When you are confident enough, start your message with '[DDX]' followed by "
        "your top 5 diagnoses separated by commas, from most to least likely. "
    )

    abstain = (
        "If you are not confident enough, start your message with '[ABSTAIN]' followed by "
        "a short reason, then on a new line write '[POSSIBLE]' followed by your best "
        "candidate diagnoses separated by commas.\n\n"
    )

    guidelines = (
        "Guidelines:\n"
        "- Ask one question at a time\n"
        "- Focus on discriminating features\n"
        "- Consider common and serious diagnoses\n"
        "- Do not guess when evidence is insufficient\n"
        f"- You have a maximum of {max_turns} questions before you must provide [DDX] or [ABSTAIN]"
    )

    return base + ddx_instr + abstain + guidelines


STANDALONE_MAX_TURNS_MESSAGE = "You've reached the maximum number of questions. Please provide your [DDX] or [ABSTAIN] now."


def build_closed_world_doctor_prompt(max_turns: int, disease_names: List[str]) -> str:
    """Standalone doctor prompt augmented with the KB disease list (closed-world hint)."""
    base = build_standalone_doctor_prompt(max_turns)
    disease_list = "\n".join(f"  - {n}" for n in sorted(set(disease_names)))
    addendum = (
        "\n\nThe knowledge base includes the following disease list:\n"
        f"{disease_list}\n\n"
        "When you provide your final [DDX], prefer disease names from this list. "
        "If you believe none of these match the patient, you may [ABSTAIN] or "
        "report your best free-form differential."
    )
    return base + addendum


def build_chain_of_thought_doctor_prompt(max_turns: int, disease_names: List[str]) -> str:
    """Closed-world doctor prompt augmented with zero-shot Chain-of-Thought reasoning.

    Standard CoT (Wei et al., NeurIPS 2022) applied to sequential differential
    diagnosis: the model is instructed to reason step-by-step before each turn,
    wrapping the rationale in [REASONING] tags so it can be parsed separately.
    The final question / [DDX] / [ABSTAIN] follows on a new line.
    """
    base = build_closed_world_doctor_prompt(max_turns, disease_names)
    cot_instructions = (
        "\n\n## Reasoning style\n"
        "Before responding each turn, think step by step:\n"
        "1. Summarize the key findings gathered so far.\n"
        "2. List the top candidate diseases with brief reasoning for each "
        "(supporting and refuting evidence).\n"
        "3. Identify the single most informative next action — either a "
        "discriminating question to ask, or the final [DDX] / [ABSTAIN].\n\n"
        "Wrap your reasoning between [REASONING] and [/REASONING] tags. Then "
        "produce either your next question or final answer on a new line."
    )
    return base + cot_instructions


def build_savage_differential_doctor_prompt(max_turns: int, disease_names: List[str]) -> str:
    """Closed-world doctor prompt with Savage et al. (2024) differential-diagnosis
    reasoning instruction.

    Adopts the *Differential Diagnosis CoT* prompt from Savage et al., npj Digital
    Medicine 2024 (Table 2). The original is designed for single-turn USMLE QA
    with two worked-example rationales; we retain the instruction verbatim and
    omit the single-turn examples (they are not directly applicable to our
    multi-turn dialogue setting). The instruction is applied each turn during
    interactive history-taking.
    """
    base = build_closed_world_doctor_prompt(max_turns, disease_names)
    addendum = (
        "\n\n## Reasoning style — Differential Diagnosis (Savage et al., 2024)\n"
        "Use step by step deduction to create a differential diagnosis and then "
        "use step by step deduction to determine the correct response.\n\n"
        "Each turn:\n"
        "1. Build or refine the differential diagnosis based on current evidence.\n"
        "2. For each candidate, list supporting and refuting findings.\n"
        "3. Identify the single most discriminating next question, or — if "
        "confident — give the final [DDX] / [ABSTAIN]."
    )
    return base + addendum


def build_savage_bayesian_doctor_prompt(max_turns: int, disease_names: List[str]) -> str:
    """Closed-world doctor prompt with Savage et al. (2024) Bayesian-inference
    reasoning instruction.

    Adopts the *Bayesian Inference CoT* prompt from Savage et al., npj Digital
    Medicine 2024 (Table 2). The original is designed for single-turn USMLE QA
    with two worked-example rationales; we retain the instruction verbatim and
    omit the single-turn examples. The instruction prompts the LLM to perform
    explicit Bayesian-style reasoning (prior + likelihood updates) each turn
    — directly analogous to BMBE's Bayesian engine but executed via prompt only.
    """
    base = build_closed_world_doctor_prompt(max_turns, disease_names)
    addendum = (
        "\n\n## Reasoning style — Bayesian Inference (Savage et al., 2024)\n"
        "Use step-by-step Bayesian Inference to create a prior probability that "
        "is updated with new information in the history to produce a posterior "
        "probability and determine the final diagnosis.\n\n"
        "Each turn:\n"
        "1. State the current posterior probability for each top candidate "
        "disease (starting from a reasonable pre-test prior on the first turn).\n"
        "2. For each new piece of evidence gathered, state the likelihood ratio "
        "and update the posterior accordingly.\n"
        "3. Either ask a question that would maximally update the posteriors, "
        "or — if a single disease dominates — give the final [DDX] / [ABSTAIN]."
    )
    return base + addendum


class PromptTemplates:
    """
    Standardized prompt templates for controlled generation and parsing.
    """
    @staticmethod
    def render_question_prompt(payload: Dict[str, Any]) -> str:
        """Prompts the LLM to ask a question based on a control payload."""
        clarification_context = ""
        if payload.get('last_user_query'):
            clarification_context = f"\n        - PREVIOUS USER QUERY: \"{payload['last_user_query']}\"\n        - INSTRUCTION: The user seems confused. Briefly explain the feature or address their query before asking the question again."

        return f"""
        System: You are MOD-BE, a professional medical diagnostic assistant.
        Task: Ask the patient a question to check for: {payload['next_feature']}.

        CONTEXT:
        - Feature Name: {payload['next_feature']}
        - Expected Details: {payload['answer_schema']}
        - System Confidence: {payload['confidence_level']}{clarification_context}

        CONSTRAINTS:
        1. NEVER use technical IDs like 'f_fever' or 'd_flu' in the dialogue.
        2. Speak naturally and empathetically.
        3. Do NOT mention probabilities or internal entropy values.
        4. If explaining, keep it simple and medical-focused.

        Question:
        """.strip()

    @staticmethod
    def render_parsing_prompt(user_response: str, feature: str, schema: List[str]) -> str:
        """Prompts the LLM to parse a natural language response into a schema value with confidence."""
        return f"""
        System: You are a medical data extractor.
        User Response: "{user_response}"
        Feature to Extract: "{feature}" (Allowed: {schema})

        TASK:
        1. Classify the user response into one of the Allowed Values.
        2. Assess confidence based on TWO factors:
           a) How certain is the patient? (definite vs hesitant language)
           b) How well does the response match the feature being asked?
           Confidence levels: very_likely, likely, uncertain, unlikely, very_unlikely

        CRITICAL RULES:
        1. If the user response is a direct answer (e.g., "Yes", "I think so", "Not really"), map it to the closest Allowed Value.
        2. If the user response is UNRELATED to the feature, return "unknown|likely".
        3. If the user is ASKING a question or seeking clarification, return "clarification_needed|likely".
        4. If the patient expresses uncertainty ("I think so", "maybe", "not sure"), use "uncertain" confidence.
        5. Use "very_likely" for clear, direct answers ("yes", "no", "definitely", "I have a fever").
        6. Use "likely" when the response requires interpretation or inference to map to the feature (e.g., describing symptoms indirectly).
        7. If the patient gives a clear answer but about something tangential or partially related, use "likely" (NOT very_likely).
        8. If the patient's response requires inference or interpretation to map to the feature, use "uncertain".
        9. If the patient denies having the symptom entirely when asked about a specific aspect of it, use "likely" (the denial is clear but the feature mapping is indirect).
        10. For yes/no features, only return "no" when the patient explicitly denies the feature. Silence, hesitation, or ambiguity should be "unknown".
        11. Prefer "unknown" over a hard negative whenever the response is partial, vague, or not directly about the asked feature.
        12. For numeric/ordinal features, only emit a concrete value if the patient directly states one or clearly maps to one.
        13. If the patient leans toward a value but sounds hesitant ("I think so", "probably not"), return the closest value with "uncertain" confidence instead of "unknown".

        Return format: "value|confidence_level"
        Examples: "high|likely", "yes|uncertain", "no|very_likely", "unknown|likely", "no|likely"

        Extracted Value:
        """.strip()

    @staticmethod
    def render_bulk_intake_prompt(user_text: str, feature_list: List[Dict[str, Any]]) -> str:
        """Prompts the LLM to extract multiple features from a free-text intake."""
        features_desc = "\n".join([f"- {f['id']}: {f['name']} (Values: {f['values']})" for f in feature_list])
        return f"""
        System: You are an expert medical intake specialist.
        User Text: "{user_text}"

        TASK:
        Extract any clinical features mentioned in the User Text.

        ALLOWED FEATURES:
        {features_desc}

        RULES:
        1. Only extract features that are explicitly mentioned or strongly implied.
        2. Map symptoms strictly to the IDs and Values provided.
        3. Multiple features may describe overlapping symptoms (e.g., "dizzy" and "lightheaded").
           Extract overlapping features only when the text directly supports each one; prefer omission over guessing.
        4. EXTRACT DEMOGRAPHICS: Also look for Age, Gender, and Location.
           - age: return as numeric value if found (e.g., "age": 45)
           - gender: return "male" or "female"
           - location: return "tropical" if mentioned.
        5. For any feature not mentioned, do NOT include it in the output.
        6. For each feature, assess how confidently the patient reported it.
           Confidence levels: very_likely, likely, uncertain, unlikely, very_unlikely
        7. Do NOT infer explicit negatives from silence. If the patient does not clearly deny a feature, omit it.
        8. Return the result as a JSON object:
           {{"feature_id": {{"value": "...", "confidence": "likely"}}, "demographics": {{"age": N, "gender": "...", "location": "..."}}}}
           Or simple format: {{"feature_id": "value|confidence", "demographics": {{...}}}}

        JSON Output:
        """.strip()

    @staticmethod
    def render_disease_matching_prompt(predictions: List[str], disease_names: List[str]) -> str:
        """Prompts the LLM to map free-text diagnoses to KB disease names."""
        pred_list = "\n".join(f"{i+1}. {p}" for i, p in enumerate(predictions))
        name_list = "\n".join(f"- {name}" for name in sorted(disease_names))
        return f"""Match each diagnosis below to the closest disease from the reference list.
If a diagnosis does not match any disease in the list, write NO_MATCH.

DIAGNOSES:
{pred_list}

REFERENCE DISEASE LIST:
{name_list}

Output ONLY a numbered list with the matched disease name (exactly as written in the reference list) or NO_MATCH.
Example format:
1. Pneumonia
2. NO_MATCH
3. Gastroenteritis""".strip()

    @staticmethod
    def render_icd10_matching_prompt(predictions: List[str], icd_map: Dict[str, str]) -> str:
        """Prompts the LLM to map free-text disease names to ICD-10 codes (legacy fallback)."""
        pred_list = "\n".join(f"{i+1}. {p}" for i, p in enumerate(predictions))
        icd_list = "\n".join(f"- {code}: {name}" for code, name in sorted(icd_map.items()))
        return f"""For each diagnosis below, provide the most specific ICD-10 code.

DIAGNOSES:
{pred_list}

REFERENCE ICD-10 CODES (from our disease database):
{icd_list}

Output ONLY a numbered list with the ICD-10 code. If unsure, give your best match from the reference list.
Example format:
1. J18.9
2. I20.0
3. F41.0""".strip()
