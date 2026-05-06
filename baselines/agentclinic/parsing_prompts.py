"""Lenient parsing prompts for AgentClinic-style patient simulators.

Two prompts here:
    render_parsing_prompt_lenient        — turn-by-turn answer extraction
    render_bulk_intake_prompt_lenient    — first-turn feature harvesting

Both share the same rationale: AgentClinic's PatientAgent hedges heavily
("I'm not sure", "I haven't been able to compare"), and its first turn often
mixes age/sex/symptoms/history into a single hedged paragraph. The default
DDxPlus prompts are conservative on hedged inputs, which causes BMBE's
posterior to stall. The lenient variants below extract the same data with
explicit instructions to make best-effort positive inferences from indirect
phrasing.

Examples this prompt set wants to handle correctly:
    - "I'm not having my periods anymore"   -> amenorrhea = yes
    - "I haven't noticed any leg coolness"  -> cool_leg = no
    - "I haven't been able to compare"      -> unknown (genuinely uninformative)
    - "I don't recall any thyroid problem"  -> no (mild lean)
    - "discomfort under my right ribs"      -> right_upper_quadrant_pain = yes
"""

from __future__ import annotations

from typing import Any, Dict, List


def render_parsing_prompt_lenient(user_response: str, feature: str, schema: List[str]) -> str:
    """Lenient turn-by-turn parser for AgentClinic / hedged patient pipelines.

    Drop-in replacement for `PromptTemplates.render_parsing_prompt` — same
    `value|confidence` output format, same allowed schema.
    """
    return f"""
    System: You are a medical data extractor working with patient transcripts that
    often hedge ("I'm not sure", "I haven't noticed", "I can't really tell").
    Your job is to make the best-effort inference yes/no/value whenever the
    patient's wording carries clear semantic content. Use "unknown" ONLY when
    the patient genuinely conveys no information.

    User Response: "{user_response}"
    Feature to Extract: "{feature}" (Allowed: {schema})

    TASK:
    1. Classify into one of the Allowed Values, applying mild inference when
       the patient's wording is hedged but informative.
    2. Assess confidence on the patient's wording:
       very_likely | likely | uncertain | unlikely | very_unlikely

    INFERENCE GUIDELINES (apply in order):

    A. Direct yes/no ("Yes", "No", "Definitely") -> map to schema, very_likely.

    B. Indirect-but-informative phrasing:
       Map these to a concrete schema value with `likely` confidence. Examples:
       - "I'm not having my periods anymore"        -> amenorrhea: yes (likely)
       - "I haven't noticed any leg coolness"       -> cool_leg: no (likely)
       - "I don't recall any thyroid problem"       -> thyroid_disease: no (likely)
       - "I haven't been told my blood counts were low" -> cytopenia: no (likely)

    C. Hedged but leaning ("I think so", "probably not", "maybe a little"):
       Map to the closest schema value with `uncertain` confidence.

    D. Genuinely uninformative responses -> return "unknown|likely":
       - "I can't really tell"
       - "I haven't been able to compare"
       - "I'm too confused to know"
       - "Could you check for me?"
       - Tangential answer that does not address the feature at all.

    E. Asking a question / seeking clarification:
       return "clarification_needed|likely".

    F. For numeric/ordinal features:
       - "they change a little" or "moderately" -> choose the closest level
       - Only return "unknown" if the patient gave no graded information.

    G. The default for ambiguous-but-informative patient phrasing on this
       benchmark is `likely`, NOT `unknown`. Resist the temptation to default
       to "unknown" when the patient has actually provided usable signal.

    Return format: "value|confidence_level"
    Examples: "yes|likely", "no|likely", "uncertain|uncertain", "unknown|likely",
              "high|uncertain", "3|likely"

    Extracted Value:
    """.strip()


def render_bulk_intake_prompt_lenient(user_text: str, feature_list: List[Dict[str, Any]]) -> str:
    """Lenient bulk intake variant for AgentClinic-style first-turn utterances.

    Differences vs upstream:
      - Pushes the model to extract from indirect phrasings, not only explicit
        word-for-word matches.
      - Recognises that AC patients often combine demographics, history,
        symptoms, and concerns in a single hedged paragraph; the prompt
        explicitly walks through these categories.
      - Permits inferring a positive symptom from descriptive wording (e.g.
        "discomfort under my right ribs" -> right_upper_quadrant_pain=yes).
    """
    features_desc = "\n".join(
        f"- {f['id']}: {f['name']} (Values: {f['values']})" for f in feature_list
    )
    return f"""
    System: You are a medical intake specialist parsing a patient's free-text
    self-introduction. The patient may hedge, ramble, mix in irrelevant info,
    or describe symptoms indirectly. Your job is to extract every feature
    from the allowed catalogue that is supported by the text — INCLUDING
    those phrased descriptively rather than by exact name.

    Patient text:
    \"\"\"{user_text}\"\"\"

    ALLOWED FEATURES (extract any that the text supports):
    {features_desc}

    EXTRACTION GUIDELINES:

    A. Walk through the text in four passes:
       1. Demographics — age, sex/gender, body location keywords (e.g. tropical).
       2. Chief complaint — primary presenting symptom(s).
       3. Associated symptoms — fever, pain location, severity, recent change.
       4. History / risk factors — prior conditions, medications, exposures, hospice, immobility.

    B. Map indirect phrasings to features when the meaning is clear:
       - "discomfort under my right ribs / right upper abdomen"
            -> right_upper_quadrant_pain = yes (likely)
       - "feeling hot / temperature went up / I've had fever"
            -> fever = yes (likely)
       - "I've been confused / can't remember well / mind is foggy"
            -> altered_mental_status = yes (likely)
       - "yellowing of my skin / eyes"
            -> jaundice = yes (very_likely)
       - "I'm 67 years old / a senior woman"
            -> age=67, gender=female

    C. Symptoms the patient explicitly DENIES count as a negative feature
       (value=no, likely confidence). Symptoms the patient DOESN'T mention
       are simply omitted (do NOT mark them as no).

    D. Confidence tiers:
       very_likely  : direct, unambiguous statement
       likely       : indirect but clear semantic content
       uncertain    : hedged, leans one way ("I think so")
       unlikely     : mostly negates with caveats
       very_unlikely: barely-plausible

    E. AVOID:
       - Inferring negatives from silence
       - Extracting features the text does not support
       - Returning multiple overlapping features for the same symptom

    Return JSON in EITHER format:
       {{"feature_id": {{"value": "...", "confidence": "likely"}},
         "demographics": {{"age": N, "gender": "...", "location": "..."}}}}
       OR shorthand:
       {{"feature_id": "value|confidence",
         "demographics": {{...}}}}

    JSON Output:
    """.strip()
