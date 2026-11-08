"""
DDXPlus → PatientSim Profile Format

Converts raw DDXPlus patient records (V_codes, feature IDs) into human-readable
PatientSim profiles. Handles:
  - V_code resolution (geographic, pain character, rash color codes)
  - Question text → concise symptom description mapping
  - Feature classification (positive/negative symptoms, medical history, categorical)
  - Chief complaint generation
  - Structured history extraction (social, family, medical)
  - Sparse feature_map construction for faithful patient simulation

This module is imported by convert_ddxplus.py (for patients.jsonl production)
and by generate_profiles.py (for persona wrapping).
"""

import csv
import json
import re
from typing import Dict, List, Tuple, Optional


# ── V_ code resolution ───────────────────────────────────────────────────────

V_CODE_MEANINGS = {
    "V_0": "North Africa", "V_1": "West Africa", "V_2": "South Africa",
    "V_3": "Central America", "V_4": "North America", "V_5": "South America",
    "V_6": "Asia", "V_7": "South East Asia", "V_8": "Caraibes",
    "V_9": "Europe", "V_10": "N", "V_11": "NA", "V_12": "Y", "V_13": "Oceania",
}

PAIN_CHARACTER_CODES = {
    "V_71": "heartbreaking", "V_112": "haunting", "V_154": "tedious",
    "V_161": "sensitive", "V_179": "stabbing", "V_180": "tugging",
    "V_181": "burning", "V_182": "cramping", "V_183": "heavy",
    "V_184": "pulsating", "V_191": "violent", "V_192": "sharp",
    "V_193": "sickening", "V_196": "scary", "V_198": "exhausting",
}

RASH_COLOR_CODES = {
    "V_11": "NA", "V_86": "dark", "V_107": "yellow",
    "V_138": "pale", "V_156": "pink", "V_157": "red",
}


def resolve_v_code(code: str) -> str:
    """Resolve a V_xx code to its English meaning."""
    if code in V_CODE_MEANINGS:
        return V_CODE_MEANINGS[code]
    if code in PAIN_CHARACTER_CODES:
        return PAIN_CHARACTER_CODES[code]
    if code in RASH_COLOR_CODES:
        return RASH_COLOR_CODES[code]
    return code


def normalize_location_text(location: str) -> str:
    """Convert DDXPlus location shorthands into natural phrasing."""
    if not location:
        return location

    loc = location.strip()
    side_patterns = (
        (r'(.+)\(L\)$', r'left \1'),
        (r'(.+)\(R\)$', r'right \1'),
    )
    for pattern, replacement in side_patterns:
        loc = re.sub(pattern, replacement, loc).strip()

    replacements = {
        'posterior aspect of the ankle': 'back of the ankle',
        'dorsal aspect of the foot': 'top of the foot',
        'internal cheek': 'inner cheek',
        'side of the chest': 'side of the chest',
        'side of the neck': 'side of the neck',
    }
    for old, new in replacements.items():
        loc = loc.replace(old, new)

    loc = re.sub(r'\s+', ' ', loc).strip()
    return loc


# ── Feature name extraction ──────────────────────────────────────────────────

def question_to_symptom(question: str, compact: bool = True) -> str:
    """
    Convert a DDXPlus question to a concise symptom description.

    Examples:
        "Do you have a fever (either felt or measured)?" → "fever"
        "Do you feel pain somewhere? [forehead]" → "forehead pain"
        "Have you traveled out of the country in the last 4 weeks?" → "recent international travel"
    """
    # Extract bracketed location (multi-choice body part)
    bracket_match = re.search(r'\[(.+?)\]', question)
    location = normalize_location_text(bracket_match.group(1)) if bracket_match else None
    q = re.sub(r'\s*\[.+?\]', '', question).strip()
    q = q.rstrip('?').strip()

    # Direct overrides for common awkward questions
    DIRECT_MAP = {
        # Pain-related
        "Do you have pain somewhere, related to your reason for consulting":
            "pain related to chief complaint",
        "Do you feel pain somewhere": "pain",
        "Does the pain radiate to another location": "radiating pain",
        "How would you characterize your pain": "pain character",
        "How fast did the pain appear": "pain onset speed",
        "How intense is the pain": "pain intensity",
        "How precisely is the pain located": "pain localization",
        "Do you have pain that is increased when you breathe in deeply":
            "pleuritic chest pain",
        # Rash/skin
        "How intense is the pain caused by the rash": "rash pain intensity",
        "Is the rash swollen": "rash swelling",
        "How severe is the itching": "itching severity",
        "Is the lesion (or are the lesions) larger than 1cm": "lesion >1cm",
        "Do your lesions peel off": "peeling lesions",
        "What color is the rash": "rash color",
        "Do you have any lesions, redness or problems on your skin that you believe are related to the condition you are consulting for":
            "skin lesions/redness",
        # Fatigue/weakness
        "Do you feel so tired that you are unable to do your usual activities or are you stuck in your bed all day long":
            "severe fatigue",
        "Do you constantly feel fatigued or do you have non-restful sleep":
            "chronic fatigue",
        "Do your symptoms of muscle weakness increase with fatigue and/or stress":
            "exercise-related muscle weakness",
        # Dizziness
        "Do you feel slightly dizzy or lightheaded": "dizziness",
        "Do you feel lightheaded and dizzy or do you feel like you are about to faint":
            "lightheadedness/near-fainting",
        # GI symptoms
        "Have you recently had stools that were black (like coal)":
            "black stools (melena)",
        "Have you noticed light red blood or blood clots in your stool":
            "bloody stool (hematochezia)",
        "Have you recently thrown up blood or something resembling coffee beans":
            "hematemesis",
        "Have you vomited several times or have you made several efforts to vomit":
            "repeated vomiting",
        "Have you recently had a loss of appetite or do you get full more quickly then usually":
            "loss of appetite/early satiety",
        "Do you have a burning sensation that starts in your stomach then goes up into your throat, and can be associated with a bitter taste in your mouth":
            "heartburn/acid reflux",
        "Do you feel your abdomen is bloated or distended":
            "abdominal bloating",
        # Respiratory
        "Are you experiencing shortness of breath or difficulty breathing in a significant way":
            "significant dyspnea",
        "Have you ever felt like you were suffocating for a very short time associated with inability to breathe or speak":
            "episodes of suffocation",
        "Do you wheeze while inhaling or is your breathing noisy after coughing spells":
            "wheezing",
        "Have you noticed a high pitched sound when breathing in":
            "stridor",
        # Cardiac
        "Do you feel your heart is beating fast (racing), irregularly (missing a beat) or do you feel palpitations":
            "palpitations",
        "Do you have symptoms that are increased with physical exertion but alleviated with rest":
            "exertional symptoms",
        "Are your symptoms worse when lying down and alleviated while sitting up":
            "orthopnea",
        # Neuro
        "Are you more irritable or has your mood been very unstable recently":
            "mood instability",
        "Have you had an involuntary weight loss over the last 3 months":
            "unintentional weight loss",
        "Have you lost consciousness associated with violent and sustained muscle contractions or had an absence episode":
            "seizures/loss of consciousness",
        "Have you had weakness or paralysis on one side of the face, which may still be present or completely resolved":
            "facial weakness/paralysis",
        "Do you have difficulty swallowing, or have a feeling of discomfort/blockage when swallowing":
            "dysphagia",
        "Do you have the perception of seeing two images of a single object seen overlapping or adjacent to each other":
            "double vision (diplopia)",
        "Do you suddenly have difficulty or an inability to open your mouth or have jaw pain when opening it":
            "trismus/jaw difficulty",
        "Have you noticed that the tone of your voice has become deeper, softer or hoarse":
            "hoarseness",
        "Have you noticed a diffuse (widespread) redness in one or both eyes":
            "eye redness",
        "Do you have numbness, loss of sensation or tingling in the feet":
            "foot numbness/tingling",
        # Travel
        "Have you traveled out of the country in the last 4 weeks":
            "recent international travel",
        # Swelling
        "Where is the swelling located": "swelling",
    }

    for q_prefix, mapped in DIRECT_MAP.items():
        if q.startswith(q_prefix):
            if location:
                return f"{location} {mapped}" if "pain" in mapped else f"{mapped} ({location})"
            return mapped

    # Pattern-based extraction
    patterns = [
        (r'^Do you have (.+)$', r'\1'),
        (r'^Have you had (.+)$', r'\1'),
        (r'^Are you (.+)$', r'\1'),
        (r'^Do you feel (.+)$', r'\1'),
        (r'^Have you (.+)$', r'\1'),
        (r'^Did you (.+)$', r'\1'),
        (r'^Is there (.+)$', r'\1'),
        (r'^Does the pain (.+)$', r'pain \1'),
        (r'^Do your (.+)$', r'\1'),
        (r'^Is the (.+)$', r'\1'),
        (r'^Were you (.+)$', r'\1'),
    ]

    result = q
    for pattern, replacement in patterns:
        m = re.match(pattern, q, re.IGNORECASE)
        if m:
            result = re.sub(pattern, replacement, q, flags=re.IGNORECASE)
            break

    # Remove parenthetical clarifications
    result = re.sub(r'\s*\(.*?\)', '', result).strip()

    # Compact common suffixes
    if compact:
        result = re.sub(r',?\s*related to your reason for consulting', '', result)
        result = re.sub(r'\s*somewhere', '', result)

    # Re-attach location for multi-choice features
    if location:
        if 'pain' in result.lower():
            result = f"{location} pain"
        else:
            result = f"{result} ({location})"

    # Lowercase
    result = result.strip().rstrip('.')
    if result:
        result = result[0].lower() + result[1:]

    return result


def is_antecedent_feature(feature_id: str) -> bool:
    """Check if a feature is an antecedent (medical history) feature."""
    return feature_id.startswith("f_ant_e_")


SOCIAL_HISTORY_DEFAULTS = {
    "tobacco": "Unknown",
    "alcohol": "Unknown",
    "illicit_drug": "Unknown",
    "exercise": "Unknown",
    "marital_status": "Unknown",
    "children": "Unknown",
    "living_situation": "Unknown",
    "occupation": "Unknown",
    "insurance": "Unknown",
    "travel": "Unknown",
}

SOCIAL_HISTORY_RULES = {
    "tobacco": {
        "f_ant_e_79": "current smoker",
        "f_ant_e_191": "former smoker",
        "f_ant_e_222": "daily secondhand smoke exposure",
    },
    "alcohol": {
        "f_ant_e_78": "heavy alcohol use",
    },
    "illicit_drug": {
        "f_ant_e_61": "intravenous drug use",
        "f_ant_e_62": "stimulant drug use",
    },
    "exercise": {
        "f_ant_e_143": "regular exercise",
    },
    "living_situation": {
        "f_ant_e_48": "lives with 4 or more people",
        "f_ant_e_207": "lives in a big city",
        "f_ant_e_195": "lives in the suburbs",
        "f_ant_e_183": "lives in a rural area",
    },
    "occupation": {
        "f_ant_e_49": "daycare exposure/work",
        "f_ant_e_198": "agriculture",
        "f_ant_e_199": "construction",
        "f_ant_e_200": "mining",
    },
}

FAMILY_HISTORY_RULES = {
    "f_ant_e_165": "family history of pneumothorax",
    "f_ant_e_25": "family history of cluster headaches",
    "f_ant_e_26": "family history of anemia",
    "f_ant_e_28": "family history of myasthenia gravis",
    "f_ant_e_29": "family history of psychiatric illness",
    "f_ant_e_86": "family history of allergies/hay fever/eczema",
    "f_ant_e_87": "family history of asthma",
    "f_ant_e_223": "family history of pancreatic cancer",
    "f_ant_e_224": "family history of lung cancer",
    "f_ant_e_225": "family history of premature cardiovascular disease",
}


def _append_unique(values: List[str], value: str) -> None:
    """Append a non-empty value once, preserving insertion order."""
    if value and value not in values:
        values.append(value)


def _join_or_unknown(values: List[str]) -> str:
    """Join collected values or return Unknown if none were observed."""
    return ", ".join(values) if values else "Unknown"


def _normalize_medical_history_item(item: str) -> str:
    """Convert antecedent question text into a concise history phrase."""
    item = re.sub(r'^taking any new ', 'taking ', item)
    item = re.sub(r'^do you regularly take ', '', item)
    item = re.sub(r'^been in contact with a person with similar symptoms.*',
                   'contact with sick person', item)
    item = re.sub(r'^been treated in hospital recently for.*',
                   'recent hospital treatment', item)
    item = re.sub(r'^significantly overweight.*', 'obesity', item)
    item = re.sub(r'^a former smoker$', 'former smoker', item)
    item = re.sub(r'^do you smoke cigarettes$', 'current smoker', item)
    item = re.sub(r'^ever been diagnosed with ', '', item)
    item = re.sub(r'^exposed to secondhand cigarette smoke.*', 'secondhand smoke exposure', item)
    item = re.sub(r'^have you ever had a diagnosis of ', '', item)
    item = re.sub(r'^have you ever had ', '', item)
    item = re.sub(r'^have you been diagnosed with ', '', item)
    item = re.sub(r'^do you have ', '', item)
    item = re.sub(r'^are you currently taking or have you recently taken ', 'taking ', item)
    item = re.sub(r'^are you more likely to develop ', '', item)
    return item.strip()


def extract_structured_history(patient: Dict, features: Dict) -> Tuple[Dict[str, str], List[str], set]:
    """
    Split observed antecedent features into social history, family history,
    and a set of antecedent feature ids consumed by those slots.
    """
    social_values = {key: [] for key in SOCIAL_HISTORY_DEFAULTS}
    family_history = []
    consumed_feature_ids = set()

    for fid, value in patient_truth_features(patient, features).items():
        if fid not in features or not is_antecedent_feature(fid):
            continue

        finfo = features[fid]
        if finfo['type'] == 'binary' and value != 'yes':
            continue
        if finfo['type'] == 'categorical' and value in ('absent', 'unknown'):
            continue

        matched = False
        for field, field_rules in SOCIAL_HISTORY_RULES.items():
            label = field_rules.get(fid)
            if label:
                _append_unique(social_values[field], label)
                consumed_feature_ids.add(fid)
                matched = True
                break
        if matched:
            continue

        if fid == "f_ant_e_204":
            resolved = resolve_v_code(value)
            if resolved == "Y":
                _append_unique(social_values["travel"], "recent international travel")
            elif resolved not in ("N", "NA", "absent", "unknown"):
                _append_unique(social_values["travel"], f"recent travel to {resolved}")
            else:
                _append_unique(social_values["travel"], "no recent international travel")
            consumed_feature_ids.add(fid)
            continue

        family_label = FAMILY_HISTORY_RULES.get(fid)
        if family_label:
            _append_unique(family_history, family_label)
            consumed_feature_ids.add(fid)

    social_history = {
        field: _join_or_unknown(values)
        for field, values in social_values.items()
    }
    return social_history, family_history, consumed_feature_ids


# ── Data loading ─────────────────────────────────────────────────────────────

def load_features(features_csv: str) -> Dict[str, Dict]:
    """Load feature schema from features.csv."""
    features = {}
    with open(features_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = row['feature_id']
            features[fid] = {
                'name': row['name'],
                'type': row['type'],
                'values': json.loads(row['values']),
            }
    return features


def load_diseases(diseases_csv: str) -> Dict[str, str]:
    """Load disease_id → name mapping from diseases.csv."""
    diseases = {}
    with open(diseases_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            diseases[row['disease_id']] = row['name']
    return diseases


def load_patients(patients_jsonl: str) -> List[Dict]:
    """Load patients from JSONL file."""
    patients = []
    with open(patients_jsonl, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                patients.append(json.loads(line))
    return patients


def patient_truth_features(patient: Dict, features: Optional[Dict[str, Dict]] = None) -> Dict[str, str]:
    """
    Return the sparse observed patient state used for simulation.

    New converter outputs store this as `features_observed`. Fall back to the
    dense `features` field for backward compatibility with older files.
    """
    observed = patient.get('features_observed')
    if isinstance(observed, dict):
        return observed

    dense = patient.get('features', {})
    if not isinstance(dense, dict) or not features:
        return dense

    inferred_observed = {}
    for fid, value in dense.items():
        if fid not in features:
            continue
        finfo = features[fid]
        if finfo['type'] == 'binary':
            if value == 'yes':
                inferred_observed[fid] = value
        else:
            if value not in ('absent', 'unknown'):
                inferred_observed[fid] = value
    return inferred_observed


# ── Profile conversion ───────────────────────────────────────────────────────

def classify_features(patient: Dict, features: Dict) -> Tuple[List[str], List[str], List[str], Dict]:
    """
    Classify patient features into positive symptoms, negative symptoms,
    medical history, and categorical findings.

    Returns:
        (positive_symptoms, negative_symptoms, medical_history, categorical_findings)
    """
    positive_symptoms = []
    negative_symptoms = []
    medical_history = []
    categorical_findings = {}
    _, _, consumed_feature_ids = extract_structured_history(patient, features)

    positive_base_questions = set()

    for fid, value in patient_truth_features(patient, features).items():
        if fid not in features:
            continue

        finfo = features[fid]
        fname = finfo['name']
        ftype = finfo['type']
        is_antecedent = is_antecedent_feature(fid)

        symptom_desc = question_to_symptom(fname)

        if ftype == 'binary':
            if value == 'yes':
                if 'nowhere' in symptom_desc.lower():
                    continue
                if symptom_desc.lower().startswith('na ') or symptom_desc.lower().startswith('n '):
                    continue
                if is_antecedent:
                    if fid in consumed_feature_ids:
                        continue
                    medical_history.append(symptom_desc)
                else:
                    positive_symptoms.append(symptom_desc)
                    base_q = re.sub(r'\s*\(.+?\)$', '', symptom_desc)
                    positive_base_questions.add(base_q.split(' (')[0].split(' in ')[0].strip())
            elif value == 'no':
                if not is_antecedent:
                    negative_symptoms.append(symptom_desc)
        elif ftype == 'categorical':
            if value == 'absent' or value == 'unknown':
                continue
            if value.isdigit():
                num_val = int(value)
                categorical_findings[symptom_desc] = num_val
            elif value.startswith('V_'):
                resolved = resolve_v_code(value)
                if resolved in ('N', 'NA'):
                    continue
                if resolved == 'Y':
                    if is_antecedent:
                        medical_history.append(symptom_desc)
                    else:
                        positive_symptoms.append(symptom_desc)
                else:
                    categorical_findings[symptom_desc] = resolved

    # Filter negatives to pertinent negatives only
    ALWAYS_RELEVANT_KEYWORDS = {
        'fever', 'chest', 'breath', 'cough', 'headache', 'nausea', 'vomit',
        'rash', 'swelling', 'bleeding', 'diarrhea', 'pain', 'dizz', 'faint',
    }
    pertinent_negatives = []
    for neg in negative_symptoms:
        neg_lower = neg.lower()
        is_pertinent = any(kw in neg_lower for kw in ALWAYS_RELEVANT_KEYWORDS)
        if not is_pertinent:
            for pos in positive_symptoms:
                pos_words = set(w.lower() for w in pos.split() if len(w) > 3)
                neg_words = set(w.lower() for w in neg.split() if len(w) > 3)
                if pos_words & neg_words:
                    is_pertinent = True
                    break
        if is_pertinent:
            pertinent_negatives.append(neg)

    if len(pertinent_negatives) > 20:
        pertinent_negatives = pertinent_negatives[:20]

    return positive_symptoms, pertinent_negatives, medical_history, categorical_findings


def generate_chief_complaint(positive_symptoms: List[str],
                             categorical_findings: Dict) -> str:
    """
    Generate a natural chief complaint from the top positive symptoms.
    Picks the most clinically significant symptoms (pain, fever, breathing issues).
    """
    SKIP_SYMPTOMS = {'pain related to chief complaint', 'pain'}
    candidates = [s for s in positive_symptoms if s.lower() not in SKIP_SYMPTOMS]

    priority_keywords = [
        'chest', 'breath', 'fever', 'cough', 'headache', 'dizz', 'faint',
        'nausea', 'vomit', 'bleed', 'swell', 'weak', 'rash', 'pain',
        'fatigue', 'stool',
    ]

    scored = []
    for s in candidates:
        score = 0
        for i, kw in enumerate(priority_keywords):
            if kw in s.lower():
                score = len(priority_keywords) - i
                break
        scored.append((score, s))

    scored.sort(key=lambda x: -x[0])
    top_symptoms = [s for _, s in scored[:3]]

    if not top_symptoms:
        if categorical_findings:
            return f"I have {list(categorical_findings.keys())[0]}"
        return "I am not feeling well"

    if len(top_symptoms) == 1:
        return f"I have {top_symptoms[0]}"
    elif len(top_symptoms) == 2:
        return f"I have {top_symptoms[0]} and {top_symptoms[1]}"
    else:
        return f"I have {top_symptoms[0]}, {top_symptoms[1]}, and {top_symptoms[2]}"


def format_categorical_description(categorical_findings: Dict) -> str:
    """Format categorical findings into a readable description."""
    parts = []
    for desc, value in categorical_findings.items():
        if isinstance(value, int):
            parts.append(f"{desc}: {value}/10")
        else:
            parts.append(f"{desc}: {value}")
    return "; ".join(parts)


def _build_feature_map(patient: Dict, features: Dict) -> Dict[str, str]:
    """
    Build a sparse feature_id → human-readable answer map from observed patient state.
    This is used as patient truth, not as the model's dense completed feature vector.
    """
    fmap = {}
    for fid, value in patient_truth_features(patient, features).items():
        if fid not in features:
            continue
        finfo = features[fid]
        fname = finfo['name']
        symptom_desc = question_to_symptom(fname)

        if finfo['type'] == 'binary':
            if value == 'yes':
                fmap[fid] = f"{symptom_desc}: YES"
        elif finfo['type'] == 'categorical':
            if value == 'absent' or value == 'unknown':
                continue
            elif value.isdigit():
                fmap[fid] = f"{symptom_desc}: {value}/10"
            elif value.startswith('V_'):
                resolved = resolve_v_code(value)
                if resolved in ('N', 'NA'):
                    fmap[fid] = f"{symptom_desc}: NO"
                elif resolved == 'Y':
                    fmap[fid] = f"{symptom_desc}: YES"
                else:
                    fmap[fid] = f"{symptom_desc}: {resolved}"
            else:
                fmap[fid] = f"{symptom_desc}: {value}"
    return fmap


def convert_patient_data(patient: Dict, features: Dict, diseases: Dict) -> Dict:
    """
    Convert a single DDXPlus patient to human-readable profile format (without persona).

    This produces the full patient data structure including demographics, symptoms,
    medical history, chief complaint, and feature map. Persona settings are added
    separately by apply_persona().
    """
    meta = patient.get('meta', {})
    age = meta.get('age', 30)
    sex = meta.get('sex', 'Unknown')
    gender = {"M": "Male", "F": "Female"}.get(sex, "Unknown")
    ground_truth_ids = patient.get('ground_truth', [])
    diagnosis = diseases.get(ground_truth_ids[0], ground_truth_ids[0]) if ground_truth_ids else "Unknown"

    # Classify features
    pos_symptoms, neg_symptoms, med_history, cat_findings = classify_features(patient, features)
    social_history, family_history, _ = extract_structured_history(patient, features)

    # Chief complaint: always generate from positive symptoms for richer context
    chief_complaint = generate_chief_complaint(pos_symptoms, cat_findings)

    # Format present illness strings — only positives
    present_positive = ", ".join(pos_symptoms) if pos_symptoms else "general malaise"

    # Clean up medical history phrasing
    cleaned_history = []
    for item in med_history:
        normalized = _normalize_medical_history_item(item)
        _append_unique(cleaned_history, normalized)
    medical_history_str = ", ".join(cleaned_history) if cleaned_history else "Unknown"
    family_history_str = ", ".join(family_history) if family_history else "Unknown"

    # Categorical details (pain intensity, rash color, etc.)
    additional_details = format_categorical_description(cat_findings) if cat_findings else ""

    profile = {
        "hadm_id": patient['patient_id'],
        "age": age,
        "gender": gender,
        "race": "Unknown",

        "Social History": social_history,

        "Medical History": {
            "allergies": "Unknown",
            "family_medical_history": family_history_str,
            "medical_device": "Unknown",
            "medical_history": medical_history_str,
        },

        "Current Visit": {
            "chiefcomplaint": chief_complaint,
            "pain": 0,
            "medication": "Unknown",
            "arrival_transport": "Walk-in",
            "disposition": "Unknown",
            "diagnosis": diagnosis,
        },

        "Present Illness": {
            "present_illness_positive": present_positive,
        },

        "_ddxplus_meta": {
            "patient_id": patient['patient_id'],
            "ground_truth_ids": ground_truth_ids,
            "diagnosis_name": diagnosis,
            "categorical_findings": additional_details,
            "num_positive_symptoms": len(pos_symptoms),
            "num_negative_symptoms": len(neg_symptoms),
            "num_medical_history": len(med_history),
            "feature_map": _build_feature_map(patient, features),
        },
    }

    # Set pain intensity if available
    for desc, val in cat_findings.items():
        if 'intense' in desc.lower() and 'pain' in desc.lower() and isinstance(val, int):
            profile["Current Visit"]["pain"] = val
            break

    return profile


def apply_persona(patient_data: Dict, persona: Dict) -> Dict:
    """
    Add persona settings to a patient data profile.

    Takes a profile from convert_patient_data() and adds persona-specific fields
    used by the patient simulator (language level, personality, recall, dazed).
    """
    profile = dict(patient_data)
    profile["Language Proficiency (CEFR Levels)"] = {
        "cefr": persona.get("cefr", "B"),
    }
    profile["Persona Settings"] = {
        "personality": persona.get("personality", "plain"),
        "recall_level": persona.get("recall_level", "high"),
        "dazed_level": persona.get("dazed_level", "normal"),
    }
    return profile
