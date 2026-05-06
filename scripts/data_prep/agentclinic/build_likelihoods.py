"""Build the LLM-elicited likelihood table P(feature | disease) for AgentClinic-50.

Each (disease, feature) pair is assigned P(YES | disease) using the full 0.0-1.0
scale (avoids the clustering pattern that lower-effort elicitations produce).

Methodology:
  - Default = 0.02 (background "unrelated organ system")
  - Slight relevance     = 0.05-0.10
  - Possible/uncommon    = 0.12-0.25
  - Common               = 0.40-0.60
  - Strong/typical       = 0.65-0.85
  - Pathognomonic        = 0.90-0.97

Output:
  data/kbs/agentclinic_a/raw_output/distributions.json
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
KB_FEATS = REPO / "data/kbs/agentclinic_a/raw_input/features.json"
KB_DZS = REPO / "data/kbs/agentclinic_a/raw_input/diseases.json"
OUT = REPO / "data/kbs/agentclinic_a/raw_output/distributions.json"

DEFAULT_PROB = 0.02


# Each entry maps feature_NAME to P(YES | this disease)
# Anything unspecified gets DEFAULT_PROB.
OVERRIDES: dict[str, dict[str, float]] = {

    "Acalculous cholecystitis": {
        # Critical-illness biliary inflammation. RUQ pain dominant. Often
        # follows trauma/surgery/sepsis. Mostly elderly/comorbid.
        "fever": 0.75, "chills": 0.55, "fatigue": 0.55, "generalized weakness": 0.55,
        "decreased appetite": 0.65, "food avoidance due to pain": 0.50,
        "abdominal pain": 0.88, "right upper quadrant pain": 0.93,
        "epigastric pain": 0.40, "postprandial abdominal pain": 0.50,
        "nausea": 0.65, "vomiting": 0.50, "jaundice": 0.30,
        "altered mental status": 0.30, "confusion": 0.25,
        "recent trauma": 0.35, "history of type 2 diabetes": 0.30,
        "history of hypertension": 0.30, "smoking": 0.30,
    },

    "Actinic Keratosis": {
        # Sun-induced precancerous lesion on chronically sun-exposed skin.
        "skin lesions": 0.95, "rough scaly skin lesion": 0.95,
        "persistent rough scaly skin patch": 0.95,
        "skin lesion on sun-exposed skin": 0.97,
        "scalp lesion": 0.45, "hair thinning over a scalp lesion": 0.30,
        "crusted skin lesion": 0.55, "slowly enlarging skin lesion": 0.65,
        "irregular skin lesion borders": 0.40,
        "multiple skin lesions": 0.55, "localized skin lesions": 0.40,
        "scattered skin lesions": 0.30,
        "nonitchy skin lesions": 0.55, "nontender skin lesions": 0.65,
        "pruritus": 0.20,
    },

    "Adhesive capsulitis": {
        # Frozen shoulder. Defined by ROM loss + pain at shoulder.
        "shoulder pain": 0.95,
        "reduced shoulder range of motion": 0.97,
        "shoulder pain worse with activity": 0.75,
        "shoulder pain at extreme range of motion": 0.92,
        "difficulty raising arms overhead": 0.85,
        "difficulty dressing": 0.65, "difficulty lifting objects": 0.55,
        "joint pain": 0.70, "joint pain worse with activity": 0.65,
        "morning stiffness": 0.45, "reduced joint range of motion": 0.92,
        "history of type 2 diabetes": 0.45, "repetitive joint overuse": 0.20,
    },

    "Autosomal dominant polycystic kidney disease": {
        "flank pain": 0.65, "abdominal pain": 0.45,
        "dark urine": 0.35,  # gross hematuria episodes
        "urinary frequency": 0.30, "recurrent urinary tract infections": 0.40,
        "burning with urination": 0.25,
        "history of hypertension": 0.85,
        "fatigue": 0.30, "generalized weakness": 0.20,
        "headache": 0.30,  # HTN + intracranial aneurysm
    },

    "Bacillus cereus infection": {
        # Food poisoning, classically reheated rice. Two forms — emetic vs diarrheal.
        "nausea": 0.85, "vomiting": 0.92,
        "abdominal pain": 0.70, "abdominal cramps": 0.85,
        "diarrhea": 0.55, "decreased appetite": 0.50,
        "fever": 0.20, "chills": 0.10,
        "generalized weakness": 0.35,
        "reheated food ingestion": 0.85,
        "recent travel": 0.20,
    },

    "Brief Psychotic Disorder": {
        "auditory hallucinations": 0.80, "paranoia": 0.75,
        "agitation": 0.65, "intrusive thoughts": 0.55,
        "altered mental status": 0.45, "sudden altered mental status": 0.50,
        "confusion": 0.40,
        "depressed mood": 0.30, "anxiety": 0.55,
        "psychological stress": 0.85, "poor sleep quality": 0.65,
        "difficulty concentrating": 0.55, "belief of serious illness": 0.20,
        "suicidal thoughts": 0.20,
    },

    "Central retinal artery occlusion": {
        "vision loss": 0.97, "painless vision loss": 0.95,
        "diplopia": 0.10,
        "history of hypertension": 0.65, "history of coronary artery disease": 0.45,
        "history of type 2 diabetes": 0.40, "smoking": 0.45,
        "headache": 0.15,
    },

    "Chronic kidney failure": {
        # Uremic syndrome.
        "fatigue": 0.75, "generalized weakness": 0.65,
        "unintentional weight loss": 0.45, "decreased appetite": 0.55,
        "nausea": 0.45, "vomiting": 0.30,
        "pruritus": 0.55,  # uremic pruritus classic
        "altered mental status": 0.30, "confusion": 0.25,
        "shortness of breath": 0.40, "exertional dyspnea": 0.45,
        "leg swelling": 0.55, "dark urine": 0.25,
        "urinary frequency": 0.30,
        "history of hypertension": 0.85, "history of type 2 diabetes": 0.55,
        "difficulty concentrating": 0.40, "headache": 0.25,
    },

    "Chronic lymphocytic leukemia": {
        "fatigue": 0.60, "generalized weakness": 0.45,
        "unintentional weight loss": 0.45, "night sweats": 0.55,
        "fever": 0.25, "chills": 0.20,
        "decreased appetite": 0.30, "early satiety": 0.40,  # splenomegaly
        "recurrent urinary tract infections": 0.30,  # immunocompromised
    },

    "Chronic lymphocytic leukemia (CLL)": {
        # duplicate disease entry; identical profile
        "fatigue": 0.60, "generalized weakness": 0.45,
        "unintentional weight loss": 0.45, "night sweats": 0.55,
        "fever": 0.25, "chills": 0.20,
        "decreased appetite": 0.30, "early satiety": 0.40,
        "recurrent urinary tract infections": 0.30,
    },

    "Chronic obstructive pulmonary disease": {
        "shortness of breath": 0.92, "exertional dyspnea": 0.92,
        "productive cough": 0.85, "dry cough": 0.45,
        "purulent sputum": 0.55, "chest pain": 0.30,
        "fatigue": 0.55, "generalized weakness": 0.40,
        "unintentional weight loss": 0.40,
        "smoking": 0.92,  # near-pathognomonic RF
        "occupational metal dust exposure": 0.20,
        "history of coronary artery disease": 0.30, "history of hypertension": 0.40,
    },

    "Ewing sarcoma": {
        # Pediatric/young-adult bone tumor. Femur/pelvis/diaphysis common.
        "thigh pain": 0.40, "thigh pain at night": 0.85,
        "thigh pain worse with activity": 0.40, "thigh tenderness": 0.65,
        "limp": 0.45, "difficulty bearing weight": 0.45,
        "thigh mass": 0.55, "firm thigh mass": 0.55,
        "fixed thigh mass": 0.45, "tender thigh mass": 0.65,
        "intramuscular thigh mass": 0.30,
        "normal skin over thigh mass": 0.65,
        "leg swelling": 0.40, "leg redness": 0.20, "leg warmth": 0.20,
        "fever": 0.30, "unintentional weight loss": 0.25,
        "fatigue": 0.40, "night sweats": 0.20,
        "recent trauma": 0.30,  # often mistakenly attributed to trauma
    },

    "Factitious disorder": {
        # Intentionally produced illness for sick role. Many somatic complaints.
        "belief of serious illness": 0.85,
        "abdominal pain": 0.30, "chest pain": 0.30, "headache": 0.30,
        "shortness of breath": 0.20, "nausea": 0.25,
        "altered mental status": 0.10, "confusion": 0.15,
        "depressed mood": 0.40, "anxiety": 0.50,
        "psychological stress": 0.55, "poor sleep quality": 0.40,
        "fever": 0.25,  # often factitious fever
        "vision loss": 0.10, "auditory hallucinations": 0.15,
        "difficulty concentrating": 0.30,
        "recent drug use": 0.20,
    },

    "Femoropopliteal artery stenosis": {
        # Peripheral arterial disease, claudication.
        "calf pain with walking relieved by rest": 0.95,  # claudication = pathognomonic
        "thigh pain worse with activity": 0.55,
        "tingling foot pain": 0.40, "digit color changes": 0.40,
        "leg swelling": 0.10,
        "smoking": 0.85,
        "history of coronary artery disease": 0.55,
        "history of hypertension": 0.65, "history of type 2 diabetes": 0.55,
        "limp": 0.30, "difficulty bearing weight": 0.30,
    },

    "Frontotemporal Dementia": {
        # Behavioral/language variant, typically <65y.
        "word-finding difficulty": 0.80,
        "difficulty concentrating": 0.65,
        "confusion": 0.45, "altered mental status": 0.30,
        "agitation": 0.40, "depressed mood": 0.45,
        "anxiety": 0.30, "intrusive thoughts": 0.20,
        "difficulty dressing": 0.30, "difficulty lifting objects": 0.20,
        "poor sleep quality": 0.40,
        "decreased appetite": 0.30, "weight gain": 0.25,  # disinhibited eating
        "food cravings": 0.30,
    },

    "Gastric peptic ulcer": {
        "epigastric pain": 0.85, "abdominal pain": 0.85,
        "postprandial abdominal pain": 0.75,  # gastric ulcer worsens with eating
        "pain relieved by eating": 0.30,  # duodenal pattern
        "nausea": 0.55, "vomiting": 0.40,
        "early satiety": 0.45, "decreased appetite": 0.45,
        "food avoidance due to pain": 0.60,
        "dark urine": 0.15, "abdominal cramps": 0.45,
        "alcohol use": 0.50, "smoking": 0.45,
        "history of peptic ulcer disease": 0.55,
    },

    "Lichen Sclerosus": {
        # Vulvar/perianal pearly white plaques; female predominant.
        "vulvar itching": 0.92, "pruritus": 0.85,
        "skin lesions": 0.85, "localized skin lesions": 0.80,
        "nonitchy skin lesions": 0.10, "nontender skin lesions": 0.40,
        "dyspareunia": 0.65, "burning with urination": 0.30,
        "irregular skin lesion borders": 0.35,
        "slowly enlarging skin lesion": 0.50,
    },

    "Molluscum contagiosum": {
        # Pox virus skin infection, common in children.
        "skin lesions": 0.95, "multiple skin lesions": 0.80,
        "clustered skin papules": 0.55,
        "smooth firm skin papules": 0.85,
        "umbilicated papules": 0.92,
        "nonitchy skin lesions": 0.55, "nontender skin lesions": 0.85,
        "pruritus": 0.30,
        "contact with similar skin lesions": 0.55,
        "localized skin lesions": 0.45, "scattered skin lesions": 0.55,
    },

    "Myasthenia gravis": {
        # NMJ autoimmune. Fluctuating weakness, ocular onset common.
        "proximal muscle weakness": 0.80, "diplopia": 0.85,
        "vision loss": 0.05,  # ptosis NOT vision loss
        "fatigue": 0.75,
        "difficulty raising arms overhead": 0.55,
        "difficulty climbing stairs": 0.55,
        "difficulty lifting objects": 0.55, "difficulty dressing": 0.45,
        "shortness of breath": 0.30,  # in crisis
        "generalized weakness": 0.50, "reduced grip strength": 0.45,
    },

    "Neuroleptic Malignant Syndrome": {
        # Antipsychotic-induced. Tetrad: hyperthermia, rigidity, AMS, autonomic.
        "altered mental status": 0.95, "sudden altered mental status": 0.80,
        "confusion": 0.85, "agitation": 0.50,
        "muscle rigidity": 0.97, "severe hyperthermia": 0.95,
        "feeling intensely hot": 0.85, "inability to cool down": 0.80,
        "profuse sweating": 0.85, "heart racing": 0.75,
        "recent drug use": 0.85,  # antipsychotic exposure
        "shaking episodes": 0.40,
        "staying conscious during shaking episodes": 0.40,
        "recent extreme heat exposure": 0.15,
        "difficulty climbing stairs": 0.20, "difficulty raising arms overhead": 0.20,
        "drowsiness": 0.45, "fever": 0.85, "chills": 0.20,
    },

    "Osteoarthritis": {
        # Mechanical/degenerative joint disease.
        "joint pain": 0.92, "joint pain worse with activity": 0.85,
        "knee pain": 0.65, "hip pain": 0.45,
        "wrist pain": 0.30, "shoulder pain": 0.30,
        "great toe pain": 0.10,
        "morning stiffness": 0.55,  # mild, brief
        "morning stiffness lasting more than 30 minutes": 0.20,  # RA-like, not OA
        "bony joint enlargement": 0.75,  # Heberden/Bouchard
        "joint crepitus": 0.85,
        "reduced joint range of motion": 0.65,
        "joint swelling": 0.30, "limp": 0.45,
        "difficulty climbing stairs": 0.55, "difficulty lifting objects": 0.40,
        "reduced grip strength": 0.40, "symmetric joint pain": 0.40,
        "repetitive joint overuse": 0.55,
    },

    "Pancreatic adenocarcinoma": {
        "abdominal pain": 0.75, "epigastric pain": 0.65,
        "unintentional weight loss": 0.85, "decreased appetite": 0.75,
        "early satiety": 0.45, "food avoidance due to pain": 0.40,
        "jaundice": 0.65,  # painless jaundice — head of pancreas
        "pruritus": 0.45, "dark urine": 0.50, "pale stools": 0.55,
        "nausea": 0.45, "vomiting": 0.30,
        "fatigue": 0.55, "generalized weakness": 0.45,
        "smoking": 0.55, "alcohol use": 0.30,
        "history of type 2 diabetes": 0.40,
        "depressed mood": 0.30,
    },

    "Phyllodes tumor": {
        # Fibroepithelial breast neoplasm — the rapidly-growing one.
        "breast mass": 0.95, "painless breast mass": 0.75,
        "firm breast mass": 0.85, "fixed breast mass": 0.30,  # usually mobile
        "mobile breast mass": 0.65,
        "rapidly enlarging breast mass": 0.92,  # the distinguishing feature
        "breast contour change": 0.55,  # large size distorts breast
    },

    "Pneumonia": {
        "fever": 0.85, "chills": 0.65,
        "productive cough": 0.85, "dry cough": 0.45, "purulent sputum": 0.70,
        "pleuritic chest pain": 0.55, "chest pain": 0.40,
        "shortness of breath": 0.75, "exertional dyspnea": 0.65,
        "fatigue": 0.65, "generalized weakness": 0.50,
        "decreased appetite": 0.55, "nausea": 0.20,
        "altered mental status": 0.30, "confusion": 0.25,  # esp. elderly
        "night sweats": 0.25, "headache": 0.20,
        "smoking": 0.40,
    },

    "Pseudocyesis": {
        # Psychogenic false pregnancy.
        "belief of pregnancy": 0.97,
        "weight gain": 0.65, "nausea": 0.55,
        "breast contour change": 0.30, "breast mass": 0.10,
        "trying to conceive": 0.65,
        "depressed mood": 0.35, "anxiety": 0.45,
        "decreased appetite": 0.20, "food cravings": 0.45,
        "psychological stress": 0.55, "poor sleep quality": 0.30,
    },

    "Psoriatic arthritis": {
        # Seronegative spondyloarthropathy with psoriasis.
        "joint pain": 0.85, "joint swelling": 0.70,
        "morning stiffness": 0.75,
        "morning stiffness lasting more than 30 minutes": 0.65,
        "symmetric joint pain": 0.30,  # often asymmetric/oligoarticular
        "knee pain": 0.40, "wrist pain": 0.40, "hip pain": 0.20,
        "great toe pain": 0.40,  # dactylitis
        "skin lesions": 0.85, "localized skin lesions": 0.55,
        "rough scaly skin lesion": 0.65,
        "crusted skin lesion": 0.40, "pruritus": 0.55,
        "multiple skin lesions": 0.65, "scalp lesion": 0.45,
        "joint pain worse with activity": 0.40,
        "reduced joint range of motion": 0.55,
    },

    "Seborrheic keratosis": {
        # Benign waxy stuck-on lesions in elderly.
        "skin lesions": 0.92, "crusted skin lesion": 0.85,
        "multiple skin lesions": 0.85, "slowly enlarging skin lesion": 0.65,
        "nonitchy skin lesions": 0.65, "nontender skin lesions": 0.85,
        "pruritus": 0.20, "irregular skin lesion borders": 0.30,
        "localized skin lesions": 0.40, "scattered skin lesions": 0.55,
        "skin lesion on sun-exposed skin": 0.45, "scalp lesion": 0.35,
    },

    "Somatization disorder": {
        # Multiple medically-unexplained somatic complaints.
        "abdominal pain": 0.55, "chest pain": 0.50,
        "headache": 0.55, "shortness of breath": 0.45,
        "nausea": 0.45, "fatigue": 0.55,
        "joint pain": 0.40, "weakness": 0.40, "generalized weakness": 0.40,
        "belief of serious illness": 0.65,
        "depressed mood": 0.55, "anxiety": 0.65,
        "psychological stress": 0.65, "difficulty concentrating": 0.35,
        "poor sleep quality": 0.50, "diplopia": 0.20, "dizziness": 0.50,
        "decreased appetite": 0.30,
    },

    "Syringomyelia": {
        # Cervical cord cyst, dissociated sensory loss + UMN/LMN.
        "proximal muscle weakness": 0.55, "generalized weakness": 0.30,
        "difficulty lifting objects": 0.55, "difficulty climbing stairs": 0.20,
        "difficulty dressing": 0.30,
        "shoulder pain": 0.30, "chest pain": 0.20,
        "tingling foot pain": 0.30, "digit color changes": 0.10,
        "reduced joint range of motion": 0.20,
        "reduced grip strength": 0.50, "headache": 0.20,
    },

    "Takotsubo cardiomyopathy": {
        # Stress-induced reversible LV apical ballooning.
        "chest pain": 0.85, "shortness of breath": 0.65,
        "exertional dyspnea": 0.45, "pleuritic chest pain": 0.20,
        "heart racing": 0.45,
        "psychological stress": 0.85,  # trigger
        "depressed mood": 0.30, "anxiety": 0.55,
        "history of hypertension": 0.55, "history of coronary artery disease": 0.20,
        "altered mental status": 0.10, "confusion": 0.10,
        "nausea": 0.25, "vomiting": 0.15,
    },

    "Vitiligo": {
        # Acquired depigmentation patches.
        "skin lesions": 0.95, "multiple skin lesions": 0.65,
        "localized skin lesions": 0.45, "scattered skin lesions": 0.55,
        "nonitchy skin lesions": 0.85, "nontender skin lesions": 0.85,
        "pruritus": 0.10, "crusted skin lesion": 0.05,
        "rough scaly skin lesion": 0.05,
        "slowly enlarging skin lesion": 0.65, "irregular skin lesion borders": 0.55,
        "scalp lesion": 0.20,
    },
}


def justify(disease: str, feature: str, p: float) -> str:
    if p >= 0.90: tag = "pathognomonic / defining"
    elif p >= 0.75: tag = "strong / typical"
    elif p >= 0.50: tag = "common"
    elif p >= 0.25: tag = "possible"
    elif p >= 0.08: tag = "uncommon"
    else: tag = "unrelated"
    return f"P={p:.2f} ({tag}) for '{feature}' in {disease}"


def main():
    feats = json.load(open(KB_FEATS))["features"]
    diseases = json.load(open(KB_DZS))["diseases"]
    name_to_fid = {f["name"]: f["feature_id"] for f in feats}

    # Validate every override key resolves to a real feature
    unknown = []
    for dz_name, ovs in OVERRIDES.items():
        for fname in ovs:
            if fname not in name_to_fid:
                unknown.append((dz_name, fname))
    if unknown:
        print(f"WARNING: {len(unknown)} unknown feature names in overrides:")
        for dz, fn in unknown[:20]:
            print(f"  {dz}: '{fn}'")
        # Continue anyway — these will silently use DEFAULT

    results = []
    for dz in diseases:
        ovs = OVERRIDES.get(dz["name"], {})
        feats_dict = {}
        for f in feats:
            p = ovs.get(f["name"], DEFAULT_PROB)
            feats_dict[f["feature_id"]] = {
                "type": "binary",
                "prob_yes": p,
                "prob_no": round(1.0 - p, 4),
                "justification": justify(dz["name"], f["name"], p),
            }
        results.append({
            "disease_id": dz["disease_id"],
            "disease_name": dz["name"],
            "features": feats_dict,
        })

    out = {
        "_meta": {
            "description": "Hand-curated likelihood parameters by disease",
            "model": "claude-opus-4-7 (clinical knowledge, manual)",
            "default_prob_yes": DEFAULT_PROB,
            "num_diseases": len(diseases),
            "num_features": len(feats),
            "num_disease_overrides": len(OVERRIDES),
            "num_unknown_overrides": len(unknown),
        },
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {OUT}")
    print(f"  diseases: {len(diseases)}, features: {len(feats)}, "
          f"overridden: {len(OVERRIDES)}, unknowns: {len(unknown)}")


if __name__ == "__main__":
    main()
