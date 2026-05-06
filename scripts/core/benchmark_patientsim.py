"""
PatientSim Benchmark Runner for MoBayes.

Benchmarks MoBayes's diagnostic accuracy by running multi-turn conversations
between our Bayesian engine (as doctor) and an LLM-powered patient simulator.

Scenarios:
    A: MoBayes (Bayesian + LLM) vs Standalone LLM doctor
    B: Different patient personas (CEFR levels, personality types, recall, dazedness)

Usage:
    # Single patient test
    python scripts/benchmark_patientsim.py --n 1

    # Full benchmark with persona variations
    python scripts/benchmark_patientsim.py --personas plain,overanxious,verbose

    # Standalone doctor comparison
    python scripts/benchmark_patientsim.py --doctor-mode standalone --n 10
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Add parent dir to path for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

# Load .env if present
_env_path = os.path.join(BASE_DIR, '.env')
if os.path.exists(_env_path):
    from dotenv import load_dotenv
    load_dotenv(_env_path)

from world_model import load_world_model_from_schema, load_disease_names, load_icd_map, StoppingRules, resolve_confidence
from config import VALID_LABELS
from decision_policy.policy import EIGPolicy
from llm_interface.prompt_templates import (
    PromptTemplates, build_patient_system_prompt, build_standalone_doctor_prompt,
    STANDALONE_MAX_TURNS_MESSAGE,
    PERSONA_INSTRUCTIONS, CEFR_INSTRUCTIONS, RECALL_INSTRUCTIONS, DAZED_INSTRUCTIONS,
)
from llm_interface.structured_parser import StructuredParser, enforce_schema
from llm_interface.llm_client import OpenAILLMClient, sanitize_openai_text, sanitize_openai_messages
from llm_interface.ddxplus_display import (
    is_numeric_scale, prepare_display, resolve_negated_name, normalize_extracted_value,
)
from config import MoBayesConfig


# ── Patient Simulator (OpenAI-based) ─────────────────────────────────────────

class SimplePatientAgent:
    """
    Lightweight patient simulator.
    Uses the profile to simulate patient responses via multi-turn LLM chat.
    """

    def __init__(self, profile: Dict, api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini", config: MoBayesConfig = None):
        self.config = config or MoBayesConfig()
        self.profile = profile
        self.diagnosis = profile.get("_ddxplus_meta", {}).get("diagnosis_name",
                         profile.get("Current Visit", {}).get("diagnosis", "Unknown"))
        self.system_prompt = build_patient_system_prompt(profile)
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.llm = OpenAILLMClient(model=model, api_key=api_key,
                                    temperature=self.config.patient_temperature)

    def reset(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def respond(self, doctor_question: str) -> str:
        """Generate a patient response to the doctor's question."""
        self.messages.append({"role": "user", "content": sanitize_openai_text(doctor_question)})
        answer, usage = self.llm.chat(self.messages, max_tokens=self.config.patient_max_tokens)
        self.messages.append({"role": "assistant", "content": answer})
        self.token_usage["prompt_tokens"] += usage["prompt_tokens"]
        self.token_usage["completion_tokens"] += usage["completion_tokens"]
        return answer


# ── Doctor Modes ─────────────────────────────────────────────────────────────

class StandaloneDoctorAgent:
    """
    Pure LLM doctor (no Bayesian engine). For comparison benchmark.
    GPT-4o acting as a doctor asking diagnostic questions.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o",
                 max_turns: int = 20, base_url: Optional[str] = None,
                 config: MoBayesConfig = None):
        self.config = config or MoBayesConfig()
        self.llm = OpenAILLMClient(model=model, api_key=api_key,
                                    temperature=self.config.standalone_temperature,
                                    base_url=base_url)
        self.max_turns = max_turns
        self.messages = []
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.reset()

    def reset(self):
        prompt = build_standalone_doctor_prompt(self.max_turns)
        self.messages = [{"role": "system", "content": prompt}]
        self.turn = 0
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def get_question(self, patient_response: Optional[str] = None) -> Tuple[str, bool]:
        """
        Generate the next doctor question or final diagnosis.
        Returns (text, is_final_diagnosis).
        """
        if patient_response:
            self.messages.append({"role": "user", "content": sanitize_openai_text(f"Patient: {patient_response}")})

        self.turn += 1
        if self.turn >= self.max_turns:
            self.messages.append({
                "role": "user",
                "content": STANDALONE_MAX_TURNS_MESSAGE,
            })

        text, usage = self.llm.chat(self.messages, max_tokens=self.config.standalone_max_tokens)
        self.messages.append({"role": "assistant", "content": text})
        self.token_usage["prompt_tokens"] += usage["prompt_tokens"]
        self.token_usage["completion_tokens"] += usage["completion_tokens"]

        is_final = "[DDX]" in text or "[ABSTAIN]" in text
        return text, is_final

    def parse_ddx(self, text: str) -> List[str]:
        """Extract diagnoses from [DDX] response."""
        ddx_part = text.split("[DDX]")[-1].strip()
        diagnoses = [d.strip().strip('.-').strip() for d in ddx_part.split(",")]
        return [d for d in diagnoses if d]

    def parse_possible(self, text: str) -> List[str]:
        """Extract possible diagnoses from [POSSIBLE] response if present."""
        if "[POSSIBLE]" not in text:
            return []
        possible_part = text.split("[POSSIBLE]")[-1].strip()
        diagnoses = [d.strip().strip('.-').strip() for d in possible_part.split(",")]
        return [d for d in diagnoses if d]


class MoBayesDoctor:
    """
    MoBayes as the doctor: Bayesian engine + EIG + LLM question generation.
    """

    def __init__(self, data_dir: str, llm_model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None, max_questions: int = 20,
                 no_taxonomy: bool = False,
                 no_abstention: bool = False,
                 eig_topk_size: int = 3,
                 eig_topk_weight: float = 0.0,
                 eig_focus_conf_threshold: float = 1.01,
                 base_url: Optional[str] = None,
                 doctor_temperature: Optional[float] = None,
                 prior_mode: str = "empirical",
                 config: MoBayesConfig = None):
        self.config = config or MoBayesConfig()
        self.data_dir = data_dir
        self.max_questions = max_questions
        self.llm_model = llm_model
        self.prior_mode = prior_mode

        # LLM interface — doctor_temperature overrides config default if provided
        temp = doctor_temperature if doctor_temperature is not None else self.config.llm_temperature
        self.llm = OpenAILLMClient(model=llm_model, api_key=api_key, temperature=temp, base_url=base_url)
        self.templates = PromptTemplates()
        self.parser = StructuredParser(self.llm, self.templates, config=self.config)

        self.disease_names = load_disease_names(data_dir)

        # Load world model and policy
        self.wm, self.leaf_diseases = load_world_model_from_schema(
            data_dir, prior_mode=prior_mode)
        if no_taxonomy:
            self.wm.taxonomy = None
        self.policy = EIGPolicy(
            topk_size=eig_topk_size,
            topk_weight=eig_topk_weight,
            focus_conf_threshold=eig_focus_conf_threshold,
        )
        self.stopping = StoppingRules(
            confidence_threshold=self.config.confidence_threshold,
            max_questions=max_questions,
            min_questions_before_abstain=999 if no_abstention else self.config.min_questions_before_abstain,
        )
        self.no_taxonomy = no_taxonomy
        self.no_abstention = no_abstention
        self.turn = 0
        self.feature_history = []

    def reset(self, patient_meta: Dict = None):
        """Reset for a new patient."""
        self.wm, self.leaf_diseases = load_world_model_from_schema(
            self.data_dir,
            prior_mode=self.prior_mode,
            patient_meta=patient_meta,
        )
        if self.no_taxonomy:
            self.wm.taxonomy = None
        self.turn = 0
        self.feature_history = []

    def disease_id_to_name(self, disease_id: str) -> str:
        """Map disease ID to human-readable name."""
        return self.disease_names.get(disease_id, disease_id)

    def get_next_question(self) -> Optional[Dict]:
        """
        Select next feature via EIG, generate natural language question.
        Returns None if stopping criteria met.
        """
        # Check stopping
        num_asked = len(self.wm.evidence())
        stopped, reason = self.stopping.should_stop_wm(self.wm, num_asked)
        if stopped:
            return {"status": "stopped", "reason": reason.name}

        # Select feature
        next_feature = self.policy.select_feature(self.wm)
        if next_feature is None:
            return {"status": "stopped", "reason": "FEATURES_EXHAUSTED"}

        feature_schema = self.wm.likelihood_model.schema.get(next_feature)
        feature_name = feature_schema.metadata.get('name', next_feature) if feature_schema else next_feature
        schema_values = feature_schema.values if feature_schema else []

        # Generate natural language question
        question = self._generate_question(next_feature, feature_name, schema_values)

        self.turn += 1
        return {
            "status": "ongoing",
            "feature_id": next_feature,
            "feature_name": feature_name,
            "schema_values": schema_values,
            "question": question,
            "entropy": self.wm.entropy(),
            "policy_info": dict(self.policy.last_selection_info),
        }

    def process_response(self, feature_id: str, patient_response: str,
                         schema_values: List[str]) -> Tuple[str, float]:
        """
        Parse patient response into a feature value and update the world model.
        Uses soft Gaussian update for numeric features, hard update for others.
        Handles negated [nowhere] features by inverting the extracted value.
        Returns (extracted_value, jeffrey_weight).
        """
        feature_schema = self.wm.likelihood_model.schema.get(feature_id)
        feature_name = feature_schema.metadata.get('name', feature_id) if feature_schema else feature_id

        extracted, confidence = self._extract_value(patient_response, feature_id, feature_name, schema_values)

        # Numeric features: soft Gaussian update (8 ≈ 7,9 but ≠ 2)
        if is_numeric_scale(schema_values) and extracted not in ("unknown", "absent"):
            self.wm.soft_update(feature_id, extracted, sigma=self.config.numeric_gaussian_sigma, confidence=confidence)
        else:
            self.wm.update(feature_id, extracted, confidence=confidence, provenance="benchmark_extraction")

        self.feature_history.append({
            "feature_id": feature_id,
            "feature_name": feature_name,
            "patient_said": patient_response,
            "extracted_value": extracted,
            "confidence": confidence,
        })
        return extracted, confidence

    def get_results(self) -> Dict:
        """Get final diagnostic results including Phase-B marginals."""
        posterior = self.wm.posterior()
        top_idx = int(np.argmax(posterior))
        top_id = self.leaf_diseases[top_idx]
        # Ranked list of disease names
        ranked_indices = np.argsort(posterior)[::-1]
        ranked_names = [self.disease_id_to_name(self.leaf_diseases[i]) for i in ranked_indices]
        ranked_ids = [self.leaf_diseases[i] for i in ranked_indices]

        result = {
            "posterior": posterior,
            "top_disease": self.disease_id_to_name(top_id),
            "top_disease_id": top_id,
            "top_diseases": ranked_names[:5],
            "top_disease_ids": ranked_ids[:5],
            "confidence": float(np.max(posterior)),
            "margin": float(np.max(posterior) - posterior[ranked_indices[1]]) if posterior.size > 1 else float(np.max(posterior)),
            "entropy": self.wm.entropy(),
            "leaf_diseases": self.leaf_diseases,
        }

        return result

    def _generate_question(self, feature_id: str, feature_name: str,
                           schema_values: List[str],
                           last_user_query: Optional[str] = None) -> str:
        """Generate a natural language question for the given feature."""
        display_name = resolve_negated_name(feature_id, feature_name)
        display_name, display_schema, _ = prepare_display(
            feature_id, display_name, schema_values
        )

        max_prob = float(np.max(self.wm.posterior()))
        confidence_level = ("low" if max_prob < self.config.confidence_display_low
                            else "medium" if max_prob < self.config.confidence_display_high
                            else "high")

        prompt = self.templates.render_question_prompt({
            "next_feature": display_name,
            "answer_schema": display_schema,
            "confidence_level": confidence_level,
            "last_user_query": last_user_query,
        })
        # max_tokens=2000: reasoning/thinking models (e.g. Qwen3.5) use internal
        # thinking tokens that count toward the budget. 500 is too low for them.
        question = self.llm.generate(prompt, max_tokens=2000).strip()
        if not question:
            # Fallback: display_name is already a question (e.g. "Do you have a cough?")
            question = display_name
        return question

    def bulk_extract_from_text(self, text: str) -> Dict[str, str]:
        """
        Extract multiple features from free text (chief complaint / initial statement).
        Uses StructuredParser.bulk_extract with render_bulk_intake_prompt.
        Mirrors main_loop.run_initial_intake().
        Returns dict of {feature_id: matched_value}.
        """
        # Build feature list with readable labels for LLM
        feature_list = []
        for fid, schema in self.wm.likelihood_model.schema.items():
            name = schema.metadata.get('name', fid)
            _, display_vals, _ = prepare_display(fid, name, schema.values)
            feature_list.append({"id": fid, "name": name, "values": display_vals})

        # StructuredParser: bulk extraction from free text
        extraction_result = self.parser.bulk_extract(text, feature_list)

        # Process results
        updated = {}
        for fid, raw_val in extraction_result.items():
            if fid == "demographics" or fid not in self.wm.likelihood_model.schema:
                continue

            # Parse value + confidence from all LLM output formats
            if isinstance(raw_val, dict):
                val = str(raw_val.get("value", raw_val))
                conf_label = raw_val.get("confidence", "likely")
            else:
                val = str(raw_val).strip()
                if '|' in val:
                    parts = val.split('|', 1)
                    val = parts[0].strip()
                    conf_label = parts[1].strip()
                    if conf_label not in VALID_LABELS:
                        conf_label = "likely"
                else:
                    conf_label = "likely"

            schema = self.wm.likelihood_model.schema[fid]

            # DDXPlus normalization (V_code reverse, numeric clamp, negation)
            val = normalize_extracted_value(fid, val, schema.values)

            normalized = enforce_schema(val, schema.values)
            if normalized == "unknown":
                continue

            confidence = resolve_confidence(conf_label)
            confidence = min(confidence, self.config.bulk_extraction_confidence_cap)
            self.wm.update(fid, normalized, confidence=confidence, provenance="bulk_extraction")
            updated[fid] = normalized

        return updated

    def _extract_value(self, patient_response: str, feature_id: str,
                       feature_name: str, schema_values: List[str]) -> Tuple[str, float]:
        """
        Extract feature value + confidence using StructuredParser.
        Returns (value, jeffrey_weight) where jeffrey_weight is in [0, 1].
        """
        display_name = resolve_negated_name(feature_id, feature_name)
        display_name, display_schema, _ = prepare_display(
            feature_id, display_name, schema_values
        )

        # StructuredParser: LLM extraction + heuristic fallback
        value, conf_label = self.parser.extract_feature_value_with_confidence(
            patient_response, display_name, display_schema
        )

        # DDXPlus normalization (V_code reverse, numeric clamp, negation)
        value = normalize_extracted_value(feature_id, value, schema_values)

        # Enforce against original schema
        value = enforce_schema(value, schema_values)

        return value, resolve_confidence(conf_label)



# ── Benchmark Orchestrator ───────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    patient_id: str
    ground_truth_id: str
    ground_truth: str
    predicted: str
    predicted_set: List[str]
    top1_hit: bool
    top3_hit: bool
    turns: int
    confidence: float
    final_margin: float
    final_entropy: float
    stop_reason: str
    doctor_mode: str
    persona: Dict
    abstained: bool = False
    leaf_diseases: List[str] = field(default_factory=list)
    final_posterior: List[float] = field(default_factory=list)
    marginals: Dict = field(default_factory=dict)
    token_usage: Dict = field(default_factory=dict)
    dialog_history: List[Dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def _record_belief_snapshot(entry: Dict, doctor: 'MoBayesDoctor'):
    """Record per-turn belief state into a dialog entry for post-hoc analysis."""
    posterior = doctor.wm.posterior()
    top5_idx = np.argsort(posterior)[::-1][:5]
    ranked = np.argsort(posterior)[::-1]
    entry["top5_posterior"] = [(doctor.leaf_diseases[i], float(posterior[i])) for i in top5_idx]
    entry["confidence_after"] = float(posterior[ranked[0]])
    entry["margin_after"] = float(posterior[ranked[0]] - posterior[ranked[1]]) if posterior.size > 1 else float(posterior[ranked[0]])
    entry["entropy_after"] = float(doctor.wm.entropy())


def topk_hit(predicted_diseases: List[str], ground_truth: str, k: int) -> bool:
    """Check if ground truth is in top-k predictions (exact match)."""
    gt_lower = ground_truth.lower()
    for d in predicted_diseases[:k]:
        if d.lower() == gt_lower:
            return True
    return False


def normalize_standalone_predictions(predictions: List[str],
                                      icd_map: Dict[str, str],
                                      disease_names: List[str],
                                      llm_client) -> List[str]:
    """
    Map free-text standalone LLM predictions to KB disease names.

    Uses LLM to match free-text diagnoses directly to the KB disease name list.
    Unmatched predictions are kept as-is (counted as incorrect during evaluation).

    Args:
        predictions: Free-text disease names from standalone doctor
        icd_map: Dict mapping ICD-10 code → disease name (unused, kept for API compat)
        disease_names: List of KB disease names
        llm_client: OpenAI LLM client for matching
    Returns:
        List of matched KB disease names
    """
    if not predictions or not disease_names:
        return predictions

    prompt = PromptTemplates.render_disease_matching_prompt(predictions, disease_names)

    try:
        response = llm_client.generate(prompt, max_tokens=300)
        matched = []
        # Build lookup for case-insensitive matching
        name_lookup = {n.lower(): n for n in disease_names}

        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Parse "1. Pneumonia" or "1. NO_MATCH" format
            parts = line.split(".", 1)
            if len(parts) == 2:
                name = parts[1].strip()
            else:
                name = line.strip()

            if name.upper() == "NO_MATCH":
                # Keep original prediction (will be counted as FP)
                idx = len(matched)
                if idx < len(predictions):
                    matched.append(predictions[idx])
            elif name.lower() in name_lookup:
                matched.append(name_lookup[name.lower()])
            else:
                # Exact match failed — keep original
                idx = len(matched)
                if idx < len(predictions):
                    matched.append(predictions[idx])

        # Pad with originals if LLM returned fewer lines
        while len(matched) < len(predictions):
            matched.append(predictions[len(matched)])
        return matched[:len(predictions)]
    except Exception:
        return predictions


def run_mobayes_session(profile: Dict, data_dir: str,
                              patient_model: str = "gpt-4o-mini",
                              doctor_llm: str = "gpt-4o-mini",
                              no_taxonomy: bool = False,
                              no_abstention: bool = False,
                              force_max_turns: bool = True,
                              eig_topk_size: int = 3,
                              eig_topk_weight: float = 0.0,
                              eig_focus_conf_threshold: float = 1.01,
                              doctor_base_url: Optional[str] = None,
                              doctor_api_key: Optional[str] = None,
                              doctor_temperature: Optional[float] = None,
                              prior_mode: str = "empirical",
                              config: MoBayesConfig = None) -> BenchmarkResult:
    """Run one diagnostic session with MoBayes as doctor."""
    config = config or MoBayesConfig()
    start = time.time()

    # Support DDXPlus and LLM KB profile formats
    if "_ddxplus_meta" in profile:
        ground_truth_ids = [profile["_ddxplus_meta"]["ground_truth_ids"][0]]
        patient_id = profile.get("hadm_id", "unknown")
    elif "_llmkb_meta" in profile:
        ground_truth_ids = profile["_llmkb_meta"]["ground_truth_ids"]
        patient_id = profile.get("hadm_id", profile["_llmkb_meta"].get("patient_id", "unknown"))
    else:
        ground_truth_ids = profile.get("ground_truth", [])
        patient_id = profile.get("patient_id", "unknown")

    # Display label for logs (resolve IDs → names)
    disease_names_map = load_disease_names(data_dir)
    ground_truth = ",".join(disease_names_map.get(gid, gid) for gid in ground_truth_ids)

    persona = profile.get("Persona Settings", {}) or profile.get("persona", {})

    # Initialize agents
    patient = SimplePatientAgent(profile, model=patient_model, config=config)
    doctor = MoBayesDoctor(
        data_dir,
        llm_model=doctor_llm,
        api_key=doctor_api_key,
        no_taxonomy=no_taxonomy,
        no_abstention=no_abstention,
        eig_topk_size=eig_topk_size,
        eig_topk_weight=eig_topk_weight,
        eig_focus_conf_threshold=eig_focus_conf_threshold,
        base_url=doctor_base_url,
        doctor_temperature=doctor_temperature,
        prior_mode=prior_mode,
        config=config,
    )
    if force_max_turns:
        # Override: disable ALL early stopping, only MAX_QUESTIONS stops
        doctor.stopping = StoppingRules(
            confidence_threshold=1.01,  # impossible to reach
            max_questions=doctor.stopping.max_questions,
            min_questions_before_abstain=999,
        )
    # Build patient_meta for conditional prior (age, sex)
    _meta = profile.get("meta", {})
    if not _meta:
        _meta = {"age": profile.get("age", 30), "sex": profile.get("gender", "male")}
    doctor.reset(_meta)

    dialog = []
    stop_reason = "MAX_TURNS"

    # Opening: patient describes chief complaint
    chief = profile.get("Current Visit", {}).get("chiefcomplaint", "I'm not feeling well")
    dialog.append({"role": "patient", "content": chief, "turn": 0})

    # Bulk extraction from chief complaint (mirrors main_loop.run_initial_intake)
    bulk_results = doctor.bulk_extract_from_text(chief)
    if bulk_results:
        dialog.append({"role": "system", "content": f"Bulk extracted: {bulk_results}", "turn": 0})

    # Main diagnostic loop
    for turn in range(config.max_questions + 10):  # safety margin; stopping rules handle actual limit
        q_result = doctor.get_next_question()
        if q_result["status"] == "stopped":
            stop_reason = q_result["reason"]
            break

        question = q_result["question"]
        dialog.append({"role": "doctor", "content": question,
                        "feature_id": q_result["feature_id"], "turn": turn + 1,
                        "policy_info": q_result.get("policy_info", {})})

        # Patient responds
        patient_response = patient.respond(question)
        dialog.append({"role": "patient", "content": patient_response, "turn": turn + 1})

        # Parse and update
        extracted, confidence = doctor.process_response(
            q_result["feature_id"], patient_response, q_result["schema_values"]
        )
        dialog[-1]["extracted_value"] = extracted
        dialog[-1]["confidence"] = confidence

        _record_belief_snapshot(dialog[-1], doctor)

        # Follow-up: if unknown/clarification_needed, re-ask once with clarification
        if extracted in ("unknown", "clarification_needed"):
            followup_q = doctor._generate_question(
                q_result["feature_id"], q_result["feature_name"],
                q_result["schema_values"], last_user_query=patient_response,
            )
            dialog.append({"role": "doctor", "content": followup_q,
                            "feature_id": q_result["feature_id"], "turn": turn + 1,
                            "is_followup": True})

            followup_response = patient.respond(followup_q)
            dialog.append({"role": "patient", "content": followup_response, "turn": turn + 1,
                            "is_followup": True})

            extracted2, confidence2 = doctor.process_response(
                q_result["feature_id"], followup_response, q_result["schema_values"]
            )
            dialog[-1]["extracted_value"] = extracted2
            dialog[-1]["confidence"] = confidence2
            _record_belief_snapshot(dialog[-1], doctor)

    # Get final results
    results = doctor.get_results()
    top_disease_ids = results["top_disease_ids"]

    elapsed = time.time() - start

    # Match by ID: check if any ground_truth ID is in top-k predicted IDs
    gt_set = set(ground_truth_ids)

    return BenchmarkResult(
        patient_id=patient_id,
        ground_truth_id=ground_truth_ids if isinstance(ground_truth_ids, list) else [ground_truth_ids],
        ground_truth=ground_truth,
        predicted=results["top_disease"],
        predicted_set=list(results["top_diseases"]),
        top1_hit=bool(gt_set & set(top_disease_ids[:1])),
        top3_hit=bool(gt_set & set(top_disease_ids[:3])),
        turns=len([d for d in dialog if d["role"] == "doctor"]),
        confidence=results["confidence"],
        final_margin=results["margin"],
        final_entropy=results["entropy"],
        stop_reason=stop_reason,
        doctor_mode="mobayes",
        persona=persona,
        abstained=False,
        leaf_diseases=list(results["leaf_diseases"]),
        final_posterior=[float(x) for x in results["posterior"]],
        marginals=results.get("marginals", {}),
        token_usage={
            "doctor": doctor.llm.total_usage.copy(),
            "patient": patient.token_usage.copy(),
        },
        dialog_history=dialog,
        elapsed_seconds=elapsed,
    )


def run_standalone_session(profile: Dict, patient_model: str = "gpt-4o-mini",
                            doctor_model: str = "gpt-4o",
                            disease_names: Optional[List[str]] = None,
                            icd_map: Optional[Dict[str, str]] = None,
                            doctor_base_url: Optional[str] = None,
                            doctor_api_key: Optional[str] = None,
                            config: MoBayesConfig = None,
                            doctor_override: Optional["StandaloneDoctorAgent"] = None,
                            mode_label: str = "standalone") -> BenchmarkResult:
    """Run one diagnostic session with standalone LLM doctor (no Bayesian engine).

    `doctor_override` lets callers pass a pre-instantiated subclass (e.g. ClosedWorldDoctorAgent)
    to reuse this loop without duplicating it. `mode_label` is recorded in the BenchmarkResult.
    """
    config = config or MoBayesConfig()
    start = time.time()

    # Support DDXPlus and LLM KB profile formats
    if "_ddxplus_meta" in profile:
        # DDXPlus format
        ground_truth = profile["_ddxplus_meta"]["diagnosis_name"]
        ground_truth_id = [profile["_ddxplus_meta"]["ground_truth_ids"][0]]
        patient_id = profile.get("hadm_id", "unknown")
    else:
        # LLM KB format (same structure as DDXPlus)
        ground_truth = profile["_llmkb_meta"]["diagnosis_name"]
        ground_truth_id = profile["_llmkb_meta"]["ground_truth_ids"]
        patient_id = profile.get("hadm_id", profile["_llmkb_meta"].get("patient_id", "unknown"))

    persona = profile.get("Persona Settings", {}) or profile.get("persona", {})

    patient = SimplePatientAgent(profile, model=patient_model, config=config)
    if doctor_override is not None:
        doctor = doctor_override
    else:
        doctor = StandaloneDoctorAgent(model=doctor_model, api_key=doctor_api_key,
                                        base_url=doctor_base_url,
                                        config=config)

    dialog = []
    stop_reason = "MAX_TURNS"

    # Doctor greets
    greeting = "Hello, I'm your doctor today. What brings you to the emergency department?"
    dialog.append({"role": "doctor", "content": greeting, "turn": 0})

    # Patient describes complaint
    patient_response = patient.respond(greeting)
    dialog.append({"role": "patient", "content": patient_response, "turn": 0})

    for turn in range(config.max_questions + 5):
        question, is_final = doctor.get_question(patient_response)
        dialog.append({"role": "doctor", "content": question, "turn": turn + 1})

        if is_final:
            stop_reason = "DDX_PROVIDED"
            break

        patient_response = patient.respond(question)
        dialog.append({"role": "patient", "content": patient_response, "turn": turn + 1})

    # Parse DDX/abstention from final doctor message
    final_msg = dialog[-1]["content"] if dialog[-1]["role"] == "doctor" else ""
    abstained = "[ABSTAIN]" in final_msg
    if "[DDX]" in final_msg:
        predicted_diseases = doctor.parse_ddx(final_msg)
        stop_reason = "DDX_PROVIDED"
    elif abstained:
        predicted_diseases = doctor.parse_possible(final_msg)
        stop_reason = "ABSTAINED"
    else:
        # Force final decision
        ddx_msg, _ = doctor.get_question("Please provide your final [DDX] or [ABSTAIN] now.")
        dialog.append({"role": "doctor", "content": ddx_msg, "turn": len(dialog)})
        abstained = "[ABSTAIN]" in ddx_msg
        if "[DDX]" in ddx_msg:
            predicted_diseases = doctor.parse_ddx(ddx_msg)
            stop_reason = "DDX_PROVIDED"
        else:
            predicted_diseases = doctor.parse_possible(ddx_msg)
            stop_reason = "ABSTAINED"
            abstained = True

    # Normalize free-text predictions to DDXPlus disease names via ICD-10 matching
    raw_predictions = list(predicted_diseases)
    if (icd_map or disease_names) and predicted_diseases:
        match_client = OpenAILLMClient(model=config.default_model, temperature=config.llm_temperature)
        if icd_map:
            predicted_diseases = normalize_standalone_predictions(
                predicted_diseases, icd_map, disease_names or [], match_client
            )
        dialog.append({
            "role": "system",
            "content": f"Raw DDX: {raw_predictions} → Normalized (ICD-10): {predicted_diseases}",
            "turn": -1
        })

    elapsed = time.time() - start

    # Multi-label aware hit detection (works for DDXPlus and LLM KB)
    # Compare both GT IDs and GT names against predicted names
    gt_ids = ground_truth_id if isinstance(ground_truth_id, list) else [ground_truth_id]
    gt_names = [g for g in ground_truth.split(",") if g.strip()] if ground_truth else []
    normalize = lambda s: s.lower().strip().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    pred_lower = [normalize(d) for d in predicted_diseases]
    gt_lower = set(normalize(g) for g in gt_ids) | set(normalize(g) for g in gt_names)
    t1_hit = bool(gt_lower & set(pred_lower[:1]))
    t3_hit = bool(gt_lower & set(pred_lower[:3]))

    return BenchmarkResult(
        patient_id=patient_id,
        ground_truth_id=ground_truth_id if isinstance(ground_truth_id, list) else [ground_truth_id],
        ground_truth=ground_truth,
        predicted="ABSTAIN" if abstained else (predicted_diseases[0] if predicted_diseases else "Unknown"),
        predicted_set=list(predicted_diseases),
        top1_hit=t1_hit,
        top3_hit=t3_hit,
        turns=len([d for d in dialog if d["role"] == "doctor"]),
        confidence=0.0,  # Standalone doesn't provide confidence
        final_margin=0.0,
        final_entropy=0.0,
        stop_reason=stop_reason,
        doctor_mode=mode_label,
        persona=persona,
        abstained=abstained,
        token_usage={
            "doctor": doctor.token_usage.copy(),
            "patient": patient.token_usage.copy(),
        },
        dialog_history=dialog,
        elapsed_seconds=elapsed,
    )


# ── Batch runner ─────────────────────────────────────────────────────────────

def run_benchmark(profiles_path: str,
                  data_dir: str,
                  doctor_mode: str = "mobayes",
                  n: Optional[int] = None,
                  patient_model: str = "gpt-4o-mini",
                  doctor_model: str = "gpt-4o-mini",
                  output_dir: Optional[str] = None,
                  no_taxonomy: bool = False,
                  no_abstention: bool = False,
                  force_max_turns: bool = True,
                  eig_topk_size: int = 3,
                  eig_topk_weight: float = 0.0,
                  eig_focus_conf_threshold: float = 1.01,
                  parallel: int = 1,
                  doctor_base_url: Optional[str] = None,
                  doctor_api_key: Optional[str] = None,
                  doctor_temperature: Optional[float] = None,
                  provide_disease_list: bool = False,
                  prior_mode: str = "empirical",
                  config: MoBayesConfig = None) -> List[BenchmarkResult]:
    """Run the full benchmark over a set of patient profiles."""
    config = config or MoBayesConfig()
    with open(profiles_path) as f:
        profiles = json.load(f)

    skipped_other = sum(
        1 for p in profiles
        if ((p.get("_ddxplus_meta", {}).get("ground_truth_ids") or [None])[0] == "d_other")
    )
    profiles = [
        p for p in profiles
        if ((p.get("_ddxplus_meta", {}).get("ground_truth_ids") or [None])[0] != "d_other")
    ]

    if n:
        profiles = profiles[:n]

    disease_name_list, icd_map = load_icd_map(data_dir)

    print(f"\n{'='*60}")
    print(f"  Benchmark | mode={doctor_mode} | n={len(profiles)} | patient={patient_model}")
    if skipped_other:
        print(f"  Skipped d_other profiles: {skipped_other}")
    print(f"{'='*60}\n")

    # Incremental JSONL output: write each result as it completes
    jsonl_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if doctor_mode == "mobayes":
            mode_label = doctor_mode
        elif doctor_mode == "closed_world":
            mode_label = f"cw_{doctor_model.replace('/', '_')}"
        elif doctor_mode == "chain_of_thought":
            mode_label = f"cot_{doctor_model.replace('/', '_')}"
        elif doctor_mode == "savage_differential":
            mode_label = f"savage_dd_{doctor_model.replace('/', '_')}"
        elif doctor_mode == "savage_bayesian":
            mode_label = f"savage_bay_{doctor_model.replace('/', '_')}"
        else:
            mode_label = f"standalone_{doctor_model.replace('/', '_')}"
        fname = f"benchmark_{mode_label}_{timestamp}.jsonl"
        jsonl_path = os.path.join(output_dir, fname)

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    write_lock = threading.Lock()
    progress = [0]
    total = len(profiles)
    results = []

    def _process_one(profile):
        """Run a single patient session (thread-safe)."""
        if doctor_mode == "mobayes":
            return run_mobayes_session(profile, data_dir,
                                            patient_model=patient_model,
                                            doctor_llm=doctor_model,
                                            no_taxonomy=no_taxonomy,
                                            no_abstention=no_abstention,
                                            force_max_turns=force_max_turns,
                                            eig_topk_size=eig_topk_size,
                                            eig_topk_weight=eig_topk_weight,
                                            eig_focus_conf_threshold=eig_focus_conf_threshold,
                                            doctor_base_url=doctor_base_url,
                                            doctor_api_key=doctor_api_key,
                                            doctor_temperature=doctor_temperature,
                                            prior_mode=prior_mode,
                                            config=config)
        elif doctor_mode == "closed_world":
            from baselines.closed_world_doctor import ClosedWorldDoctorAgent
            cw_doctor = ClosedWorldDoctorAgent(
                disease_names=disease_name_list,
                model=doctor_model,
                api_key=doctor_api_key,
                base_url=doctor_base_url,
                config=config,
            )
            return run_standalone_session(profile,
                                          patient_model=patient_model,
                                          doctor_model=doctor_model,
                                          disease_names=disease_name_list,
                                          icd_map=icd_map,
                                          doctor_base_url=doctor_base_url,
                                          doctor_api_key=doctor_api_key,
                                          config=config,
                                          doctor_override=cw_doctor,
                                          mode_label="closed_world")
        elif doctor_mode == "chain_of_thought":
            from baselines.cot_doctor import ChainOfThoughtDoctorAgent
            cot_doctor = ChainOfThoughtDoctorAgent(
                disease_names=disease_name_list,
                model=doctor_model,
                api_key=doctor_api_key,
                base_url=doctor_base_url,
                config=config,
            )
            return run_standalone_session(profile,
                                          patient_model=patient_model,
                                          doctor_model=doctor_model,
                                          disease_names=disease_name_list,
                                          icd_map=icd_map,
                                          doctor_base_url=doctor_base_url,
                                          doctor_api_key=doctor_api_key,
                                          config=config,
                                          doctor_override=cot_doctor,
                                          mode_label="chain_of_thought")
        elif doctor_mode in ("savage_differential", "savage_bayesian"):
            from baselines.savage_doctor import (
                DifferentialDiagnosisDoctorAgent,
                BayesianReasoningDoctorAgent,
            )
            cls = (DifferentialDiagnosisDoctorAgent
                   if doctor_mode == "savage_differential"
                   else BayesianReasoningDoctorAgent)
            savage_doctor = cls(
                disease_names=disease_name_list,
                model=doctor_model,
                api_key=doctor_api_key,
                base_url=doctor_base_url,
                config=config,
            )
            return run_standalone_session(profile,
                                          patient_model=patient_model,
                                          doctor_model=doctor_model,
                                          disease_names=disease_name_list,
                                          icd_map=icd_map,
                                          doctor_base_url=doctor_base_url,
                                          doctor_api_key=doctor_api_key,
                                          config=config,
                                          doctor_override=savage_doctor,
                                          mode_label=doctor_mode)
        else:
            return run_standalone_session(profile,
                                         patient_model=patient_model,
                                         doctor_model=doctor_model,
                                         disease_names=disease_name_list,
                                         icd_map=icd_map,
                                         doctor_base_url=doctor_base_url,
                                         doctor_api_key=doctor_api_key,
                                         config=config)

    def _handle_result(profile, r):
        """Log and save result (under lock)."""
        with write_lock:
            progress[0] += 1
            status = "ABSTAIN" if r.abstained else ("HIT" if r.top1_hit else "MISS")
            pid = profile.get('hadm_id') or profile.get('patient_id', 'unknown')
            gt_name = profile.get('_ddxplus_meta', {}).get('diagnosis_name') or r.ground_truth
            print(f"  [{progress[0]}/{total}] {pid} "
                  f"(GT: {gt_name}) "
                  f"{status} | pred={r.predicted} | conf={r.confidence:.1%} | "
                  f"turns={r.turns} | {r.elapsed_seconds:.1f}s")
            results.append(r)
            if jsonl_path:
                _append_result(jsonl_path, r)

    if parallel <= 1:
        # Sequential execution (original behavior)
        for profile in profiles:
            try:
                r = _process_one(profile)
                _handle_result(profile, r)
            except Exception as e:
                with write_lock:
                    progress[0] += 1
                    pid = profile.get('hadm_id') or profile.get('patient_id', 'unknown')
                    print(f"  [{progress[0]}/{total}] {pid} ERROR: {e}")
    else:
        # Parallel execution
        print(f"  Running with {parallel} parallel workers\n")
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            future_to_profile = {
                executor.submit(_process_one, p): p for p in profiles
            }
            for future in as_completed(future_to_profile):
                profile = future_to_profile[future]
                try:
                    r = future.result()
                    _handle_result(profile, r)
                except Exception as e:
                    with write_lock:
                        progress[0] += 1
                        pid = profile.get('hadm_id') or profile.get('patient_id', 'unknown')
                        print(f"  [{progress[0]}/{total}] {pid} ERROR: {e}")

    # Summary
    if results:
        print_summary(results, doctor_mode)
        if jsonl_path:
            print(f"  Results saved to: {jsonl_path}")

    return results


def print_summary(results: List[BenchmarkResult], mode: str):
    """Print benchmark summary statistics."""
    n = len(results)
    top1 = sum(r.top1_hit for r in results) / n
    top3 = sum(r.top3_hit for r in results) / n
    mean_turns = np.mean([r.turns for r in results])
    mean_conf = np.mean([r.confidence for r in results])
    mean_entropy = np.mean([r.final_entropy for r in results])
    mean_time = np.mean([r.elapsed_seconds for r in results])
    abstain_rate = np.mean([r.abstained for r in results])

    print(f"\n{'='*60}")
    print(f"  SUMMARY | {mode} | n={n}")
    print(f"{'='*60}")
    print(f"  Top-1 Accuracy   : {top1:.1%}")
    print(f"  Top-3 Accuracy   : {top3:.1%}")
    print(f"  Abstain Rate     : {abstain_rate:.1%}")
    print(f"  Mean Turns       : {mean_turns:.1f}")
    if mode == "mobayes":
        print(f"  Mean Confidence  : {mean_conf:.1%}")
        print(f"  Mean Entropy     : {mean_entropy:.2f} bits")
    print(f"  Mean Time/Patient: {mean_time:.1f}s")

    # Stop reason distribution
    reason_counts = {}
    for r in results:
        reason_counts[r.stop_reason] = reason_counts.get(r.stop_reason, 0) + 1
    print(f"\n  Stop Reasons:")
    for sr, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"    {sr:<30} {cnt:>3} ({cnt/n:.0%})")

    print(f"{'='*60}\n")


def _result_to_record(r: BenchmarkResult) -> dict:
    """Convert a BenchmarkResult to a JSON-serializable dict."""
    return {
        "patient_id": r.patient_id,
        "ground_truth_id": r.ground_truth_id,
        "ground_truth": r.ground_truth,
        "predicted": r.predicted,
        "predicted_set": r.predicted_set,
        "top1_hit": r.top1_hit,
        "top3_hit": r.top3_hit,
        "turns": r.turns,
        "confidence": r.confidence,
        "final_margin": r.final_margin,
        "final_entropy": r.final_entropy,
        "stop_reason": r.stop_reason,
        "doctor_mode": r.doctor_mode,
        "persona": r.persona,
        "abstained": r.abstained,
        "leaf_diseases": r.leaf_diseases,
        "final_posterior": r.final_posterior,
        "marginals": r.marginals,
        "token_usage": r.token_usage,
        "elapsed_seconds": r.elapsed_seconds,
        "dialog_history": r.dialog_history,
    }


def _append_result(jsonl_path: str, r: BenchmarkResult):
    """Append a single result to the JSONL file (incremental write)."""
    with open(jsonl_path, 'a') as f:
        f.write(json.dumps(_result_to_record(r)) + "\n")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    _defaults = MoBayesConfig()
    parser = argparse.ArgumentParser(description='PatientSim Benchmark for MoBayes')
    parser.add_argument('--profiles', default=None,
                        help='Path to profiles.json (default: benchmark_data/profiles.json)')
    parser.add_argument('--data-dir', default=None,
                        help='Path to DDXPlus data dir (default: ddxplus_data/)')
    parser.add_argument('--doctor-mode', default='mobayes',
                        choices=['mobayes', 'standalone', 'closed_world',
                                 'chain_of_thought',
                                 'savage_differential', 'savage_bayesian'],
                        help='Doctor mode: mobayes (Bayesian), standalone (pure LLM), '
                             'closed_world (standalone + KB disease list in prompt), '
                             'chain_of_thought (closed_world + zero-shot CoT reasoning), '
                             'savage_differential / savage_bayesian (Savage et al. 2024 '
                             'diagnostic reasoning prompts)')
    parser.add_argument('--n', type=int, default=None,
                        help='Number of patients to run (default: all)')
    parser.add_argument('--patient-model', default=_defaults.default_model,
                        help='LLM model for patient simulator')
    parser.add_argument('--doctor-model', default=_defaults.default_model,
                        help='LLM model for doctor (question generation / standalone)')
    parser.add_argument('--output', default=None,
                        help='Output directory for results')
    parser.add_argument('--no-taxonomy', action='store_true',
                        help='Disable hierarchical EIG (use flat entropy instead)')
    parser.add_argument('--no-abstention', action='store_true',
                        help='Disable abstention rules when early stopping is enabled')
    parser.add_argument('--force-max-turns', dest='force_max_turns', action='store_true',
                        help='Disable ALL early stopping and run until MAX_QUESTIONS (default)')
    parser.add_argument('--allow-early-stop', dest='force_max_turns', action='store_false',
                        help='Enable confidence/entropy/abstention stopping instead of forcing MAX_QUESTIONS')
    parser.add_argument('--eig-topk-size', type=int, default=3,
                        help='Size of the focused top-k set used for the additional EIG discrimination term')
    parser.add_argument('--eig-topk-weight', type=float, default=0.0,
                        help='Lambda for focused top-k EIG. 0.0 keeps the original global EIG policy')
    parser.add_argument('--eig-focus-conf-threshold', type=float, default=1.01,
                        help='Only activate the focused top-k EIG term once max posterior reaches this threshold')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel patient workers (default: 1 = sequential)')
    parser.add_argument('--doctor-base-url', default=None,
                        help='Base URL for doctor LLM (e.g. http://localhost:8000/v1 for vLLM). Patient always uses OpenAI API.')
    parser.add_argument('--doctor-api-key', default=None,
                        help='API key for doctor LLM (overrides OPENAI_API_KEY for non-OpenAI providers like Together/OpenRouter)')
    parser.add_argument('--confidence-threshold', type=float, default=None,
                        help='Stopping confidence threshold (default: from config)')
    parser.add_argument('--doctor-temperature', type=float, default=None,
                        help='Override temperature for doctor LLM (e.g. 0.3 for Qwen). Default: use config value')
    parser.add_argument('--prior-mode', default='empirical',
                        choices=['empirical', 'uniform', 'conditional'],
                        help='Prior distribution mode: empirical (default), uniform (1/N), or conditional (age x sex)')
    parser.add_argument('--provide-disease-list', action='store_true', default=False,
                        help='[Deprecated] Disease names are now always used in the post-hoc mapping stage. This flag is kept for backward compatibility but has no effect.')
    parser.set_defaults(force_max_turns=True)
    args = parser.parse_args()

    _defaults = MoBayesConfig()
    if args.confidence_threshold is not None:
        _defaults.confidence_threshold = args.confidence_threshold
    data_dir = args.data_dir or os.path.join(BASE_DIR, _defaults.data_dir)
    profiles_path = args.profiles or os.path.join(BASE_DIR, _defaults.cases_dir, 'profiles.json')
    output_dir = args.output or os.path.join(BASE_DIR, _defaults.benchmark_results_dir)

    run_benchmark(
        profiles_path=profiles_path,
        data_dir=data_dir,
        doctor_mode=args.doctor_mode,
        n=args.n,
        patient_model=args.patient_model,
        doctor_model=args.doctor_model,
        output_dir=output_dir,
        no_taxonomy=args.no_taxonomy,
        no_abstention=args.no_abstention,
        force_max_turns=args.force_max_turns,
        eig_topk_size=args.eig_topk_size,
        eig_topk_weight=args.eig_topk_weight,
        eig_focus_conf_threshold=args.eig_focus_conf_threshold,
        parallel=args.parallel,
        doctor_base_url=args.doctor_base_url,
        doctor_api_key=args.doctor_api_key,
        doctor_temperature=args.doctor_temperature,
        provide_disease_list=args.provide_disease_list,
        prior_mode=args.prior_mode,
        config=_defaults,
    )


if __name__ == '__main__':
    main()
