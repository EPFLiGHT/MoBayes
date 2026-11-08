#!/usr/bin/env python3
"""Run MEDDxAgent (ddxdriver) as the doctor on one bench column.

MEDDxAgent's doctor/orchestrator stack is used as released; only the
benchmark class and the patient are ours, so every baseline faces the same
patient:

    --column patientsim   MoBayesDDxPlus50 + MoBayesPatientAgent (our PatientSim)
    --column htsim        DDxPlusHTSim50   + MEDDxAgent's own LLMPatient
    --column agentclinic  AgentClinic50    + ACMEDDxAgentPatient (AC PatientAgent)

Called by baselines/meddxagent/row.py with the MEDDxAgent venv. Writes
MEDDxAgent's results.json plus token_usage.json under
<experiment-dir>/<timestamp>/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEDDX_ROOT = REPO_ROOT.parent.parent / "meddxagent"
_RESULTS = REPO_ROOT / "data/benchmark_results"

# Per-column defaults (only used when the flag is not given).
_DEFAULTS = {
    "patientsim": dict(num_patients=50, rag_source="pubmed", fewshot_type="none", num_shots=5,
                       experiment_dir=_RESULTS / "meddxagent_patientsim"),
    "htsim": dict(num_patients=50, rag_source="pubmed", fewshot_type="dynamic", num_shots=5,
                  experiment_dir=_RESULTS / "ddxplus_htsim_meddxagent_raw"),
    "agentclinic": dict(num_patients=10, rag_source="off", fewshot_type="none", num_shots=0,
                        experiment_dir=_RESULTS / "agentclinic_10_meddxagent_raw"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--column", required=True, choices=sorted(_DEFAULTS))
    p.add_argument("--meddxagent-root", default=str(DEFAULT_MEDDX_ROOT),
                   help="Path to the local meddxagent checkout.")
    p.add_argument("--cases", "--profiles", "--cases-path", dest="cases", default=None,
                   help="patientsim: profiles.json; agentclinic: selected_cases.jsonl "
                        "(htsim reads DDXPLUS_HTSIM_CASES_PATH).")
    p.add_argument("--diseases", "--diseases-csv", dest="diseases", default=None,
                   help="agentclinic: the KB's diseases.csv.")
    p.add_argument("--num-patients", type=int, default=None)
    p.add_argument("--doctor-model", default="gpt-5.4-nano")
    p.add_argument("--patient-model", default="gpt-5.4-nano")
    p.add_argument("--doctor-base-url", default=None,
                   help="patientsim: base URL for doctor-side OpenAI-compatible calls.")
    p.add_argument("--patient-base-url", default=None,
                   help="patientsim: base URL for PatientSim OpenAI-compatible calls.")
    p.add_argument("--experiment-dir", default=None,
                   help="Output directory for raw MEDDxAgent artifacts.")
    p.add_argument("--rag-source", choices=("pubmed", "wikipedia", "closedworld", "off"), default=None)
    p.add_argument("--fewshot-type", choices=("none", "static", "dynamic"), default=None)
    p.add_argument("--num-shots", type=int, default=None)
    p.add_argument("--fewshot-embedding-model", default="BAAI/bge-base-en-v1.5",
                   help="patientsim/htsim: embedding model for dynamic few-shot retrieval.")
    p.add_argument("--source-pool-path", default="/tmp/agentclinic-tmp/agentclinic_medqa_extended.jsonl",
                   help="agentclinic: AC source pool for dynamic few-shot (213 cases).")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--questions-per-iteration", type=int, default=5)
    args = p.parse_args()
    for key, value in _DEFAULTS[args.column].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.column == "patientsim" and args.cases is None:
        args.cases = str(REPO_ROOT / "data/cases/ddxplus_patientsim_a/plain/profiles.json")
    if args.column == "agentclinic":
        args.cases = args.cases or str(REPO_ROOT / "data/cases/agentclinic_a/selected_cases.jsonl")
        args.diseases = args.diseases or str(REPO_ROOT / "data/kbs/agentclinic_a/diseases.csv")
    return args


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def maybe_override_model(cfg: dict, model_name: str, base_url: str | None = None) -> None:
    model_cfg = cfg.get("config", {}).get("model", {}).get("config", {})
    if not model_cfg:
        return
    model_cfg["model_name"] = model_name
    if base_url:
        model_cfg["base_url"] = base_url


def ensure_api_env() -> None:
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except Exception:
            pass
    # MEDDxAgent reads OAI_KEY
    if os.environ.get("OPENAI_API_KEY") and not os.environ.get("OAI_KEY"):
        os.environ["OAI_KEY"] = os.environ["OPENAI_API_KEY"]


# ── Column-specific bench + patient configs ──────────────────────────────

def _bench_and_patient(args, bench_cfg: dict, meddx_root: Path) -> tuple[dict, str]:
    """Fill bench_cfg for the column; return (patient_agent_cfg, kb_dir for closed-world RAG)."""
    bench_cfg.setdefault("config", {})
    if args.column == "patientsim":
        profiles = str(Path(args.cases).expanduser().resolve())
        bench_cfg["class_name"] = "baselines.meddxagent.benchmarks.MoBayesDDxPlus50"
        bench_cfg["config"]["profiles_path"] = profiles
        bench_cfg["config"].setdefault("knn_search_cfg", {})["embedding_model"] = args.fewshot_embedding_model
        patient_cfg = {
            "class_name": "baselines.meddxagent.patients.MoBayesPatientAgent",
            "config": {
                "profiles_path": profiles,
                "repo_root": str(REPO_ROOT),
                "patient_model": args.patient_model,
                "base_url": args.patient_base_url,
            },
        }
        return patient_cfg, str(REPO_ROOT / "data/kbs/ddxplus")
    if args.column == "htsim":
        bench_cfg["class_name"] = "baselines.meddxagent.benchmarks.DDxPlusHTSim50"
        bench_cfg["config"].setdefault("knn_search_cfg", {})["embedding_model"] = args.fewshot_embedding_model
        # Patient: MEDDxAgent's native LLMPatient (paper-faithful HT simulator).
        patient_cfg = load_yaml(meddx_root / "configs/patient_agents/llm_patient.yml")
        maybe_override_model(patient_cfg, args.patient_model)
        return patient_cfg, str(REPO_ROOT / "data/kbs/ddxplus")
    # agentclinic
    bench_cfg["class_name"] = "baselines.meddxagent.benchmarks.AgentClinic50"
    bench_cfg["config"]["cases_path"] = args.cases
    bench_cfg["config"]["diseases_csv"] = args.diseases
    bench_cfg["config"]["source_pool_path"] = args.source_pool_path
    patient_cfg = {
        "class_name": "baselines.meddxagent.patients.ACMEDDxAgentPatient",
        "config": {
            "selected_cases_path": args.cases,
            "repo_root": str(REPO_ROOT),
            "patient_model": args.patient_model,
        },
    }
    return patient_cfg, str(Path(args.diseases).parent)


def main() -> None:
    args = parse_args()
    if args.column == "agentclinic":
        # torch must be imported before faiss (MEDDxAgent's _knn_search.py imports
        # faiss first, which segfaults on macOS with conflicting BLAS bindings).
        import torch  # noqa: F401

    ensure_api_env()
    meddx_root = Path(args.meddxagent_root).expanduser().resolve()
    if not meddx_root.exists():
        raise FileNotFoundError(f"MEDDxAgent repo not found: {meddx_root}")
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(meddx_root))

    from ddxdriver.run_ddxdriver import run_ddxdriver
    if args.column in ("htsim", "agentclinic"):
        # Disable MEDDxAgent's seeded shuffle so the first N cases run in file
        # order, matching every other baseline on these columns.
        import ddxdriver.run_ddxdriver as _runner
        _orig_sample = _runner.sample_dataset

        def _sample_dataset_no_shuffle(bench, num_patients=0, seed=None):
            return _orig_sample(bench=bench, num_patients=num_patients, seed=None)
        _runner.sample_dataset = _sample_dataset_no_shuffle

    bench_cfg = load_yaml(meddx_root / "configs/bench.yml")
    ddx_cfg = load_yaml(meddx_root / "configs/ddx_drivers/fixed_choice.yml")
    diag_cfg = load_yaml(meddx_root / "configs/diagnosis_agents/single_llm_standard.yml")
    hist_cfg = load_yaml(meddx_root / "configs/history_taking_agents/llm_history_taking.yml")
    rag_cfg = load_yaml(meddx_root / "configs/rag_agents/searchrag.yml")

    bench_cfg["num_patients"] = args.num_patients
    patient_cfg, kb_dir = _bench_and_patient(args, bench_cfg, meddx_root)

    for cfg in (ddx_cfg, diag_cfg, hist_cfg, rag_cfg):
        maybe_override_model(cfg, args.doctor_model, args.doctor_base_url)
    ddx_cfg["config"]["iterations"] = args.iterations
    hist_cfg["config"]["max_questions"] = args.questions_per_iteration
    diag_cfg["config"]["fewshot"]["type"] = args.fewshot_type
    diag_cfg["config"]["fewshot"]["num_shots"] = args.num_shots

    if args.rag_source == "off":
        active_rag_cfg = None
        ddx_cfg["config"]["agent_order"] = ["history_taking", "diagnosis"]
    elif args.rag_source == "closedworld":
        # Local KB retrieval — drop-in for PubMed/Wikipedia external RAG.
        active_rag_cfg = {
            "class_name": "baselines.meddxagent.rag.ClosedWorldDDxRAG",
            "config": {
                "top_k_search": rag_cfg["config"].get("top_k_search", 2),
                "max_keyword_searches": rag_cfg["config"].get("max_keyword_searches", 3),
                "data_dir": kb_dir,
                "model": rag_cfg["config"]["model"],
            },
        }
    else:
        active_rag_cfg = rag_cfg
        active_rag_cfg["config"]["corpus_name"] = "PubMed" if args.rag_source == "pubmed" else "Wikipedia"

    experiment_dir = Path(args.experiment_dir).expanduser().resolve() / time.strftime("%Y%m%d_%H%M%S")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    from baselines.meddxagent import usage_tracker
    usage_tracker.reset()

    run_ddxdriver(
        bench_cfg=bench_cfg,
        ddxdriver_cfg=ddx_cfg,
        diagnosis_agent_cfg=diag_cfg,
        history_taking_agent_cfg=hist_cfg,
        patient_agent_cfg=patient_cfg,
        rag_agent_cfg=active_rag_cfg,
        experiment_folder=experiment_dir,
    )

    usage = usage_tracker.snapshot()
    record = {
        "doctor_model": args.doctor_model,
        "patient_model": args.patient_model,
        "rag_source": args.rag_source,
        "fewshot_type": args.fewshot_type,
        "num_shots": args.num_shots,
    }
    if args.column == "agentclinic":
        record["source_pool_path"] = args.source_pool_path
    record.update({
        "iterations": args.iterations,
        "questions_per_iteration": args.questions_per_iteration,
        "num_patients": args.num_patients,
        "usage": usage,
    })
    with (experiment_dir / "token_usage.json").open("w") as f:
        json.dump(record, f, indent=2)
    print(f"\n[done] Raw artifacts: {experiment_dir}")
    print(f"[tokens] doctor={usage['doctor']}, patient={usage['patient']}")


if __name__ == "__main__":
    main()
