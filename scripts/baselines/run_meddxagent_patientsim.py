#!/usr/bin/env python3
"""Run MEDDxAgent as doctor with MoBayes's benchmark PatientSim.

This keeps MEDDxAgent's doctor/orchestrator stack intact while replacing only
the patient module so comparisons use the same patient behavior as our other
benchmarks.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
_REPO_PARENT = REPO_ROOT.parent
DEFAULT_MEDDX_ROOT = _REPO_PARENT.parent / "meddxagent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meddxagent-root",
        default=str(DEFAULT_MEDDX_ROOT),
        help="Path to the local meddxagent checkout.",
    )
    parser.add_argument(
        "--profiles",
        default=str(REPO_ROOT / "data/cases/ddxplus_patientsim_a/plain/profiles.json"),
        help="Path to MoBayes benchmark profiles.json.",
    )
    parser.add_argument(
        "--num-patients",
        type=int,
        default=50,
        help="How many patients to evaluate.",
    )
    parser.add_argument(
        "--doctor-model",
        default="gpt-5.4-nano",
        help="Model used by MEDDxAgent doctor-side modules.",
    )
    parser.add_argument(
        "--patient-model",
        default="gpt-5.4-nano",
        help="Model used by the benchmark PatientSim.",
    )
    parser.add_argument(
        "--doctor-base-url",
        default=None,
        help="Optional base URL for MEDDxAgent doctor-side OpenAI-compatible calls.",
    )
    parser.add_argument(
        "--patient-base-url",
        default=None,
        help="Optional base URL for PatientSim OpenAI-compatible calls.",
    )
    parser.add_argument(
        "--experiment-dir",
        default=str(REPO_ROOT / "data/benchmark_results/meddxagent_patientsim"),
        help="Output directory for raw MEDDxAgent experiment artifacts.",
    )
    parser.add_argument(
        "--rag-source",
        choices=("pubmed", "wikipedia", "closedworld", "off"),
        default="pubmed",
        help="Knowledge source for the MEDDxAgent RAG module.",
    )
    parser.add_argument(
        "--fewshot-type",
        choices=("none", "static", "dynamic"),
        default="none",
        help="Few-shot mode for MEDDxAgent's diagnosis module.",
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=5,
        help="Number of diagnosis examples when few-shot is enabled.",
    )
    parser.add_argument(
        "--fewshot-embedding-model",
        default="BAAI/bge-base-en-v1.5",
        help="Embedding model for dynamic few-shot retrieval.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of fixed MEDDxAgent iterations.",
    )
    parser.add_argument(
        "--questions-per-iteration",
        type=int,
        default=5,
        help="History-taking questions per iteration.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def ensure_api_env(repo_root: Path) -> None:
    env_path = repo_root / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path)
        except Exception:
            pass

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key and not os.environ.get("OAI_KEY"):
        os.environ["OAI_KEY"] = openai_key


def maybe_override_model(cfg: dict, model_name: str, base_url: str | None) -> None:
    model_cfg = cfg.get("config", {}).get("model", {}).get("config", {})
    if not model_cfg:
        return
    model_cfg["model_name"] = model_name
    if base_url:
        model_cfg["base_url"] = base_url


def main() -> None:
    args = parse_args()
    ensure_api_env(REPO_ROOT)

    meddx_root = Path(args.meddxagent_root).expanduser().resolve()
    if not meddx_root.exists():
        raise FileNotFoundError(f"MEDDxAgent repo not found: {meddx_root}")

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(meddx_root))

    from ddxdriver.run_ddxdriver import run_ddxdriver

    bench_cfg = load_yaml(meddx_root / "configs/bench.yml")
    ddxdriver_cfg = load_yaml(meddx_root / "configs/ddx_drivers/fixed_choice.yml")
    diagnosis_agent_cfg = load_yaml(
        meddx_root / "configs/diagnosis_agents/single_llm_standard.yml"
    )
    history_taking_agent_cfg = load_yaml(
        meddx_root / "configs/history_taking_agents/llm_history_taking.yml"
    )
    rag_agent_cfg = load_yaml(meddx_root / "configs/rag_agents/searchrag.yml")

    bench_cfg["num_patients"] = args.num_patients
    bench_cfg["class_name"] = "baselines.meddxagent.benchmark.MoBayesDDxPlus50"
    bench_cfg.setdefault("config", {})
    bench_cfg["config"]["profiles_path"] = str(Path(args.profiles).expanduser().resolve())
    knn_cfg = bench_cfg["config"].setdefault("knn_search_cfg", {})
    knn_cfg["embedding_model"] = args.fewshot_embedding_model

    maybe_override_model(ddxdriver_cfg, args.doctor_model, args.doctor_base_url)
    maybe_override_model(diagnosis_agent_cfg, args.doctor_model, args.doctor_base_url)
    maybe_override_model(
        history_taking_agent_cfg, args.doctor_model, args.doctor_base_url
    )
    maybe_override_model(rag_agent_cfg, args.doctor_model, args.doctor_base_url)
    ddxdriver_cfg["config"]["iterations"] = args.iterations
    history_taking_agent_cfg["config"]["max_questions"] = args.questions_per_iteration
    diagnosis_agent_cfg["config"]["fewshot"]["type"] = args.fewshot_type
    diagnosis_agent_cfg["config"]["fewshot"]["num_shots"] = args.num_shots

    patient_agent_cfg = {
        "class_name": "baselines.meddxagent.patient.MoBayesPatientAgent",
        "config": {
            "profiles_path": str(Path(args.profiles).expanduser().resolve()),
            "repo_root": str(REPO_ROOT),
            "patient_model": args.patient_model,
            "base_url": args.patient_base_url,
        },
    }

    active_rag_agent_cfg = rag_agent_cfg
    if args.rag_source == "pubmed":
        active_rag_agent_cfg["config"]["corpus_name"] = "PubMed"
    elif args.rag_source == "wikipedia":
        active_rag_agent_cfg["config"]["corpus_name"] = "Wikipedia"
    elif args.rag_source == "closedworld":
        active_rag_agent_cfg = {
            "class_name": "baselines.meddxagent.rag.ClosedWorldDDxRAG",
            "config": {
                "top_k_search": rag_agent_cfg["config"].get("top_k_search", 2),
                "max_keyword_searches": rag_agent_cfg["config"].get("max_keyword_searches", 3),
                "data_dir": str(REPO_ROOT / "data/kbs/ddxplus"),
                "model": rag_agent_cfg["config"]["model"],
            },
        }
    elif args.rag_source == "off":
        active_rag_agent_cfg = None
        ddxdriver_cfg["config"]["agent_order"] = ["history_taking", "diagnosis"]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(args.experiment_dir).expanduser().resolve() / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Reset token accumulators before this run
    from baselines.meddxagent import usage_tracker
    usage_tracker.reset()

    run_ddxdriver(
        bench_cfg=bench_cfg,
        ddxdriver_cfg=ddxdriver_cfg,
        diagnosis_agent_cfg=diagnosis_agent_cfg,
        history_taking_agent_cfg=history_taking_agent_cfg,
        patient_agent_cfg=patient_agent_cfg,
        rag_agent_cfg=active_rag_agent_cfg,
        experiment_folder=experiment_dir,
    )

    # Snapshot usage and write to disk alongside results.json
    usage = usage_tracker.snapshot()
    usage_path = experiment_dir / "token_usage.json"
    import json as _json
    with usage_path.open("w") as f:
        _json.dump({
            "doctor_model": args.doctor_model,
            "patient_model": args.patient_model,
            "rag_source": args.rag_source,
            "fewshot_type": args.fewshot_type,
            "num_shots": args.num_shots,
            "iterations": args.iterations,
            "questions_per_iteration": args.questions_per_iteration,
            "num_patients": args.num_patients,
            "usage": usage,
        }, f, indent=2)

    print(f"MEDDxAgent run complete. Raw artifacts: {experiment_dir}")
    print(f"Token usage: doctor={usage['doctor']}, patient={usage['patient']}")


if __name__ == "__main__":
    main()
