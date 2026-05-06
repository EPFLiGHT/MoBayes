"""Run MEDDxAgent on AgentClinic-50 cases (AC patient sim).

Forks scripts/baselines/run_meddxagent_patientsim.py:
    - Benchmark class: AgentClinic50 (loads our AC selected_cases.jsonl)
    - Patient agent  : ACMEDDxAgentPatient (drives AC PatientAgent)
    - Fewshot        : "none" (DDxPlus validate-split pool is irrelevant for AC)
    - RAG            : disabled by default (--rag-source off) — turn it on
                       only if you want closed-world / pubmed retrieval

Run:
    OPENAI_API_KEY=... PYTHONPATH=. python3 scripts/baselines/run_meddxagent_agentclinic.py \\
        --num-patients 10 --rag-source off
"""

from __future__ import annotations

# IMPORTANT: torch MUST be imported before faiss (MEDDxAgent's _knn_search.py
# imports faiss before torch — that order segfaults on macOS where torch and
# faiss carry conflicting BLAS bindings). Pre-import torch here so the symbol
# table is set up correctly before MEDDxAgent's own modules load.
import torch  # noqa: F401

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml


THIS_FILE = Path(__file__).resolve()
MODN_ROOT = THIS_FILE.parents[2]
REPO_ROOT = MODN_ROOT.parent
DEFAULT_MEDDX_ROOT = REPO_ROOT.parent / "meddxagent"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--meddxagent-root", default=str(DEFAULT_MEDDX_ROOT))
    p.add_argument("--cases-path",
                   default=str(MODN_ROOT / "data/cases/agentclinic_a/selected_cases.jsonl"))
    p.add_argument("--diseases-csv",
                   default=str(MODN_ROOT / "data/kbs/agentclinic_a/diseases.csv"))
    p.add_argument("--num-patients", type=int, default=10)
    p.add_argument("--doctor-model", default="gpt-5.4-nano")
    p.add_argument("--patient-model", default="gpt-5.4-nano")
    p.add_argument("--rag-source", choices=("off", "closedworld", "pubmed", "wikipedia"),
                   default="off")
    p.add_argument("--fewshot-type", choices=("none", "static", "dynamic"),
                   default="none")
    p.add_argument("--num-shots", type=int, default=0)
    p.add_argument("--source-pool-path", default="/tmp/agentclinic-tmp/agentclinic_medqa_extended.jsonl",
                   help="AC source pool for dynamic fewshot (213 cases)")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--questions-per-iteration", type=int, default=5)
    p.add_argument("--experiment-dir",
                   default=str(MODN_ROOT / "data/benchmark_results/agentclinic_10_meddxagent_raw"))
    return p.parse_args()


def load_yaml(p: Path) -> dict:
    with p.open() as f:
        return yaml.safe_load(f)


def maybe_override_model(cfg: dict, model_name: str) -> None:
    model_cfg = cfg.get("config", {}).get("model", {}).get("config", {})
    if model_cfg:
        model_cfg["model_name"] = model_name


def main() -> None:
    args = parse_args()

    # Load .env / mirror OPENAI_API_KEY -> OAI_KEY
    env_path = MODN_ROOT / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except Exception:
            pass
    if os.environ.get("OPENAI_API_KEY") and not os.environ.get("OAI_KEY"):
        os.environ["OAI_KEY"] = os.environ["OPENAI_API_KEY"]

    meddx_root = Path(args.meddxagent_root).expanduser().resolve()
    if not meddx_root.exists():
        raise FileNotFoundError(f"MEDDxAgent repo not found: {meddx_root}")

    sys.path.insert(0, str(MODN_ROOT))
    sys.path.insert(0, str(meddx_root))

    from ddxdriver.run_ddxdriver import run_ddxdriver
    # Disable MEDDxAgent's seeded shuffle so we run on the first N cases in
    # AC-50 order (ac_000..ac_009), matching every other AC baseline.
    import ddxdriver.run_ddxdriver as _runner
    _orig_sample = _runner.sample_dataset
    def _sample_dataset_no_shuffle(bench, num_patients=0, seed=None):
        return _orig_sample(bench=bench, num_patients=num_patients, seed=None)
    _runner.sample_dataset = _sample_dataset_no_shuffle

    # Load default configs from MEDDxAgent
    bench_cfg = load_yaml(meddx_root / "configs/bench.yml")
    ddx_cfg = load_yaml(meddx_root / "configs/ddx_drivers/fixed_choice.yml")
    diag_cfg = load_yaml(meddx_root / "configs/diagnosis_agents/single_llm_standard.yml")
    hist_cfg = load_yaml(meddx_root / "configs/history_taking_agents/llm_history_taking.yml")
    rag_cfg = load_yaml(meddx_root / "configs/rag_agents/searchrag.yml")

    # Bench: point at our AC class
    bench_cfg["num_patients"] = args.num_patients
    bench_cfg["class_name"] = "baselines.meddxagent.benchmark_agentclinic.AgentClinic50"
    bench_cfg.setdefault("config", {})
    bench_cfg["config"]["cases_path"] = args.cases_path
    bench_cfg["config"]["diseases_csv"] = args.diseases_csv
    bench_cfg["config"]["source_pool_path"] = args.source_pool_path

    # Models
    maybe_override_model(ddx_cfg, args.doctor_model)
    maybe_override_model(diag_cfg, args.doctor_model)
    maybe_override_model(hist_cfg, args.doctor_model)
    maybe_override_model(rag_cfg, args.doctor_model)
    ddx_cfg["config"]["iterations"] = args.iterations
    hist_cfg["config"]["max_questions"] = args.questions_per_iteration
    diag_cfg["config"]["fewshot"]["type"] = args.fewshot_type
    diag_cfg["config"]["fewshot"]["num_shots"] = args.num_shots

    # Patient agent: AC adapter
    patient_cfg = {
        "class_name": "baselines.meddxagent.patient_agentclinic.ACMEDDxAgentPatient",
        "config": {
            "selected_cases_path": args.cases_path,
            "modn_root": str(MODN_ROOT),
            "patient_model": args.patient_model,
        },
    }

    # RAG
    if args.rag_source == "off":
        active_rag_cfg = None
        ddx_cfg["config"]["agent_order"] = ["history_taking", "diagnosis"]
    elif args.rag_source == "closedworld":
        active_rag_cfg = {
            "class_name": "baselines.meddxagent.rag.ClosedWorldDDxRAG",
            "config": {
                "top_k_search": rag_cfg["config"].get("top_k_search", 2),
                "max_keyword_searches": rag_cfg["config"].get("max_keyword_searches", 3),
                "data_dir": str(MODN_ROOT / "data/cases/agentclinic_a"),
                "model": rag_cfg["config"]["model"],
            },
        }
    else:
        active_rag_cfg = rag_cfg
        active_rag_cfg["config"]["corpus_name"] = "PubMed" if args.rag_source == "pubmed" else "Wikipedia"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(args.experiment_dir).expanduser().resolve() / timestamp
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
    with (experiment_dir / "token_usage.json").open("w") as f:
        json.dump({
            "doctor_model": args.doctor_model,
            "patient_model": args.patient_model,
            "rag_source": args.rag_source,
            "fewshot_type": args.fewshot_type,
            "num_shots": args.num_shots,
            "source_pool_path": args.source_pool_path,
            "iterations": args.iterations,
            "questions_per_iteration": args.questions_per_iteration,
            "num_patients": args.num_patients,
            "usage": usage,
        }, f, indent=2)
    print(f"\n[done] Raw artifacts: {experiment_dir}")
    print(f"[tokens] doctor={usage['doctor']}, patient={usage['patient']}")


if __name__ == "__main__":
    main()
