"""DDxPlus HT-sim runner for MEDDxAgent (paper-best PubMed+dynamic-fewshot or closed-world ablation)."""

from __future__ import annotations

import argparse
import json
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
    p = argparse.ArgumentParser()
    p.add_argument("--meddxagent-root", default=str(DEFAULT_MEDDX_ROOT))
    p.add_argument("--num-patients", type=int, default=50)
    p.add_argument("--doctor-model", default="gpt-5.4-nano")
    p.add_argument("--patient-model", default="gpt-5.4-nano")
    p.add_argument("--rag-source", choices=("pubmed", "wikipedia", "closedworld", "off"),
                   default="pubmed")
    p.add_argument("--fewshot-type", choices=("none", "static", "dynamic"),
                   default="dynamic")
    p.add_argument("--num-shots", type=int, default=5)
    p.add_argument("--fewshot-embedding-model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--questions-per-iteration", type=int, default=5)
    p.add_argument("--experiment-dir",
                   default=str(REPO_ROOT / "data/benchmark_results/ddxplus_htsim_meddxagent_raw"))
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

    env_path = REPO_ROOT / ".env"
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

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(meddx_root))

    from ddxdriver.run_ddxdriver import run_ddxdriver
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
    bench_cfg["class_name"] = "baselines.streambench.htsim_meddxagent_bench.DDxPlusHTSim50"
    bench_cfg.setdefault("config", {})
    knn_cfg = bench_cfg["config"].setdefault("knn_search_cfg", {})
    knn_cfg["embedding_model"] = args.fewshot_embedding_model

    maybe_override_model(ddx_cfg, args.doctor_model)
    maybe_override_model(diag_cfg, args.doctor_model)
    maybe_override_model(hist_cfg, args.doctor_model)
    maybe_override_model(rag_cfg, args.doctor_model)
    ddx_cfg["config"]["iterations"] = args.iterations
    hist_cfg["config"]["max_questions"] = args.questions_per_iteration
    diag_cfg["config"]["fewshot"]["type"] = args.fewshot_type
    diag_cfg["config"]["fewshot"]["num_shots"] = args.num_shots

    # Patient: MEDDxAgent's native LLMPatient (paper-faithful HT simulator).
    patient_agent_cfg = load_yaml(meddx_root / "configs/patient_agents/llm_patient.yml")
    maybe_override_model(patient_agent_cfg, args.patient_model)

    if args.rag_source == "off":
        active_rag_cfg = None
        ddx_cfg["config"]["agent_order"] = ["history_taking", "diagnosis"]
    elif args.rag_source == "closedworld":
        # Local DDxPlus KB retrieval — drop-in for PubMed/Wikipedia external RAG.
        active_rag_cfg = {
            "class_name": "baselines.meddxagent.rag.ClosedWorldDDxRAG",
            "config": {
                "top_k_search": rag_cfg["config"].get("top_k_search", 2),
                "max_keyword_searches": rag_cfg["config"].get("max_keyword_searches", 3),
                "data_dir": str(REPO_ROOT / "data/kbs/ddxplus"),
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
        patient_agent_cfg=patient_agent_cfg,
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
            "iterations": args.iterations,
            "questions_per_iteration": args.questions_per_iteration,
            "num_patients": args.num_patients,
            "usage": usage,
        }, f, indent=2)
    print(f"\n[done] Raw artifacts: {experiment_dir}")
    print(f"[tokens] doctor={usage['doctor']}, patient={usage['patient']}")


if __name__ == "__main__":
    main()
