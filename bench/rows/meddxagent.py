"""MEDDxAgent row — Rose et al. ACL 2025 multi-agent driver.

3 LLM agents (history-taking + diagnosis + ddx-driver), 3 iterations
× 5 questions, optional PubMed RAG + dynamic 5-shot fewshot.

Two paper variants (selectable via row_kwargs):
  - "paper-best"      → rag_source=pubmed, fewshot_type=dynamic, num_shots=5
  - "closed-world"    → rag_source=off,    fewshot_type=none,    num_shots=0

Runs MEDDxAgent's published `ddxdriver.run_ddxdriver` in its own venv
(Python 3.12), then converts the final `results.json` into our per-case
JSON schema. Currently supports column='ac'; HT-sim / PatientSim columns
need their own ddxdriver benchmark class.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


NAME = "meddxagent"

_REPO = Path(__file__).resolve().parents[2]
_MEDDX = Path(os.environ.get("MEDDXAGENT_REPO",
                              Path.home() / "Documents/GitHub/meddxagent"))

# Per-column dispatch — different ddxdriver bench class needs a different
# wrapper script. Args are compatible across the canonical and v1 variants
# of the same column.
_RUNNER_BY_COLUMN = {
    "ac":             _REPO / "scripts/baselines/run_meddxagent_agentclinic.py",
    "ac_set2":        _REPO / "scripts/baselines/run_meddxagent_agentclinic.py",
    "htsim":          _REPO / "baselines/streambench/run_meddxagent_ddxplus_htsim.py",
    "htsim_v1":       _REPO / "baselines/streambench/run_meddxagent_ddxplus_htsim.py",
    "patientsim":     _REPO / "scripts/baselines/run_meddxagent_patientsim.py",
    "patientsim_v1":  _REPO / "scripts/baselines/run_meddxagent_patientsim.py",
}

# Column-specific extra args for each runner.
def _column_args(column_name: str) -> List[str]:
    if column_name == "ac":
        return ["--cases",    str(_REPO / "data/cases/agentclinic_a/selected_cases.jsonl"),
                "--diseases", str(_REPO / "data/kbs/agentclinic_a/diseases.csv")]
    if column_name == "ac_set2":
        return ["--cases",    str(_REPO / "data/cases/agentclinic_b/selected_cases.jsonl"),
                "--diseases", str(_REPO / "data/kbs/agentclinic_b/diseases.csv")]
    if column_name in ("htsim", "htsim_v1"):
        # HT-sim runner reads cases via DDXPLUS_HTSIM_CASES_PATH env (set in run_batch).
        return []
    if column_name == "patientsim":
        return ["--profiles", str(_REPO / "data/cases/ddxplus_patientsim_a/plain/profiles.json")]
    if column_name == "patientsim_v1":
        return ["--profiles", str(_REPO / "data/cases/ddxplus_patientsim_b/plain/profiles.json")]
    raise ValueError(column_name)


# Per-column env vars (e.g., HT-sim cases path override).
def _column_env(column_name: str) -> Dict[str, str]:
    if column_name == "htsim":
        return {"DDXPLUS_HTSIM_CASES_PATH": str(_REPO / "data/cases/ddxplus_htsim_a/selected_cases.jsonl")}
    if column_name == "htsim_v1":
        return {"DDXPLUS_HTSIM_CASES_PATH": str(_REPO / "data/cases/ddxplus_htsim_b/selected_cases.jsonl")}
    return {}


def make_doctor(*args, **kwargs):
    raise NotImplementedError("meddxagent is batch-mode; uses run_batch.")


def _variant_flags(variant: str) -> List[str]:
    if variant == "paper-best":
        return ["--rag-source", "pubmed", "--fewshot-type", "dynamic",
                "--num-shots", "5"]
    if variant == "closed-world":
        return ["--rag-source", "off", "--fewshot-type", "none",
                "--num-shots", "0"]
    raise ValueError(f"Unknown meddxagent variant: {variant!r}")


def run_batch(*, cases: List[Dict[str, Any]], start_idx: int, column,
              doctor_model: str, patient_model: str, max_turns: int,
              api_key: str, out_dir: Path, row_kwargs: Dict[str, Any]) -> None:
    if start_idx != 0:
        raise NotImplementedError(
            "meddxagent --start-idx > 0 not supported (ddxdriver runs first N).")
    runner = _RUNNER_BY_COLUMN.get(column.NAME)
    if runner is None or not runner.exists():
        sys.exit(f"ERROR: meddxagent runner for column={column.NAME!r} missing: {runner}")
    if not _MEDDX.exists():
        sys.exit(f"ERROR: MEDDXAGENT_REPO={_MEDDX} not found.")
    venv_py = _MEDDX / ".venv/bin/python"
    if not venv_py.exists():
        sys.exit(f"ERROR: meddxagent venv missing: {venv_py}")

    variant = row_kwargs.get("variant", "paper-best")
    raw_dir = out_dir.parent / f"{out_dir.name}_raw"

    cmd = [
        str(venv_py),
        str(runner),
        "--meddxagent-root", str(_MEDDX),
        *_column_args(column.NAME),
        "--num-patients", str(len(cases)),
        "--doctor-model", doctor_model,
        "--patient-model", patient_model,
        "--experiment-dir", str(raw_dir),
        *_variant_flags(variant),
    ]
    env = os.environ.copy()
    env["AGENTCLINIC_REPO"] = env.get("AGENTCLINIC_REPO", "/tmp/agentclinic-tmp")
    env["PYTHONPATH"] = str(_REPO)
    env.update(_column_env(column.NAME))

    print(f"[meddxagent] variant={variant}, n={len(cases)} ...", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=str(_REPO))
    print(f"[meddxagent] exit={proc.returncode}, elapsed {time.time()-t0:.0f}s",
          flush=True)
    if proc.returncode != 0:
        sys.exit("meddxagent driver failed; check logs above.")

    # Find newest results.json under raw_dir/<timestamp>/
    runs = sorted(raw_dir.glob("*/results.json"))
    if not runs:
        sys.exit(f"ERROR: no results.json under {raw_dir}")
    results_path = runs[-1]
    data = json.loads(results_path.read_text())
    n_hits = 0
    for r in data.get("Results", []):
        idx = r.get("patient", {}).get("patient_id", "")
        # patient_id is the case index in our input (0-based)
        try:
            case_idx = int(idx) if not isinstance(idx, str) else int(
                idx.split("_")[-1]
            )
        except Exception:
            continue
        gt = r["patient"]["gt_pathology"]
        pred_set = list(r.get("final_ddx_prediction") or [])
        top1 = pred_set[0] if pred_set else "Unknown"
        hit = column.matches_gt(top1, gt)
        in_top3 = any(column.matches_gt(p, gt) for p in pred_set[:3])
        n_hits += int(hit)
        out_path = out_dir / f"case_{case_idx:03d}.json"
        with out_path.open("w") as f:
            json.dump({
                "case_idx": case_idx,
                "baseline": NAME,
                "variant": variant,
                "column": column.NAME,
                "ground_truth": gt,
                "final": {
                    "predicted": top1,
                    "predicted_set": pred_set[:5],
                    "top1_hit": bool(hit),
                    "top3_hit": bool(in_top3),
                    "abstained": False,
                },
                "raw_results_path": str(results_path),
            }, f, indent=2, ensure_ascii=False)
    print(f"\n[meddxagent] {n_hits}/{len(data.get('Results', []))} top-1. "
          f"Per-case: {out_dir} | raw: {results_path}", flush=True)
