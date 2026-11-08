# MoBayes — Modular Bayesian Framework for Conversational Clinical Decision Support

Companion code for the paper *"MoBayes: A Modular Bayesian
Framework for Separating Reasoning from Language in Conversational Clinical
Decision Support"* (NeurIPS 2026;
[arXiv:2604.20022](https://arxiv.org/abs/2604.20022)).

---

## Overview

MoBayes separates language understanding from probabilistic inference:

```
Patient utterance
   │
   ▼
LLM parser  ──►  (feature, value, confidence)   (language only)
                       │
                       ▼
        ┌───────────────────────────────┐
        │  Bayesian engine               │
        │  • posterior over diseases     │
        │  • expected-information-gain   │
        │  • commit / defer threshold τ  │
        └───────────────────────────────┘
                       │
                       ▼
              next question  /  diagnosis
                       │
                       ▼
                  LLM verbaliser            (language only)
```

The Bayesian module performs all probabilistic reasoning deterministically;
the LLM only parses patient responses and verbalises engine-selected
questions.

---

## Setup

```bash
# 1. Python environment (3.10+ recommended)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. PYTHONPATH so engine imports work from the repo root
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 3. API keys (only needed to run new LLM sessions, NOT to regenerate tables/figures)
export OPENAI_API_KEY="..."
export OPEN_ROUTER="..."    # optional; used for non-OpenAI sensor / judge models via OpenRouter
```

The `experiments/*.sh` orchestrators source a local `.env` if one is
present; you can either put the same exports there or just leave them in
your shell. A minimal `.env` looks like:

```bash
OPENAI_API_KEY=sk-...
OPEN_ROUTER=sk-or-...
```

### Optional baseline dependencies

The saved results under `data/benchmark_results/{set_a,set_b,judged}/` are
sufficient to regenerate every table and figure. The clones below are only
needed when *re-running* the corresponding baselines.

| Baseline | Clone | Env var | Notes |
|---|---|---|---|
| MediQ-Expert | `stellalisy/mediQ` | `MEDIQ_REPO` | see `baselines/mediq/README.md` for the patch set |
| MEDDxAgent | `nec-research/meddxagent` | `MEDDXAGENT_REPO` | 2-line token-tracking patch, see `baselines/meddxagent/README.md` |
| AgentClinic | `SamuelSchmidgall/AgentClinic` | `AGENTCLINIC_REPO` | — |
| UoT | `zhiyuanhubj/UoT` @ `e08f17f` | — | `bash baselines/uot/fetch_uot.sh` clones it into `baselines/uot/repo/` |

---

## Running experiments

Every (system × benchmark) cell of the baseline comparison goes through the
unified driver (the ablation and LLM-KB scripts under `experiments/` call
`scripts/core/benchmark_patientsim.py` directly):

```bash
PYTHONPATH=. python3 -m bench --row <row> --column <column> --num-cases 50
```

Rows: `bmbe`, `sa`, `cw`, `cot`, `savage_dd`, `savage_bay`, `ac_native`,
`mediq`, `meddxagent`, `cod`, `uot`.
Columns: `patientsim`, `htsim`, `ac`, plus the disjoint second halves
(`patientsim_v1`, `htsim_v1`, `ac_set2`) used for the combined n=100 evaluation.

Output goes to
`data/benchmark_results/<column>_50_<row>[_<suffix>]/case_NNN.json`
(one JSON per patient). See `bench/README.md` for the full Row × Column
command table and the per-case JSON schema.

> Note on layouts. The bench driver writes fresh runs to
> `data/benchmark_results/<column>_50_<row>...` (per-case JSON). The
> pre-committed paper results are stored in the older layout
> `data/benchmark_results/{set_a,set_b}/<column>/<row>/*.jsonl`, which is
> what `scripts/paper_tables/*.py` reads. To regenerate paper tables from
> new bench runs you would need to re-aggregate into the `set_{a,b}`
> layout; the saved results already do this so no extra step is needed
> for reproduction.

### Interactive session (play as the patient)

Drive a live diagnostic dialogue from your terminal: MoBayes asks
questions, you type the answers.

```bash
PYTHONPATH=. python3 scripts/core/run_session.py \
  --data-dir data/kbs/ddxplus/ --model gpt-5.4-nano
```

Useful flags: `--max-questions N`, `--confidence-threshold τ`, `--no-llm`
(raw schema questions with manual input — no LLM call).

### Detailed trace on a simulated patient

For debugging / inspection: runs MoBayes against a PatientSim profile and
prints every step (bulk extraction → EIG selection → posterior update,
per turn).

```bash
PYTHONPATH=. python3 scripts/utils/trace_session.py \
  --profiles data/cases/ddxplus_patientsim_a/plain/profiles.json \
  --data-dir data/kbs/ddxplus/ --patient-index 0
```

### Pre-defined paper experiments

```bash
# Main table (Table 2): SA doctors and MoBayes sensors on HT-sim and AgentClinic
bash experiments/ddxplus/main_table.sh

# Ablations (Appendix)
bash experiments/ddxplus/persona_ablation.sh    # 4 adversarial personas
bash experiments/ddxplus/prior_ablation.sh      # empirical / uniform / conditional
bash experiments/ddxplus/eig_ablation.sh        # global vs focused top-k EIG
bash experiments/ddxplus/disease_scaling.sh     # K=10/20/30/40/49 disease subsets

# LLM-elicited KB ecosystems (Section 4.3)
bash experiments/llmkb/gpt_ecosystem.sh         # GPT-5.4-elicited KB
bash experiments/llmkb/gemini_ecosystem.sh      # Gemini 2.5 Flash-elicited KB
bash experiments/llmkb/cross_kb_transfer.sh     # each ecosystem tested on the other's patients
```

### LLM-judge re-scoring (post-processing)

Free-text doctor outputs are mapped to the closed disease list by a one-shot
LLM judge pass; tables read the resulting `top1_hit_judged`, `top3_hit_judged`,
`abstained_judged` fields. Judged outputs under
`data/benchmark_results/judged/` are pre-committed; re-run only if you add
new cells. The remap script takes a single `--in-dir` of `case_*.json`
files at a time:

```bash
# Example: re-judge one new cell.
PYTHONPATH=. python3 scripts/analysis/llm_remap_unified.py \
  --in-dir data/benchmark_results/<column>_50_<row>[_<suffix>]
# Then recompute aggregate metrics.
PYTHONPATH=. python3 scripts/analysis/recompute_metrics_from_judged.py
```

---

## Knowledge base construction

All KBs are pre-built under `data/kbs/`. The scripts below are kept for
transparency and re-elicitation; rebuilding is only needed to construct a
new KB from scratch.

### DDxPlus (empirical, no LLM)

```bash
PYTHONPATH=. python3 scripts/data_prep/ddxplus/convert_ddxplus.py \
  --output-dir data/kbs/ddxplus/
PYTHONPATH=. python3 scripts/data_prep/ddxplus/compute_conditional_priors.py
PYTHONPATH=. python3 scripts/data_prep/validate_kb.py data/kbs/ddxplus/
```

### LLM-elicited KBs (Section 4.3)

The Gemini- and GPT-elicited KBs reported in Section 4.3 are built by the
pipeline under `scripts/data_prep/llmkb/`. The pipeline reads paths from
`scripts/data_prep/llmkb/config.yml`; the input disease list lives at
`scripts/data_prep/llmkb/config.yml::paths.diseases_json` (default points
under `data/raw/llmkb/input/`). Place your `diseases<N>.json` there before
running:

```bash
PYTHONPATH=. python3 scripts/data_prep/llmkb/generate_features.py
PYTHONPATH=. python3 scripts/data_prep/llmkb/generate_distributions.py
PYTHONPATH=. python3 scripts/data_prep/llmkb/generate_priors.py \
  --diseases data/raw/llmkb/input/diseases18.json \
  --output data/raw/llmkb/output/priors_diseases18.csv
PYTHONPATH=. python3 scripts/data_prep/llmkb/convert_llmkb.py \
  --output data/kbs/llmkb_gpt        # or llmkb_gemini
```

`config.yml` selects the elicitation backbone (e.g. GPT-5.4 or Gemini
3.1 Pro) and the disease list. The resulting KBs (under
`data/kbs/llmkb_{gpt,gemini}/`) are pre-shipped and the
`experiments/llmkb/` scripts read directly from them; the raw inputs under
`data/raw/` are not included in this release.

The disease and symptom lists can also be extracted from a case set
(`scripts/data_prep/agentclinic/extract_features.py`) instead of written by
hand; the `LLMKB_*_JSON_PATH` variables point the scripts at those files.

### Patient profiles (PatientSim)

```bash
PYTHONPATH=. python3 scripts/data_prep/generate_profiles.py \
  --patients data/kbs/ddxplus/patients.jsonl \
  --n 50 --stratify --seed 42 \
  --output data/cases/ddxplus_patientsim_a
```

Personas (`plain`, `dazed`, `distrustful`, `overanxious`,
`verbose_lowrecall`) are defined in `scripts/data_prep/personas.py`.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `world_model/` | Bayesian inference engine (Dirichlet-Categorical, Jeffrey's conditioning, stopping rules) |
| `decision_policy/` | Expected information gain (EIG) question selection |
| `llm_interface/` | LLM parsing and verbalisation (structured extraction + prompting) |
| `bench/` | Unified benchmark driver: columns (benchmarks + patient simulators), row registry, the MoBayes row |
| `baselines/` | Prior systems, one folder each with its row and code (see `baselines/README.md`) |
| `scripts/core/` | Single-session and PatientSim benchmark entry points (MoBayes doctor, PatientSim patient) |
| `scripts/data_prep/` | Knowledge base and case construction (DDxPlus, AgentClinic, LLM-elicited) |
| `scripts/analysis/` | LLM-judge remapping and metric recomputation |
| `experiments/` | Pre-defined paper experiments (`.sh` orchestrators) and figure generators |
| `additional/` | Appendix C.15–C.20 experiments, self-contained (see `additional/README.md`) |
| `scripts/paper_tables/` | LaTeX table generators (read directly from `data/benchmark_results/`) |
| `data/kbs/`, `data/cases/`, `data/benchmark_results/` | Knowledge bases, patient profiles, saved per-case results |
| `data/raw/` | LLM-KB rebuild inputs (not shipped; created when rebuilding an LLM KB) |
| `tables/`, `figures/` | Auto-generated LaTeX / PDF / PNG outputs (created on first run) |

---

## Data sources

| Dataset | Source | Notes |
|---------|--------|-------|
| DDxPlus | [DDxPlus 2022](https://github.com/mila-iqia/ddxplus) | Original benchmark; we use a 50-patient stratified subset (`data/cases/ddxplus_patientsim_a/`) and a disjoint 50-patient sample (`data/cases/ddxplus_patientsim_b/`) for the combined n=100 evaluation. |
| AgentClinic-MedQA | [AgentClinic](https://github.com/SamuelSchmidgall/AgentClinic) | 213-case USMLE-derived dialogue benchmark; we sample two disjoint 50-case subsets. |
| LLM-elicited KBs | `data/kbs/agentclinic_{a,b}/` | Generated by `scripts/data_prep/{agentclinic,llmkb}/`; can be rebuilt with the commands above. |

---

## Regenerating paper tables and figures

The saved cases under `data/benchmark_results/` are enough to reproduce
every LaTeX table and figure in the paper without re-running any LLM
session.

### LaTeX tables

```bash
python3 scripts/paper_tables/render_paper_tables.py           # Table 1 + Table 21 → tables/
python3 scripts/paper_tables/ci_tables.py --appendix > tables/tab_ci.tex   # Tables 19 + 20 (bootstrap CIs)
python3 scripts/paper_tables/tokentable.py                    # per-cell token breakdown (Tok/case sources)
```

Other utilities in `scripts/paper_tables/` cover paired-bootstrap significance
(`m7_bootstrap_significance.py`), threshold tuning (`find_optimal_tau.py`,
`m4_loocv_tau_tuning.py`), a matcher-accuracy audit (`m11_matcher_audit.py`),
and the token estimates for UoT / CoD whose runners did not log doctor tokens
(`estimate_tokens_uot_cod.py`, `cod_yi34b_token_pass_v3.py`; their outputs are
shipped in `data/benchmark_results/costs/`).

### Figures

```bash
PYTHONPATH=. python3 experiments/analysis/generate_paper_plots.py        # cost-vs-scale + persona radar
PYTHONPATH=. python3 experiments/analysis/plot_acc_coverage.py           # accuracy-coverage frontier
PYTHONPATH=. python3 experiments/analysis/scaling_analysis.py --plot     # disease scaling (Table 18 / Fig. 12)
PYTHONPATH=. python3 experiments/analysis/failure_analysis.py \
  --data-dir data/kbs/ddxplus \
  --profiles data/cases/ddxplus_patientsim_a/plain/profiles.json \
  --results data/benchmark_results/set_a/ddxplus_patientsim/mobayes/bmbe_*.jsonl
```

---

## License

Released under the Apache License 2.0 — see `LICENSE`.
Baselines that need third-party code (UoT, MediQ, MEDDxAgent, AgentClinic,
Chain-of-Diagnosis) fetch it from upstream; it is not redistributed here.

## Citation

```bibtex
@misc{kesmen2026mobayes,
  title         = {MoBayes: A Modular Bayesian Framework for Separating Reasoning from Language in Conversational Clinical Decision Support},
  author        = {Yusuf Kesmen and Fay Elhassan and Jiayi Ma and Julien Stalhandske and Yena Chang and David Sasu and Alexandra Kulinkina and Akhil Arora and Lars Klein and Mary-Anne Hartley},
  year          = {2026},
  eprint        = {2604.20022},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2604.20022}
}
```
