# MoBayes — Modular Bayesian Framework for Conversational Clinical Decision Support

Companion code for the paper *"MoBayes: A Modular Bayesian
Framework for Separating Reasoning from Language in Conversational Clinical
Decision Support."*

> **Anonymous release.** This repository is hosted on
> [anonymous.4open.science](https://anonymous.4open.science/) and contains no
> author names, institutional identifiers, or links to non-anonymous
> repositories. Identifying information has also been stripped from
> intermediate artefacts (logs, JSON case files).

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
| MediQ-Expert | `stellalisy/mediQ` | `MEDIQ_REPO` | see `baselines/mediq.md` for the patch set |
| MEDDxAgent | `AgentAITech/meddxagent` | `MEDDXAGENT_REPO` | 2-line token-tracking patch, see `baselines/meddxagent/README.md` |
| AgentClinic | `SamuelSchmidgall/AgentClinic` | `AGENTCLINIC_REPO` | — |
| UoT | already vendored under `baselines/uot/repo/` | — | `baselines/uot/fetch_uot.sh` is a no-op when the clone is present (it skips with a message) |

---

## Running experiments

Every (system × benchmark) cell goes through the unified driver:

```bash
PYTHONPATH=. python3 -m bench --row <row> --column <column> --num-cases 50
```

Rows: `bmbe`, `sa`, `cw`, `cot`, `savage_dd`, `savage_bay`, `ac_native`,
`mediq`, `meddxagent`, `cod`, `uot`.
Columns: `patientsim`, `htsim`, `ac`, plus disjoint v1 splits
(`patientsim_v1`, `htsim_v1`) used for the combined n=100 evaluation.

Output goes to
`data/benchmark_results/<column>_50_<row>[_<suffix>]/case_NNN.json`
(one JSON per patient). See `bench/README.md` for the full Row × Column
command table and the per-case JSON schema.

> Note on layouts. The bench driver writes fresh runs to
> `data/benchmark_results/<column>_50_<row>...` (per-case JSON). The
> pre-committed paper results are stored in the older layout
> `data/benchmark_results/{set_a,set_b}/<column>/<row>/*.jsonl`, which is
> what `data/paper_tables/*.py` reads. To regenerate paper tables from
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
# Main table (Table 2): 6 SA + 6 MoBayes × 3 benchmarks
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

### AgentClinic (LLM-elicited from cases)

The AC KB uses the same LLM-driven elicitation pipeline as Section 4.3
(`scripts/data_prep/llmkb/`); the feature/disease vocabulary is grounded
in the AC case set instead of a free-form disease list. Both AC sets
(`agentclinic_a`, `agentclinic_b`) share the same driver, parameterised
by their `raw_input/` directory.

```bash
KB=data/kbs/agentclinic_a   # or agentclinic_b

# (1) extract candidate features from the AC case set
PYTHONPATH=. python3 scripts/data_prep/agentclinic/extract_features.py \
  --cases data/cases/agentclinic_a/selected_cases.jsonl \
  --out   data/cases/agentclinic_a/features.csv

# (2) LLM-driven P(feature|disease) elicitation (same script as Section 4.3)
cd scripts/data_prep/llmkb && \
  LLMKB_MODEL=gpt-5.4-nano \
  LLMKB_DISEASES_JSON_PATH=../../../${KB}/raw_input/diseases.json \
  LLMKB_FEATURES_JSON_PATH=../../../${KB}/raw_input/features.json \
  LLMKB_DISTRIBUTIONS_JSON_PATH=../../../${KB}/raw_output/distributions.json \
  PYTHONPATH=. python3 generate_distributions.py
cd -

# (3) convert distributions.json into the MoBayes KB CSVs
PYTHONPATH=. python3 scripts/data_prep/llmkb/convert_llmkb.py \
  --features      ${KB}/raw_input/features.json \
  --distributions ${KB}/raw_output/distributions.json \
  --output        ${KB}

PYTHONPATH=. python3 scripts/data_prep/validate_kb.py ${KB}
```

### General LLM-elicited KB (Section 4.3)

The Gemini- and GPT-elicited KBs reported in Section 4.3 are built by the
pipeline under `scripts/data_prep/llmkb/`. The pipeline reads paths from
`scripts/data_prep/llmkb/config.yml`; the input disease list lives at
`scripts/data_prep/llmkb/config.yml::paths.diseases_json` (default points
under `data/raw/llmkb/input/`). Place your `diseases<N>.json` there before
running:

```bash
PYTHONPATH=. python3 scripts/data_prep/llmkb/generate_features.py
PYTHONPATH=. python3 scripts/data_prep/llmkb/generate_distributions.py
PYTHONPATH=. python3 scripts/data_prep/llmkb/generate_priors.py
PYTHONPATH=. python3 scripts/data_prep/llmkb/convert_llmkb.py \
  --output data/kbs/llmkb_gpt        # or llmkb_gemini
```

`config.yml` selects the elicitation backbone (e.g. GPT-5.4 or Gemini
3.1 Pro) and the disease list. Both the raw inputs (under
`data/raw/llmkb/input/`) and the resulting KBs (under
`data/kbs/llmkb_{gpt,gemini}/`) are pre-shipped; the
`experiments/llmkb/` scripts read directly from the latter.

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
| `bench/` | Unified benchmark driver (rows × columns) |
| `baselines/` | Adapters for prior diagnostic-dialogue systems |
| `scripts/core/` | Single-session and benchmark entry points |
| `scripts/data_prep/` | Knowledge base construction (DDxPlus, AgentClinic, LLM-elicited) |
| `scripts/baselines/` | Standalone runner scripts for heavyweight baselines (CoD, MEDDxAgent) |
| `scripts/analysis/` | LLM-judge remapping and metric recomputation |
| `experiments/` | Pre-defined paper experiments (`.sh` orchestrators) and figure generators |
| `data/paper_tables/` | LaTeX table generators (read directly from `data/benchmark_results/`) |
| `data/kbs/`, `data/cases/`, `data/benchmark_results/` | Knowledge bases, patient profiles, saved per-case results |
| `data/raw/` | LLM-KB rebuild inputs (disease lists + intermediate generated features/distributions JSONs) |
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
python3 data/paper_tables/render_paper_tables.py     # headline tables
python3 data/paper_tables/ci_tables.py               # bootstrap 95% CIs
python3 data/paper_tables/tokentable.py              # token-usage column
```

Other utilities in `data/paper_tables/` cover paired-bootstrap significance
(`m7_bootstrap_significance.py`), threshold tuning (`find_optimal_tau.py`,
`m4_loocv_tau_tuning.py`), and a matcher-accuracy audit
(`m11_matcher_audit.py`).

### Figures

```bash
PYTHONPATH=. python3 experiments/analysis/generate_paper_plots.py        # cost-vs-scale + persona radar
PYTHONPATH=. python3 experiments/analysis/plot_acc_coverage.py           # accuracy-coverage frontier
PYTHONPATH=. python3 experiments/analysis/scaling_analysis.py --plot     # disease-scaling
PYTHONPATH=. python3 experiments/analysis/failure_analysis.py \
  --data-dir data/kbs/ddxplus \
  --profiles data/cases/ddxplus_patientsim_a/plain/profiles.json \
  --results data/benchmark_results/set_a/ddxplus_patientsim/mobayes/bmbe_*.jsonl
```

---

## License

Released under the MIT License — see `LICENSE`.
Third-party code under `baselines/uot/repo/` retains its upstream license.

## Citation

> *Anonymised for review.* Citation details will be added after acceptance.
