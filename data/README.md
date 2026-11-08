# `data/`

Three top-level groups:

- `kbs/` — knowledge bases (priors, likelihoods, feature universe).
  - `ddxplus/` — empirical DDxPlus KB (49 diseases).
  - `agentclinic_a/`, `agentclinic_b/` — two LLM-elicited AgentClinic-MedQA KBs.
  - `scaling/` — random K-disease subsets of the DDxPlus KB (disease-scaling
    experiment; `manifest.json` lists the subsets).

- `cases/` — patient profiles fed to the simulators.
  - `ddxplus_patientsim_{a,b}/<persona>/profiles.json` — PatientSim profiles
    across five personas (`plain`, `dazed`, `distrustful`, `overanxious`,
    `verbose_lowrecall`).
  - `ddxplus_htsim_{a,b}/` — MEDDxAgent history-taking-simulator profiles.
  - `agentclinic_{a,b}/` — disjoint 50-case subsets of AgentClinic-MedQA.
  - `llmkb_gpt/`, `llmkb_gemini/` — LLM-elicited KB evaluation patients.

- `benchmark_results/` — saved results that the paper tables and figures
  read (one `case_*.json` per session, or one `.jsonl` per run).
  - `set_a/`, `set_b/` — the two disjoint 50-patient halves; aggregating
    them gives the combined n=100 reported in the main tables.
  - `judged/{set_a,set_b}/` — same cases re-scored by the closed-list
    LLM judge (`top1_hit_judged`, `top3_hit_judged`, `abstained_judged`).
    Tables read these fields, not the strict-match originals.
  - `costs/` — token counts for UoT and CoD, the two baselines whose runners
    did not log doctor tokens (see `costs/README.md`).

Within each `set_*/<benchmark>/` cell, results are split by category:
`mobayes/`, `standalone/`, `baselines/`, and (where applicable)
`ablations/`. Directories ending in `_raw` hold a baseline's native output
format next to its converted results: MediQ's `results.jsonl`
(`baselines/mediq_raw/`) and MEDDxAgent's per-run `results.json` /
`token_usage.json` (`_raw/meddxagent_*`, `baselines/meddxagent*_raw/`).
`set_a/scaling/` holds the disease-scaling runs (`{bmbe,sa}_k<K>_s<seed>/`).

The table generators in `scripts/paper_tables/` read the case files above
and write LaTeX into the top-level `tables/` directory.
