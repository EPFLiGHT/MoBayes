# `data/`

Three top-level groups:

- `kbs/` — knowledge bases (priors, likelihoods, feature universe).
  - `ddxplus/` — empirical DDxPlus KB (49 diseases).
  - `agentclinic_a/`, `agentclinic_b/` — two LLM-elicited AgentClinic-MedQA KBs.
  - `scaling/` — disease-scaling sweeps over the DDxPlus KB.

- `cases/` — patient profiles fed to the simulators.
  - `ddxplus_patientsim_{a,b}/<persona>/profiles.json` — PatientSim profiles
    across five personas (`plain`, `dazed`, `distrustful`, `overanxious`,
    `verbose_lowrecall`).
  - `ddxplus_htsim_{a,b}/` — MEDDxAgent history-taking-simulator profiles.
  - `agentclinic_{a,b}/` — disjoint 50-case subsets of AgentClinic-MedQA.
  - `llmkb_gpt/`, `llmkb_gemini/` — LLM-elicited KB evaluation patients.

- `benchmark_results/` — saved per-case results (one `case_*.json` per
  session) that the paper tables and figures read.
  - `set_a/`, `set_b/` — the two disjoint 50-patient halves; aggregating
    them gives the combined n=100 reported in the main tables.
  - `judged/{set_a,set_b}/` — same cases re-scored by the closed-list
    LLM judge (`top1_hit_judged`, `top3_hit_judged`, `abstained_judged`).
    Tables read these fields, not the strict-match originals.
  - `costs/` — fallback token estimates for cells whose case JSONs lack
    `record.token_usage` (see `costs/README.md`).

Within each `set_*/<benchmark>/` cell, results are split by category:
`mobayes/`, `standalone/`, `baselines/`, and (where applicable)
`ablations/`.

`paper_tables/` sits alongside the data it consumes: scripts here read
the case files above and write LaTeX into the top-level `tables/`
directory.
