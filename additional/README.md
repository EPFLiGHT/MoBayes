# Additional experiments

Self-contained copy of the analyses in appendix C.15 to C.20. The commands below read saved logs and make no LLM calls.
Run from this folder. Needs Python 3 with `numpy`, `pandas`, `matplotlib`.

| Appendix | Experiment | Command |
|---|---|---|
| C.15 | True disease removed (OOD) | `python3 experiments/reproduce_missing_disease.py` |
| C.16 | KB noise and coverage | `PYTHONPATH=. python3 experiments/kb_noise.py --seeds 20 --tau 0.3` |
| C.17 | Parser audit (automatic check) | `PYTHONPATH=. python3 experiments/parser_audit.py` |
| C.18 | Model-matched pairings | `python3 experiments/analysis/reproduce_w1_2x2.py` |
| C.19 | Full AgentClinic-MedQA | `PYTHONPATH=. python3 experiments/agentclinic_full/tau_tune.py` |
| C.20 | VivaBench feasibility | `cd vivabench && python3 eig_dialog.py --ci` |

Rerun from scratch (live, needs `OPENAI_API_KEY`, `openai`, `python-dotenv`)
- C.15 `PYTHONPATH=. python3 experiments/missing_disease.py --arm both --num-cases 50 --out-suffix rerun`

Layout
- `data/kbs/` : `ddxplus` (empirical) and `agentclinic_full_cat` (177 diseases, 828 features).
- `vivabench/kb_grounded`, `vivabench/kb50` : VivaBench KBs (29 diseases, 264 features).
- `data/benchmark_results/` : logged dialogues each script replays.
- `world_model/`, `decision_policy/`, `llm_interface/`, `scripts/`, `baselines/`, `config.py` : MoBayes modules the scripts import.
