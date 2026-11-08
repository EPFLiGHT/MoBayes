# `costs/`

Token counts for the two baselines whose runners did not log doctor-side
tokens: UoT (its tree-search calls were not tracked) and CoD /
DiagnosisGPT-34B (a local model, no API usage). Every other row stores
`token_usage` in its own case files, so it needs nothing here.

- `uot_samples/<bench>_{a,b}/` — 5-case real-token reruns through the
  patched UoT adapter (per-benchmark, per-split).
- `cod_yi34b_token_pass.json` — offline Yi-34B tokenizer pass over saved
  CoD dialogs (3-pass / 1-pass branch aware).
- `estimated_tokens.json` — aggregated per-cell estimates consumed by
  `scripts/paper_tables/{tokentable,ci_tables}.py` as a fallback when
  native usage is missing.
