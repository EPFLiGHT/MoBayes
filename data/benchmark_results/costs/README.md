# `costs/`

Fallback token estimates for cells whose case JSONs lack
`record.token_usage` (UoT pre-patch, CoD/DiagnosisGPT-34B).

- `uot_samples/<bench>_{a,b}/` — 5-case real-token reruns through the
  patched UoT adapter (per-benchmark, per-split).
- `cod_yi34b_token_pass.json` — offline Yi-34B tokenizer pass over saved
  CoD dialogs (3-pass / 1-pass branch aware).
- `estimated_tokens.json` — aggregated per-cell estimates consumed by
  `data/paper_tables/{tokentable,ci_tables}.py` as a fallback when
  native usage is missing.
