# baselines/uot

UoT (Hu et al., NeurIPS 2024) adapter. Vendor source at `repo/` is left
untouched.

## Setup

```bash
bash baselines/uot/fetch_uot.sh   # clones zhiyuanhubj/UoT into ./repo
```

## Running

```bash
PYTHONPATH=. python -m bench --row uot --column {patientsim,htsim,ac}
```

`adapter.py` exposes `UoTHostTask`, `UoTAdapter`, and `build_uot_adapter`,
including the token-tracking patch (`_configure_models._accumulate`) used
by the cost tables.
