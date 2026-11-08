# baselines/uot

UoT (Hu et al., NeurIPS 2024) adapter. The upstream code
([zhiyuanhubj/UoT](https://github.com/zhiyuanhubj/UoT), commit `e08f17f`) is
cloned into `repo/` (gitignored) and used unmodified.

## Setup

```bash
bash baselines/uot/fetch_uot.sh   # clones zhiyuanhubj/UoT @ e08f17f into baselines/uot/repo/
```

## Running

```bash
PYTHONPATH=. python -m bench --row uot --column {patientsim,htsim,ac}
```

`adapter.py` exposes `UoTHostTask`, `UoTAdapter`, and `build_uot_adapter`,
including the token-tracking patch (`_configure_models._accumulate`) used
by the cost tables.
