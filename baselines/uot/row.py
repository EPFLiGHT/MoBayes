"""UoT (Hu et al., NeurIPS 2024) row — wraps `baselines.uot.adapter`."""

from __future__ import annotations

from typing import Any, List


NAME = "uot"


def make_doctor(
    disease_universe: List[str],
    doctor_model: str,
    api_key: str,
    max_turns: int,
    column_name: str,
    **kwargs: Any,
):
    from baselines.uot.adapter import build_uot_adapter
    fwd = {k: kwargs[k] for k in ("reward_lambda", "skip_self_repo", "base_url")
            if k in kwargs}
    return build_uot_adapter(
        leaf_disease_names=disease_universe,
        doctor_model=doctor_model,
        max_turns=max_turns,
        api_key=api_key,
        **fwd,
    )
