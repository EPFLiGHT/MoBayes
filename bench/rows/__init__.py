"""Row adapters: per-baseline doctor wiring."""

from importlib import import_module
from typing import Protocol, runtime_checkable, List, Any


@runtime_checkable
class Row(Protocol):
    """Contract every row module must satisfy."""

    NAME: str
    """Short id, used in output dir."""

    def make_doctor(
        self,
        disease_universe: List[str],
        doctor_model: str,
        api_key: str,
        max_turns: int,
        column_name: str,
        **kwargs: Any,
    ):
        """Return a doctor exposing the standard interface:

            get_question(patient_response: str) -> (str, bool)
            parse_ddx(text: str) -> List[str]
            parse_possible(text: str) -> List[str]

        Optional: `llm.total_usage` or `token_usage` for cost tracking,
        `reset()` to clear state between cases.
        """


def get(name: str) -> Row:
    """Lazy-load by name: get('cw') → bench.rows.cw module."""
    return import_module(f"bench.rows.{name}")
