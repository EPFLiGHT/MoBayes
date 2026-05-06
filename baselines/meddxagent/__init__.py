"""MEDDxAgent-specific baseline adapters and runners.

Lazy-load top-level symbols so that lightweight submodules (usage_tracker,
etc.) can be imported in venvs that lack heavy deps like `datasets`.
"""

import importlib

__all__ = [
    "BMBEDDxPlus50",
    "BMBEPatientAgent",
    "ClosedWorldDDxRAG",
]


def __getattr__(name: str):
    if name == "BMBEDDxPlus50":
        return importlib.import_module("baselines.meddxagent.benchmark").BMBEDDxPlus50
    if name == "BMBEPatientAgent":
        return importlib.import_module("baselines.meddxagent.patient").BMBEPatientAgent
    if name == "ClosedWorldDDxRAG":
        return importlib.import_module("baselines.meddxagent.rag").ClosedWorldDDxRAG
    raise AttributeError(name)
