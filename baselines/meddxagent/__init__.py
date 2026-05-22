"""MEDDxAgent-specific baseline adapters and runners.

Lazy-load top-level symbols so that lightweight submodules (usage_tracker,
etc.) can be imported in venvs that lack heavy deps like `datasets`.
"""

import importlib

__all__ = [
    "MoBayesDDxPlus50",
    "MoBayesPatientAgent",
    "ClosedWorldDDxRAG",
]


def __getattr__(name: str):
    if name == "MoBayesDDxPlus50":
        return importlib.import_module("baselines.meddxagent.benchmark").MoBayesDDxPlus50
    if name == "MoBayesPatientAgent":
        return importlib.import_module("baselines.meddxagent.patient").MoBayesPatientAgent
    if name == "ClosedWorldDDxRAG":
        return importlib.import_module("baselines.meddxagent.rag").ClosedWorldDDxRAG
    raise AttributeError(name)
