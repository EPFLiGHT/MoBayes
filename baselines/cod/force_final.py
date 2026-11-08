"""Shared CoD post-loop helpers.

Provides:
  - map_to_closed(): LLM mapper from CoD's free-text final diagnosis
    to a closed disease list (DDxPlus 49 or AC-50 31).
  - force_final_via_llm(): when CoD's natural loop ends without emitting
    Diagnosis:, ask an LLM (using the saved dialogue) to either commit
    to a closed-list disease or abstain.
  - posthoc_recompute(): re-run only the post-loop logic on a saved
    canonical row, without GPU model inference. Field names differ
    across runners (Col 1 PatientSim uses ``dialog_history`` /
    ``cod_diagnosis_text_raw`` / ``cod_confidence_distribution_raw``;
    Col 2 HT-sim and Col 3 AC-MedQA use ``dialog`` / ``diagnosis_text``
    / ``confidence_distribution``), so the caller passes them in.

Used by:
  - baselines/cod/run.py --column agentclinic
"""
from __future__ import annotations
from typing import Any, Dict, List


_MAP_PROMPT = """You are a medical-coding assistant. Given a free-form diagnosis from one model and a closed list of {n} diseases, return EXACTLY ONE disease name from the list that semantically matches the free-form output, OR "Unknown" if no listed disease is a semantic match.

Closed disease list:
{disease_list}

Free-form diagnosis to map:
---
diagnosis_text: {diag_text}
top confidence diseases (model's own ranked candidates): {top_conf}
---

Rules:
- Output ONLY the disease name (one of the {n} above) or "Unknown" - no explanation.
- A semantic match is when the free-form output refers to the same disease entity as a listed one, even if wording differs.
- If fundamentally different from any listed disease, return "Unknown".

Answer:"""


_FORCE_PROMPT = """You are a clinical reasoning assistant. The conversation below is between a doctor and a patient. The doctor has not committed to a final diagnosis (ran out of inquiry budget). Based on the dialogue, identify the single most likely disease from the closed list below, or return "Abstain" if the dialogue does not provide enough evidence to commit.

Closed disease list:
{disease_list}

Dialogue:
---
{dialog_text}
---

Rules:
- Output ONLY the disease name (one of the {n} above) or "Abstain". No explanation, no preamble.
- Prefer "Abstain" over guessing when the conversation is sparse, irrelevant, or off-topic.

Answer:"""


def _llm_call(client, model: str, prompt: str, max_tokens: int = 80) -> str:
    is_reasoning = model.startswith(("gpt-5", "o"))
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if is_reasoning:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["temperature"] = 0.0
        kwargs["max_tokens"] = max_tokens
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        print(f"  [warn] LLM call failed: {e}", flush=True)
        return ""
    raw = (resp.choices[0].message.content or "").strip().strip('"').strip("'")
    return raw.split("\n")[0].strip()


def _normalise_to_list(raw: str, diseases: List[str]) -> str:
    """Case-insensitive match raw against the closed list; '' if no match."""
    if not raw:
        return ""
    for d in diseases:
        if d.lower() == raw.lower():
            return d
    return ""


def map_to_closed(diagnosis_text: str, confidence: Dict[str, float],
                  diseases: List[str], llm_client=None,
                  llm_model: str = "gpt-5.4-nano") -> List[str]:
    """LLM mapper: free-text + confidence dict -> closed-list disease.
    Returns [name] or [] for Unknown."""
    if llm_client is None:
        return []
    top_conf = ", ".join(
        f"{d}({c:.2f})"
        for d, c in sorted((confidence or {}).items(), key=lambda kv: -kv[1])[:5]
    ) or "(none)"
    prompt = _MAP_PROMPT.format(
        n=len(diseases),
        disease_list="\n".join(f"- {d}" for d in diseases),
        diag_text=(diagnosis_text or "")[:600],
        top_conf=top_conf,
    )
    raw = _llm_call(llm_client, llm_model, prompt)
    if not raw or raw.lower() == "unknown":
        return []
    matched = _normalise_to_list(raw, diseases)
    return [matched] if matched else []


def force_final_via_llm(dialog: List[Dict], diseases: List[str],
                        llm_client=None, llm_model: str = "gpt-5.4-nano") -> str:
    """LLM diagnose-or-abstain over the saved dialogue. Returns a
    closed-list disease name or '' for Abstain."""
    if llm_client is None or not dialog:
        return ""
    text = "\n".join(
        f"{(t.get('role') or '?').upper()}: {(t.get('content') or '').strip()}"
        for t in dialog if (t.get('content') or '').strip()
    )
    if not text:
        return ""
    prompt = _FORCE_PROMPT.format(
        n=len(diseases),
        disease_list="\n".join(f"- {d}" for d in diseases),
        dialog_text=text[:4000],
    )
    raw = _llm_call(llm_client, llm_model, prompt)
    if not raw or raw.lower() == "abstain":
        return ""
    return _normalise_to_list(raw, diseases)


def finalise(dialog: List[Dict], diagnosis_text: str, confidence: Dict[str, float],
              diseases: List[str], llm_client, llm_model: str = "gpt-5.4-nano") -> Dict[str, Any]:
    """Apply the full post-loop logic. Returns a dict with the keys
    that should overwrite the saved/produced row:
      predicted, predicted_set, top1_hit_*, abstained, stop_reason,
      diagnosis_text, forced (the force-final result for traceability).
    Caller still has to provide ground_truth and merge with their own row.
    """
    forced = ""
    if not diagnosis_text:
        forced = force_final_via_llm(dialog, diseases, llm_client, llm_model)

    if forced:
        ranked, diagnosis_text, abstained = [forced], forced, False
    elif diagnosis_text:
        ranked = map_to_closed(diagnosis_text, confidence, diseases,
                                llm_client, llm_model)
        abstained = not ranked
    else:
        if confidence:
            diagnosis_text = max(confidence.items(), key=lambda kv: kv[1])[0]
            ranked = map_to_closed(diagnosis_text, confidence, diseases,
                                    llm_client, llm_model)
        else:
            ranked = []
        abstained = not ranked

    return {
        "ranked": ranked,
        "abstained": abstained,
        "diagnosis_text": diagnosis_text,
        "forced": forced,
        "stop_reason": ("DDX_PROVIDED" if diagnosis_text and not forced
                        else ("FORCE_FINAL_LLM" if forced else "MAX_TURNS_ABSTAIN")),
    }


def posthoc_recompute(saved: Dict[str, Any], diseases: List[str], llm_client,
                       llm_model: str = "gpt-5.4-nano",
                       *,
                       dialog_field: str = "dialog",
                       diag_text_field: str = "diagnosis_text",
                       conf_field: str = "confidence_distribution") -> Dict[str, Any]:
    """Re-run only the post-loop logic on a saved canonical row, using
    the dialogue / diagnosis_text / confidence already present (no GPU
    model inference). Field-name overrides cover Col 1 (PatientSim
    uses dialog_history etc.) vs Col 2/Col 3 schemas."""
    dialog = saved.get(dialog_field) or []
    diagnosis_text = saved.get(diag_text_field) or ""
    confidence = saved.get(conf_field) or {}
    gt = saved.get("ground_truth") or ""

    res = finalise(dialog, diagnosis_text, confidence, diseases, llm_client, llm_model)

    norm = lambda s: (s or "").lower().strip()
    ranked = res["ranked"]
    abstained = res["abstained"]
    top1 = ranked[0] if ranked else "Unknown"
    top1_hit = (not abstained) and norm(top1) == norm(gt)
    top3_hit = (not abstained) and any(norm(d) == norm(gt) for d in ranked[:3])

    out = dict(saved)
    out.update({
        "predicted": top1,
        "predicted_set": ranked[:5],
        "top1_hit": bool(top1_hit),
        "top3_hit": bool(top3_hit),
        "abstained": bool(abstained),
        "stop_reason": res["stop_reason"],
        "diagnosis_text": res["diagnosis_text"],
        "posthoc_recomputed": True,
    })
    return out
