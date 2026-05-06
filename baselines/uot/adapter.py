"""UoT (Hu et al., NeurIPS 2024) adapter for `--doctor-mode uot`.

Vendor source at `baselines/uot/repo` is left untouched; cooperation is
through a host-side task class plus scoped runtime patches.
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_HERE = Path(__file__).resolve().parent
_REPO_SRC = _HERE / "repo" / "src"


def _ensure_uot_on_path() -> None:
    if not _REPO_SRC.exists():
        raise RuntimeError(
            f"UoT repo not found at {_REPO_SRC}. "
            f"Run `bash baselines/uot/fetch_uot.sh`."
        )
    p = str(_REPO_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)


def _configure_models(model: str, api_key: Optional[str], base_url: Optional[str],
                       usage_target: Optional[Dict[str, int]] = None,
                       ) -> None:
    _ensure_uot_on_path()
    from uot import models  # type: ignore[import-not-found]
    from openai import OpenAI

    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
    if base_url and not resolved_key:
        resolved_key = "EMPTY"
    kwargs: Dict[str, Any] = {"api_key": resolved_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)

    from openai import BadRequestError
    _orig_create = client.chat.completions.create

    def _accumulate(res):
        if usage_target is None:
            return
        u = getattr(res, "usage", None)
        if u is None:
            return
        usage_target["prompt_tokens"] = (usage_target.get("prompt_tokens", 0)
                                          + (getattr(u, "prompt_tokens", 0) or 0))
        usage_target["completion_tokens"] = (usage_target.get("completion_tokens", 0)
                                              + (getattr(u, "completion_tokens", 0) or 0))
        usage_target["total_tokens"] = (usage_target.get("total_tokens", 0)
                                         + (getattr(u, "total_tokens", 0) or 0))

    def _create_compat(**call_kwargs):
        # gpt-5/o1/o3/o4 reasoning models reject `temperature` and need
        # `max_completion_tokens`; their reasoning tokens also consume
        # the budget before any visible output, so scale up by 6x.
        model_name = str(call_kwargs.get("model", "")).lower()
        is_reasoning = (model_name.startswith(("gpt-5", "o1", "o3", "o4"))
                         and "nano" not in model_name)
        if is_reasoning and "temperature" in call_kwargs:
            call_kwargs.pop("temperature", None)
        if "max_tokens" in call_kwargs:
            mt = call_kwargs.pop("max_tokens")
            if is_reasoning:
                mt = max(mt * 6, 4000)
            for name in ("max_completion_tokens", "max_tokens"):
                try:
                    res = _orig_create(**call_kwargs, **{name: mt})
                    _accumulate(res)
                    return res
                except BadRequestError:
                    continue
            res = _orig_create(**call_kwargs)
            _accumulate(res)
            return res
        res = _orig_create(**call_kwargs)
        _accumulate(res)
        return res

    client.chat.completions.create = _create_compat
    models.client = client

    # Replace vendor's `gpt_response`: vendor sleeps `time_gap.get(model, 3)`
    # seconds before every call (default 3s). Setting `time_gap = {}` does
    # nothing — `dict.get(missing, 3)` still returns 3. We need to replace
    # the function. Keep the retry-on-error pattern but drop the leading
    # sleep; on failure, sleep 1s and recurse.
    def _gpt_response_no_sleep(message, model="gpt-4", temperature=0,
                                  max_tokens=500):
        try:
            res = models.client.chat.completions.create(
                model=model, messages=message, temperature=temperature, n=1,
                max_tokens=max_tokens,
            )
            return res.choices[0].message.content
        except Exception as e:
            print(e)
            time.sleep(1)
            return _gpt_response_no_sleep(message, model, temperature, max_tokens)

    models.gpt_response = _gpt_response_no_sleep


class UoTHostTask:
    """Drop-in replacement for upstream's `MDTask` with paper-default knobs."""

    def __init__(self, leaf_disease_names: List[str], guesser_model: str,
                  *, max_turns: int = 5, reward_lambda: float = 0.4) -> None:
        _ensure_uot_on_path()
        from uot.chat_utils import import_prompts_by_task  # type: ignore[import-not-found]
        from uot.uot import UoTNode  # type: ignore[import-not-found]
        self._UoTNode = UoTNode

        self.task = "md"
        self.dataset = "host"
        self.free_answer = False
        self.max_turn = max_turns
        self.prompts = import_prompts_by_task("md")
        self.set: List[str] = list(leaf_disease_names)
        self.data: List[Dict[str, Any]] = []
        self.root: Optional[Any] = None

        self.guesser_model = guesser_model
        self.examiner_model = guesser_model
        self.temperature = 0.0

        # Paper §2 + Table 2 defaults.
        self.reward_lambda = reward_lambda
        self.n_extend_layers = 3
        self.n_potential_actions = 3
        self.n_pruned_nodes = 0
        self.expected_action_tokens = 50
        self.expected_target_tokens = 10
        self.none_acc_reward = False
        self.expected_reward_method = "avg"

        # Closed-set: doctor knows Ω up front (paper §2.1).
        self.open_set_size = -1
        self.size_to_renew = -1
        self.n_pre_ask = 0
        self.inform = True

    def create_root(self) -> None:
        self.root = self._UoTNode("ROOT", True, list(self.set), None,
                                    self.guesser_model)
        self.root.set_config(self.n_extend_layers, self.none_acc_reward,
                              self.expected_reward_method)


class UoTAdapter:
    """Doctor-side adapter: `get_question(patient_response) -> (text, is_final)`."""

    # Cap consecutive ambiguous answers before forcing forward, so a
    # patient stuck on "I don't know" can't hang the session.
    _MAX_CONSECUTIVE_AMBIGUOUS = 2

    def __init__(self, task: UoTHostTask, *, skip_self_repo: bool = False,
                  token_usage: Optional[Dict[str, int]] = None) -> None:
        _ensure_uot_on_path()
        self.task = task
        self.skip_self_repo = skip_self_repo
        self.task.create_root()
        self._node: Optional[Any] = None
        self._pending_question_text: Optional[str] = None
        self._self_repo_consumed = False
        self.turn = 0
        self._consecutive_ambiguous = 0
        self.max_turns = task.max_turn
        self.per_turn_trace: List[Dict[str, Any]] = []
        # Live dict shared with the patched OpenAI client (see
        # `_configure_models._accumulate`). Runner reads `doctor.token_usage`.
        self.token_usage = (token_usage if token_usage is not None
                              else {"prompt_tokens": 0, "completion_tokens": 0,
                                    "total_tokens": 0})
        self.messages: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.task.create_root()
        self._node = None
        self._pending_question_text = None
        self._self_repo_consumed = False
        self.turn = 0
        self._consecutive_ambiguous = 0
        self.per_turn_trace = []
        self.messages = []

    def should_normalize_predictions(self) -> bool:
        return False

    def get_question(self, patient_response: Optional[str] = None
                      ) -> Tuple[str, bool]:
        from uot.method import get_guesser_response  # type: ignore[import-not-found]

        if self._node is None:
            # First call: optionally consume chief complaint via upstream's
            # `handle_self_repo`. With `skip_self_repo=True`, paper §2.1
            # strict CS — Ω stays full until the first asked question.
            if patient_response and not self.skip_self_repo:
                self._node = self.task.root.handle_self_repo(self.task,
                                                                patient_response)
                self._self_repo_consumed = True
            else:
                self._node = self.task.root
            if patient_response:
                self.messages.append({
                    "role": "user",
                    "content": self.task.prompts.guesser_prologue.format(
                        repo=patient_response.strip()),
                })
        elif patient_response is not None and self._pending_question_text is not None:
            self.messages.append({"role": "user",
                                   "content": f"Patient: {patient_response}"})
            answer = _parse_yes_no(patient_response)
            # Only consume a turn-cap slot when the answer parses AND
            # advances the search tree. Ambiguous answers re-prompt.
            if answer is not None:
                self._consecutive_ambiguous = 0
                advanced = self._node.ans2node(answer)
                if advanced is not None:
                    self._node = advanced
                    self.turn += 1
                    self._record_turn(answer)
                else:
                    self.per_turn_trace.append({
                        "turn": self.turn,
                        "question": getattr(self._node, "question", None),
                        "answer": bool(answer),
                        "items_after": list(getattr(self._node, "items", [])),
                        "items_other_branch": [],
                        "idiv_reward": float("nan"),
                        "no_op_reason": "ans2node_returned_None",
                    })
            else:
                self._consecutive_ambiguous += 1
                if (self._consecutive_ambiguous
                        >= self._MAX_CONSECUTIVE_AMBIGUOUS):
                    advanced = self._node.ans2node(False)
                    if advanced is not None:
                        self._node = advanced
                        self.turn += 1
                        self._record_turn(False)
                        self.per_turn_trace[-1]["no_op_reason"] = (
                            "force_no_after_consecutive_ambiguous")
                    self._consecutive_ambiguous = 0
                else:
                    self.per_turn_trace.append({
                        "turn": self.turn,
                        "question": getattr(self._node, "question", None),
                        "answer": None,
                        "items_after": list(getattr(self._node, "items", [])),
                        "items_other_branch": [],
                        "idiv_reward": float("nan"),
                        "no_op_reason": "ambiguous_patient_response",
                    })
            self._pending_question_text = None
            if self.turn >= self.max_turns or self._is_terminal():
                return self._emit_final()

        try:
            self._node, question_text, _flag = get_guesser_response(
                self.task, self.messages, self.turn + 1, self._node
            )
        except Exception:
            return self._emit_final()
        if not question_text:
            return self._emit_final()

        self._pending_question_text = f"[ASK] {question_text}"
        self.messages.append({"role": "system", "content": question_text})
        return self._pending_question_text, False

    def parse_ddx(self, text: str) -> List[str]:
        # Lex sort on residual Ω: cross-run determinism (vs hash-randomised
        # upstream `set(...)`).
        items = list(getattr(self._node, "items", []))
        if not items:
            return []
        return sorted(items, key=lambda d: str(d).strip())[:5]

    def parse_possible(self, text: str) -> List[str]:
        return self.parse_ddx(text)

    def get_session_metrics(self) -> Dict[str, Any]:
        items = list(getattr(self._node, "items", []))
        n = max(1, len(items))
        return {
            "leaf_diseases": list(self.task.set),
            "confidence": 1.0 / float(n),
            "final_margin": 0.0,
            "final_entropy": float(math.log2(n)),
            "final_posterior": [],
            "benchmark_metrics": {
                "uot_residual_omega_size": n,
                "uot_skip_self_repo": bool(self.skip_self_repo),
                "uot_self_repo_consumed": bool(self._self_repo_consumed),
            },
            "per_turn_trace": list(self.per_turn_trace),
        }

    def _record_turn(self, answer: bool) -> None:
        node = self._node
        partner = node.ans2node(not answer) if node else None
        items_self = list(getattr(node, "items", []))
        items_partner = list(getattr(partner, "items", [])) if partner else []
        try:
            reward_self = float(node.idiv_reward)
        except Exception:
            reward_self = float("nan")
        self.per_turn_trace.append({
            "turn": self.turn,
            "question": getattr(node, "question", None),
            "answer": bool(answer),
            "items_after": items_self,
            "items_other_branch": items_partner,
            "idiv_reward": reward_self,
        })

    def _is_terminal(self) -> bool:
        return bool(getattr(self._node, "is_terminal", False))

    def _emit_final(self) -> Tuple[str, bool]:
        items = self.parse_ddx("") or sorted(self.task.set, key=str)[:5]
        text = "[DDX] " + ", ".join(items)
        return text, True


def _parse_yes_no(text: str) -> Optional[bool]:
    """Yes/no answer parser. Hedges ("don't know", "maybe") return None."""
    import re as _re
    t = (text or "").strip().lower()
    if not t:
        return None
    head = t[:80]
    ambiguous = (
        r"\b(i\s*don'?t\s*know|don'?t\s*know|not\s*sure|unsure|"
        r"can'?t\s*remember|hard\s*to\s*say|maybe|perhaps|sometimes|"
        r"occasionally|i\s*think\s*so|i\s*guess|kinda|sorta)\b"
    )
    if _re.search(ambiguous, head):
        return None
    # Negatives first so "not really" / "no" / "nope" / "nothing" don't
    # leak into the yes branch via partial matches.
    if _re.match(r"\s*(no|nope|nah|never|none|nothing|negative|"
                  r"not\s+(really|at\s*all|that\s*i|particularly))\b",
                  head):
        return False
    if _re.search(r"\bno\b", head[:30]) and not _re.search(
            r"\b(yes|yeah|yep)\b", head[:30]):
        return False
    if _re.match(r"\s*(yes|yeah|yep|yup|correct|right|true|"
                  r"absolutely|definitely|of\s*course|sure|"
                  r"i\s*do|i\s*have|i\s*am)\b", head):
        return True
    if _re.search(r"\b(yes|yeah|yep)\b", head[:30]):
        return True
    return None


def build_uot_adapter(*, leaf_disease_names: List[str], doctor_model: str,
                       max_turns: int = 5,
                       reward_lambda: float = 0.4,
                       skip_self_repo: bool = False,
                       api_key: Optional[str] = None,
                       base_url: Optional[str] = None) -> UoTAdapter:
    token_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0,
                                     "total_tokens": 0}
    _configure_models(doctor_model, api_key, base_url, usage_target=token_usage)
    task = UoTHostTask(
        leaf_disease_names=leaf_disease_names,
        guesser_model=doctor_model,
        max_turns=max_turns,
        reward_lambda=reward_lambda,
    )
    return UoTAdapter(task, skip_self_repo=skip_self_repo, token_usage=token_usage)
