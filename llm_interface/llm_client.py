"""
OpenAI LLM client adapter.

Wraps the OpenAI chat API for both single-shot (.generate) and
multi-turn (.chat) usage across the entire system.
"""
import os
import re
from typing import Any, Dict, List, Optional, Tuple

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def sanitize_openai_text(value: Any) -> str:
    """Remove characters that can break JSON request serialization."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = "".join(ch for ch in value if not 0xD800 <= ord(ch) <= 0xDFFF)
    value = _CONTROL_CHAR_RE.sub("", value)
    return value.encode("utf-8", "replace").decode("utf-8")


def sanitize_openai_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Return a JSON-safe message list for the OpenAI chat API."""
    return [
        {
            "role": sanitize_openai_text(m.get("role", "user")),
            "content": sanitize_openai_text(m.get("content", "")),
        }
        for m in messages
    ]


class OpenAILLMClient:
    """
    Central OpenAI adapter used by all LLM-calling code.

    - generate(prompt)         → single-shot (StructuredParser, question generation)
    - chat(messages, ...)      → multi-turn  (PatientAgent, StandaloneDoctorAgent)
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 temperature: float = 0.0, base_url: Optional[str] = None):
        from openai import OpenAI
        kwargs = {"api_key": api_key or os.environ.get("OPENAI_API_KEY")}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def _create(self, base_kwargs: dict, max_tok: Optional[int] = None):
        """Create a chat completion with provider-agnostic token limit handling."""
        from openai import BadRequestError
        if max_tok is None:
            return self.client.chat.completions.create(**base_kwargs)
        # Try max_completion_tokens first (OpenAI), then max_tokens (OpenRouter)
        for param in ("max_completion_tokens", "max_tokens"):
            try:
                r = self.client.chat.completions.create(
                    **base_kwargs, **{param: max_tok}
                )
                if r.choices is not None:
                    return r
            except BadRequestError:
                continue
        # Last resort: no token limit
        return self.client.chat.completions.create(**base_kwargs)

    def chat(self, messages: List[Dict[str, str]],
             temperature: Optional[float] = None,
             max_tokens: int = 500) -> Tuple[str, Dict[str, int]]:
        """
        Multi-turn chat completion.

        Args:
            messages: Full conversation history [{role, content}, ...]
            temperature: Override instance default if provided.
            max_tokens: Max completion tokens.

        Returns:
            (response_text, {"prompt_tokens": int, "completion_tokens": int})
        """
        base_kwargs = dict(
            model=self.model,
            messages=sanitize_openai_messages(messages),
            temperature=temperature if temperature is not None else self.temperature,
        )
        response = self._create(base_kwargs, max_tokens)
        content = (response.choices[0].message.content
                   if response.choices else None)
        if content is None:
            # Reasoning models may return content=None when reasoning
            # tokens exhaust the budget. Retry with higher limit.
            response = self._create(base_kwargs, max(max_tokens * 4, 2000))
            content = ((response.choices[0].message.content
                        if response.choices else None) or "")
        text = sanitize_openai_text(content).strip()
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
        self.total_usage["prompt_tokens"] += usage["prompt_tokens"]
        self.total_usage["completion_tokens"] += usage["completion_tokens"]
        return text, usage

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Single-shot prompt → response (no conversation history)."""
        text, _ = self.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return text
