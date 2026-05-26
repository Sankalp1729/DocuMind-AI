from __future__ import annotations

from functools import lru_cache

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None


@lru_cache(maxsize=4)
def _encoding(model_name: str = "cl100k_base"):
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0

    encoding = _encoding()
    if encoding is not None:
        return len(encoding.encode(text))

    return max(1, len(text.split()))


def estimate_turn_tokens(question: str, answer: str, context: str | None = None) -> dict[str, int]:
    prompt_tokens = estimate_tokens(question) + estimate_tokens(context)
    completion_tokens = estimate_tokens(answer)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }