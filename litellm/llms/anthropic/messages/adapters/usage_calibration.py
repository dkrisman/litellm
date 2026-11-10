"""Self-calibrating prompt-token estimates for streamed ``message_start`` usage.

The ``message_start`` estimate (see ``_estimate_stream_prompt_tokens``) counts the
translated messages with a local tokenizer, but the backend renders the prompt
through its own chat template — tool schemas, reasoning wrappers and message
framing all land differently, so the estimate carries a provider-specific bias
(measured ~-10% against a vLLM/Qwen deployment). Every finished stream reports
the backend's true prompt accounting in its final usage, so the bias is
observable: this module keeps an exponential moving average of
``true / raw_estimate`` per deployment model and scales the next estimate by it.
One finished turn gets within a couple percent; the EMA then tracks template or
workload changes on its own. Calibration samples below ``_MIN_SAMPLE_TOKENS``
are ignored (fixed framing overhead dominates tiny prompts) and ratios are
clamped so one aberrant turn cannot poison the estimate.
"""

from __future__ import annotations

import threading
from typing import Final

_ALPHA: Final = 0.3
_MIN_SAMPLE_TOKENS: Final = 5000
_RATIO_BOUNDS: Final = (0.5, 2.0)

_ratios: dict[str, float] = {}  # mutable-ok: module-level EMA state, guarded by _lock
_lock: Final = threading.Lock()


def calibrated_estimate(key: str, raw_estimate: int) -> int:
    """``raw_estimate`` scaled by the deployment's observed render ratio (identity until one is learned)."""
    with _lock:
        ratio = _ratios.get(key)
    if ratio is None:
        return raw_estimate
    return int(raw_estimate * ratio)


def record_estimate_calibration(key: str, raw_estimate: int | None, true_input_tokens: int) -> None:
    """Fold one finished stream's ``true / raw_estimate`` ratio into the deployment's EMA."""
    if not key or not raw_estimate or raw_estimate < _MIN_SAMPLE_TOKENS or true_input_tokens <= 0:
        return
    low, high = _RATIO_BOUNDS
    ratio: Final = min(high, max(low, true_input_tokens / raw_estimate))
    with _lock:
        previous = _ratios.get(key)
        _ratios[key] = ratio if previous is None else previous + (ratio - previous) * _ALPHA


def reset_estimate_calibration() -> None:
    """Testing hook: drop all learned ratios."""
    with _lock:
        _ratios.clear()
