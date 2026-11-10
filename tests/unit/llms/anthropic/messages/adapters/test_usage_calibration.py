"""The message_start estimate self-calibrates from each stream's final usage."""

from typing import Final

import pytest

import litellm
from litellm.llms.anthropic.messages.adapters.usage_calibration import (
    calibrated_estimate,
    record_estimate_calibration,
    reset_estimate_calibration,
)


@pytest.fixture(autouse=True)
def _clean_calibration():
    reset_estimate_calibration()
    yield
    reset_estimate_calibration()


def test_estimate_is_identity_until_a_ratio_is_learned():
    assert calibrated_estimate("vllm-a", 10_000) == 10_000


def test_one_finished_turn_calibrates_the_next_estimate():
    record_estimate_calibration("vllm-a", raw_estimate=10_000, true_input_tokens=11_000)
    assert calibrated_estimate("vllm-a", 10_000) == 11_000
    assert calibrated_estimate("vllm-b", 10_000) == 10_000


def test_small_samples_and_degenerate_ratios_are_ignored_or_clamped():
    record_estimate_calibration("vllm-a", raw_estimate=100, true_input_tokens=1_000_000)
    assert calibrated_estimate("vllm-a", 10_000) == 10_000
    record_estimate_calibration("vllm-a", raw_estimate=10_000, true_input_tokens=1_000_000)
    assert calibrated_estimate("vllm-a", 10_000) == 20_000  # clamped at 2.0


def test_ema_tracks_later_samples():
    record_estimate_calibration("vllm-a", raw_estimate=10_000, true_input_tokens=11_000)
    record_estimate_calibration("vllm-a", raw_estimate=10_000, true_input_tokens=10_000)
    # 1.1 + (1.0 - 1.1) * 0.3 = 1.07
    assert calibrated_estimate("vllm-a", 10_000) == 10_700


def test_initial_usage_delta_applies_the_learned_ratio(monkeypatch: pytest.MonkeyPatch):
    from litellm.llms.anthropic.messages.adapters.streaming_iterator import AnthropicStreamWrapper

    monkeypatch.setattr(litellm, "anthropic_messages_estimate_initial_input_tokens", True)
    wrapper: Final = AnthropicStreamWrapper.__new__(AnthropicStreamWrapper)
    wrapper.model = "vllm-a"
    monkeypatch.setattr(AnthropicStreamWrapper, "_estimate_initial_input_tokens", lambda self: 10_000)

    record_estimate_calibration("vllm-a", raw_estimate=10_000, true_input_tokens=11_000)
    usage: Final = wrapper._create_initial_usage_delta()
    assert usage["input_tokens"] == 11_000
    assert wrapper._estimate_calibration == ("vllm-a", 10_000)
