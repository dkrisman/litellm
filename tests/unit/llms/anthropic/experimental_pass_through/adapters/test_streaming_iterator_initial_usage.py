"""message_start usage: initial_input_tokens fills input_tokens so clients
that read input accounting from message_start alone (e.g. Claude Code's
context meter) see the real prompt size instead of zeros. The exact split
still arrives in the final message_delta."""

import pytest

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)


async def _empty_stream():
    if False:  # pragma: no cover
        yield None


@pytest.mark.asyncio
async def test_message_start_carries_initial_input_tokens():
    wrapper = AnthropicStreamWrapper(
        completion_stream=_empty_stream(),
        model="test-model",
        initial_input_tokens=123456,
    )
    event = await wrapper.__anext__()
    assert event["type"] == "message_start"
    usage = event["message"]["usage"]
    assert usage["input_tokens"] == 123456
    assert usage["output_tokens"] == 0
    assert usage["cache_creation_input_tokens"] == 0
    assert usage["cache_read_input_tokens"] == 0


@pytest.mark.asyncio
async def test_message_start_defaults_to_zero_without_estimate():
    wrapper = AnthropicStreamWrapper(
        completion_stream=_empty_stream(),
        model="test-model",
    )
    event = await wrapper.__anext__()
    assert event["type"] == "message_start"
    assert event["message"]["usage"]["input_tokens"] == 0


def test_estimate_includes_tool_schemas():
    from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
        _estimate_stream_prompt_tokens,
    )

    base = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
    }
    without_tools = _estimate_stream_prompt_tokens(base)
    with_tools = _estimate_stream_prompt_tokens(
        {
            **base,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": f"tool_{i}",
                        "description": "a tool that does things with its argument",
                        "parameters": {
                            "type": "object",
                            "properties": {"arg": {"type": "string"}},
                        },
                    },
                }
                for i in range(20)
            ],
        }
    )
    assert without_tools is not None and with_tools is not None
    assert with_tools > without_tools


def test_estimate_calibration_ema():
    from litellm.llms.anthropic.experimental_pass_through.adapters.usage_calibration import (
        calibrated_estimate,
        record_estimate_calibration,
        reset_estimate_calibration,
    )

    reset_estimate_calibration()
    try:
        # No ratio learned yet: identity.
        assert calibrated_estimate("dep", 100_000) == 100_000
        # Below the sample floor: ignored.
        record_estimate_calibration("dep", 1_000, 2_000)
        assert calibrated_estimate("dep", 100_000) == 100_000
        # First real sample sets the ratio outright (-10% bias -> x1.111...).
        record_estimate_calibration("dep", 180_000, 200_000)
        assert abs(calibrated_estimate("dep", 180_000) - 200_000) <= 1
        # Clamped: a wild turn cannot poison the EMA beyond the bounds.
        record_estimate_calibration("dep2", 10_000, 1_000_000)
        assert calibrated_estimate("dep2", 10_000) == 20_000
        # Keys are independent.
        assert abs(calibrated_estimate("dep", 180_000) - 200_000) <= 1
    finally:
        reset_estimate_calibration()


@pytest.mark.asyncio
async def test_stream_records_calibration_from_final_usage():
    from litellm.llms.anthropic.experimental_pass_through.adapters.usage_calibration import (
        calibrated_estimate,
        reset_estimate_calibration,
    )
    from litellm.types.utils import ModelResponseStream, Usage

    reset_estimate_calibration()
    try:
        finish = ModelResponseStream(
            choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
            model="m",
        )
        usage_chunk = ModelResponseStream(choices=[], model="m")
        usage_chunk.usage = Usage(prompt_tokens=200_000, completion_tokens=5, total_tokens=200_005)

        async def stream():
            yield finish
            yield usage_chunk

        wrapper = AnthropicStreamWrapper(
            completion_stream=stream(),
            model="m",
            initial_input_tokens=180_000,
            estimate_calibration=("dep-stream", 180_000),
        )
        events = []
        async for event in wrapper:
            events.append(event)
        assert abs(calibrated_estimate("dep-stream", 180_000) - 200_000) <= 1
    finally:
        reset_estimate_calibration()
