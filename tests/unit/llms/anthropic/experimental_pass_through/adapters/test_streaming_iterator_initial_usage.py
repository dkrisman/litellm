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
