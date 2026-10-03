"""Token counting for hosted_vllm deployments through the server's own ``POST /tokenize``.

vLLM renders the chat template server-side, so the only exact count of what a request
costs is the one the server computes. The local tokenizer fallback also counts
``thinking`` blocks that the chat path drops before sending, which doubles the answer
on long agentic conversations.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Final, cast

import httpx

from litellm._logging import verbose_logger
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.utils import LlmProviders, TokenCountResponse

TOKENIZER_TYPE: Final = "hosted_vllm_tokenize"
HOSTED_VLLM_API_BASE_ENV: Final = "HOSTED_VLLM_API_BASE"
_OPENAI_ONLY_ROLES: Final = frozenset({"system", "developer", "tool", "function"})
_ANTHROPIC_BLOCK_TYPES: Final = frozenset(
    {"tool_use", "tool_result", "thinking", "redacted_thinking", "image", "document"}
)


def tokenize_url(api_base: str) -> str:
    """``/tokenize`` lives at the server root, next to the ``/v1`` prefix a deployment's api_base names."""
    trimmed: Final = api_base.rstrip("/")
    return f"{trimmed.removesuffix('/v1')}/tokenize"


def is_anthropic_shaped(messages: Sequence[Mapping[str, object]], system: object) -> bool:
    """Whether ``messages`` use the Anthropic Messages layout rather than OpenAI chat.

    A separate ``system`` value only exists in the Anthropic layout. Otherwise the roles and
    content block types decide; plain-text turns fit both layouts and count as OpenAI chat,
    which needs no translation.
    """
    if system is not None:
        return True
    for message in messages:
        if message.get("role") in _OPENAI_ONLY_ROLES or "tool_calls" in message:
            return False
        if _has_anthropic_blocks(message):
            return True
    return False


def _has_anthropic_blocks(message: Mapping[str, object]) -> bool:
    content: Final = message.get("content")
    if not isinstance(content, list):
        return False
    blocks: Final = cast(Sequence[object], content)  # cast-ok: raw content list from the request body
    return any(isinstance(block, Mapping) and block.get("type") in _ANTHROPIC_BLOCK_TYPES for block in blocks)


def chat_request_for_tokenize(
    model: str,
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[Mapping[str, object]] | None,
    system: object,
    *,
    request_model: str | None = None,
    litellm_params: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """The ``/tokenize`` body for ``messages``: what the chat completion path would send.

    Anthropic-shaped input goes through the same adapter translation as ``/v1/messages``,
    under the model name the client asked for (``request_model``) as that path does, and
    every input through the provider's message transform, so the count matches the prompt
    the server later renders for the real request. ``model`` is the served model name the
    tokenize call names. ``litellm_params`` (the deployment's) reaches the provider
    transform so reasoning-transport settings (``forward_reasoning_content`` /
    ``reasoning_content_field``) render in the count exactly as they render in the
    completion — without them, a deployment that forwards prior reasoning
    under-counts by the whole reasoning share of the conversation.
    """
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )
    from litellm.llms.hosted_vllm.chat.transformation import HostedVLLMChatConfig
    from litellm.types.llms.anthropic import AnthropicMessagesRequest
    from litellm.types.llms.openai import AllMessageValues

    if is_anthropic_shaped(messages, system):
        anthropic_request: Final = cast(  # cast-ok: raw Anthropic-shaped request dicts from the count endpoint
            AnthropicMessagesRequest,
            {
                "model": request_model or model,
                "messages": list(messages),
                "max_tokens": 1,
                **({"system": system} if system is not None else {}),
                **({"tools": list(tools)} if tools else {}),
            },
        )
        translated, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
            anthropic_request, custom_llm_provider=LlmProviders.HOSTED_VLLM.value
        )
        chat_messages: Sequence[AllMessageValues] = translated["messages"]
        chat_tools: Sequence[object] | None = translated.get("tools")
    else:
        chat_messages = cast(Sequence[AllMessageValues], messages)  # cast-ok: OpenAI chat dicts as received
        chat_tools = tools
    chat_request: Final = HostedVLLMChatConfig().transform_request(
        model=model,
        messages=list(chat_messages),
        optional_params={},
        litellm_params=dict(litellm_params or {}),  # mutable-ok: provider request contract
        headers={},
    )
    body: Final[dict[str, object]] = {  # mutable-ok: JSON request body, sent once
        "model": model,
        "messages": chat_request["messages"],
        "add_generation_prompt": True,
    }
    if chat_tools:
        body["tools"] = list(chat_tools)
    return body


class HostedVLLMTokenCounter(BaseTokenCounter):
    """Counts with the deployment's ``/tokenize``; returns None so local counting takes over when it can't."""

    def should_use_token_counting_api(self, custom_llm_provider: str | None = None) -> bool:
        return custom_llm_provider == LlmProviders.HOSTED_VLLM.value

    async def count_tokens(
        self,
        model_to_use: str,
        messages: list[dict[str, object]] | None,
        contents: list[dict[str, object]] | None,
        deployment: dict[str, object] | None = None,
        request_model: str = "",
        tools: list[dict[str, object]] | None = None,
        system: object | None = None,
    ) -> TokenCountResponse | None:
        if not messages:
            return None
        litellm_params: Final = (deployment or {}).get("litellm_params")
        params: Final[Mapping[str, object]] = litellm_params if isinstance(litellm_params, Mapping) else {}
        api_base: Final = params.get("api_base") or os.getenv(HOSTED_VLLM_API_BASE_ENV)
        if not isinstance(api_base, str) or not api_base:
            verbose_logger.warning("hosted_vllm token counting needs an api_base; using the local tokenizer")
            return None
        api_key: Final = params.get("api_key")
        headers: Final = {"Authorization": f"Bearer {api_key}"} if isinstance(api_key, str) and api_key else {}
        try:
            body: Final = chat_request_for_tokenize(
                model_to_use,
                messages,
                tools,
                system,
                request_model=request_model or None,
                litellm_params=params,
            )
            response: Final = await get_async_httpx_client(llm_provider=LlmProviders.HOSTED_VLLM).post(
                tokenize_url(api_base), json=body, headers=headers
            )
            response.raise_for_status()
            payload: Final = cast(object, response.json())  # cast-ok: httpx json() is untyped; shape checked below
            count: Final = payload.get("count") if isinstance(payload, Mapping) else None
            max_model_len: Final = payload.get("max_model_len") if isinstance(payload, Mapping) else None
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
            verbose_logger.warning("hosted_vllm /tokenize failed (%s); using the local tokenizer", error)
            return None
        if not isinstance(count, int):
            verbose_logger.warning("hosted_vllm /tokenize returned no count; using the local tokenizer")
            return None
        return TokenCountResponse(
            total_tokens=count,
            request_model=request_model,
            model_used=model_to_use,
            tokenizer_type=TOKENIZER_TYPE,
            # the token id list is as long as the prompt; keep the scalar fields only
            original_response={"count": count, "max_model_len": max_model_len},
        )
