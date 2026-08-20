"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass
from typing import Any

from openai import OpenAI, OpenAIError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


# Pricing per 1M tokens (gpt-4o-mini estimates)
COST_PER_1M_INPUT_TOKENS = 0.150
COST_PER_1M_OUTPUT_TOKENS = 0.600


class LLMClient:
    """Provider-agnostic LLM client implementation using OpenAI."""

    def __init__(
        self,
        settings: Settings | None = None,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        self.settings = settings or get_settings()
        self.model = model or self.settings.openai_model
        self.temperature = temperature
        self._client: OpenAI | None = None

        if self.settings.openai_api_key:
            self._client = OpenAI(
                api_key=self.settings.openai_api_key,
                timeout=float(self.settings.timeout_seconds),
            )

    def _estimate_cost(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        if input_tokens is None or output_tokens is None:
            return None
        cost = (input_tokens / 1_000_000 * COST_PER_1M_INPUT_TOKENS) + (
            output_tokens / 1_000_000 * COST_PER_1M_OUTPUT_TOKENS
        )
        return round(cost, 6)

    def complete(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> LLMResponse:
        """Return a model completion with retry and token tracking."""
        if not self._client:
            raise AgentExecutionError(
                "OPENAI_API_KEY is not configured. Please add your key to .env file."
            )

        @retry(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=6),
            retry=retry_if_exception_type(OpenAIError),
        )
        def _call_api() -> LLMResponse:
            try:
                response = self._client.chat.completions.create(  # type: ignore[union-attr]
                    model=kwargs.get("model", self.model),
                    temperature=kwargs.get("temperature", self.temperature),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                choice = response.choices[0]
                content = choice.message.content or ""

                input_tokens = None
                output_tokens = None
                if response.usage:
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens

                cost_usd = self._estimate_cost(input_tokens, output_tokens)

                return LLMResponse(
                    content=content.strip(),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                )
            except OpenAIError as exc:
                logger.error("OpenAI API call failed: %s", exc)
                raise

        try:
            return _call_api()
        except Exception as exc:
            raise AgentExecutionError(f"LLM completion failed: {exc}") from exc
