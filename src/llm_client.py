"""LLM client with retry mechanism for OpenAI-compatible APIs."""

import logging
from typing import List, Dict, Optional

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .config import LLMConfig, RetryConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM client supporting OpenAI-compatible APIs with retry mechanism."""

    def __init__(self, llm_config: LLMConfig, retry_config: RetryConfig):
        """Initialize LLM client.

        Args:
            llm_config: LLM service configuration.
            retry_config: Retry configuration.
        """
        self.config = llm_config
        self.retry_config = retry_config

        self.client = OpenAI(
            api_key=llm_config.api_key,
            base_url=llm_config.api_base_url,
            timeout=llm_config.timeout,
        )

        # Create retry decorator with config
        self._chat_with_retry = self._create_retry_wrapper()

    def _create_retry_wrapper(self):
        """Create a retry-wrapped chat completion function."""
        @retry(
            stop=stop_after_attempt(self.retry_config.max_retries),
            wait=wait_exponential(
                multiplier=self.retry_config.retry_interval,
                max=self.retry_config.max_retry_interval,
            ),
            retry=retry_if_exception_type((Exception,)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        def _chat(
            messages: List[Dict[str, str]],
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            enable_thinking: Optional[bool] = None,
        ) -> str:
            # Build request parameters
            params = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.config.temperature,
            }

            # Determine max_tokens (use override if provided, else config)
            effective_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
            # Only set max_tokens if > 0 (0 means no limit)
            if effective_max_tokens > 0:
                params["max_tokens"] = effective_max_tokens

            # Determine enable_thinking (use override if provided, else config)
            # A value of None means "not set", so we need to handle sentinel
            effective_thinking = enable_thinking if enable_thinking is not None else self.config.enable_thinking
            
            # Add Qwen3 thinking mode if configured (for vLLM deployment)
            if effective_thinking is not None:
                params["extra_body"] = {
                    "chat_template_kwargs": {
                        "enable_thinking": effective_thinking
                    }
                }

            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content

        return _chat

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        """Send chat completion request with retry.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Optional temperature override.
            max_tokens: Optional max_tokens override.
            enable_thinking: Optional Qwen3 thinking mode override.

        Returns:
            Model response content.

        Raises:
            Exception: If all retries fail.
        """
        try:
            response = self._chat_with_retry(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )
            return response
        except Exception as e:
            logger.error(
                f"LLM request failed after {self.retry_config.max_retries} retries: {e}")
            raise

    def chat_simple(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        """Simple chat interface with system and user prompts.

        Args:
            system_prompt: System prompt content.
            user_prompt: User prompt content.
            temperature: Optional temperature override.
            max_tokens: Optional max_tokens override.
            enable_thinking: Optional Qwen3 thinking mode override.

        Returns:
            Model response content.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
