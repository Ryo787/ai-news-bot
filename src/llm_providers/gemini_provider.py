"""
Gemini Provider - Google Gemini API implementation
"""

import os
from typing import List, Dict, Any, Optional

from google import genai
from google.genai import types

from .base_provider import BaseLLMProvider
from ..logger import setup_logger


logger = setup_logger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize Gemini provider.

        Args:
            api_key: Google API key.
            model: Model name to use.
        """

        api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "Google API key must be provided or set in "
                "GOOGLE_API_KEY environment variable"
            )

        super().__init__(
            api_key=api_key,
            model=model or self.default_model
        )

        # Create the new Gemini client.
        self.client = genai.Client(api_key=self.api_key)

        logger.info(
            f"Gemini provider initialized with model: {self.model}"
        )

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def default_model(self) -> str:
        return "gemini-3.6-flash"

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2000,
        temperature: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate a response using Gemini.
        """

        try:
            logger.debug(
                f"Calling Gemini API with {len(messages)} messages"
            )

            # Convert the repository's message format
            # into a single prompt.
            prompt = self._convert_messages_to_gemini_format(
                messages
            )

            # Gemini 3.6 Flash no longer uses the old
            # temperature parameter in this API path.
            config = types.GenerateContentConfig(
                max_output_tokens=max_tokens
            )

            # Generate the response.
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )

            if response.text:
                return response.text

            raise Exception("No response received from Gemini")

        except Exception as e:
            logger.error(
                f"Gemini API error: {str(e)}",
                exc_info=True
            )
            raise

    def generate_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 2000,
        max_iterations: int = 8,
        tool_handler: Optional[callable] = None,
        **kwargs
    ) -> str:
        """
        Generate a response with tool calling support.

        The original repository has simplified tool support,
        so we preserve that behavior for now.
        """

        try:
            logger.debug(
                f"Calling Gemini API with tools, "
                f"max_iterations={max_iterations}"
            )

            # Use normal generation for now.
            return self.generate(
                messages,
                max_tokens=max_tokens,
                **kwargs
            )

        except Exception as e:
            logger.error(
                f"Gemini API error with tools: {str(e)}",
                exc_info=True
            )
            raise

    def _convert_messages_to_gemini_format(
        self,
        messages: List[Dict[str, str]]
    ) -> str:
        """
        Convert standard messages into a Gemini prompt.
        """

        prompt_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(
                    f"System: {content}"
                )

            elif role == "user":
                prompt_parts.append(
                    f"User: {content}"
                )

            elif role == "assistant":
                prompt_parts.append(
                    f"Assistant: {content}"
                )

        return "\n\n".join(prompt_parts)

    def _convert_tools_to_gemini_format(
        self,
        tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert tool definitions.

        Tool calling is not implemented in the original
        repository's Gemini provider, so this is retained
        for interface compatibility.
        """

        return tools
