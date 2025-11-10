"""Anthropic Claude LangChain provider implementation."""

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseLanguageModel

from git_llm_tool.core.config import AppConfig
from git_llm_tool.core.exceptions import ApiError
from git_llm_tool.providers.langchain_base import LangChainProvider


class AnthropicLangChainProvider(LangChainProvider):
    """Anthropic Claude provider using LangChain with intelligent chunking support."""

    def _create_llm(self) -> BaseLanguageModel:
        """Create Anthropic LangChain LLM instance."""
        # Get API key
        api_key = self.config.llm.api_keys.get("anthropic")
        if not api_key:
            raise ApiError("Anthropic API key not found in configuration")

        # Determine model
        model = self.config.llm.default_model
        if not model.startswith("claude-"):
            # Fallback to Claude 3.5 Sonnet if model doesn't look like Anthropic model
            model = "claude-3-5-sonnet-20241024"

        try:
            # Create LangChain Anthropic instance
            return ChatAnthropic(
                api_key=api_key,
                model=model,
                temperature=0.7,
                max_tokens=500,  # Increased for better commit messages
                # LangChain will handle retries and error handling automatically
            )

        except Exception as e:
            raise ApiError(f"Failed to create Anthropic LangChain instance: {e}")

    def __str__(self) -> str:
        """String representation for debugging."""
        return f"AnthropicLangChainProvider(model={self.llm.model})"