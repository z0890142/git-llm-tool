"""OpenAI LangChain provider implementation."""

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel

from git_llm_tool.core.config import AppConfig
from git_llm_tool.core.exceptions import ApiError
from git_llm_tool.providers.langchain_base import LangChainProvider


class OpenAiLangChainProvider(LangChainProvider):
    """OpenAI provider using LangChain with intelligent chunking support."""

    def _create_llm(self) -> BaseLanguageModel:
        """Create OpenAI LangChain LLM instance."""
        # Get API key
        api_key = self.config.llm.api_keys.get("openai")
        if not api_key:
            raise ApiError("OpenAI API key not found in configuration")

        # Determine model
        model = self.config.llm.default_model
        if not model.startswith(("gpt-", "o1-")):
            # Fallback to GPT-4o if model doesn't look like OpenAI model
            model = "gpt-4o"

        try:
            # Create LangChain OpenAI instance
            return ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=0.7,
                max_tokens=500,  # Increased for better commit messages
                # LangChain will handle retries and error handling automatically
            )

        except Exception as e:
            raise ApiError(f"Failed to create OpenAI LangChain instance: {e}")

    def __str__(self) -> str:
        """String representation for debugging."""
        return f"OpenAiLangChainProvider(model={self.llm.model_name})"