"""Ollama LangChain provider implementation."""

from typing import Optional
from langchain_ollama import OllamaLLM

from git_llm_tool.core.config import AppConfig
from git_llm_tool.core.exceptions import ApiError
from git_llm_tool.providers.langchain_base import LangChainProvider


class OllamaLangChainProvider(LangChainProvider):
    """Ollama provider using LangChain."""

    def __init__(self, config: AppConfig):
        """Initialize Ollama provider."""
        try:
            # Initialize the Ollama LLM
            llm = OllamaLLM(
                model=config.llm.ollama_model,
                base_url=config.llm.ollama_base_url,
                temperature=0.1  # Keep it deterministic for commit messages
            )

            super().__init__(config, llm)

        except Exception as e:
            raise ApiError(f"Failed to initialize Ollama provider: {e}")

    def validate_config(self) -> bool:
        """Validate Ollama configuration."""
        try:
            # Test connection to Ollama
            response = self.llm.invoke("test")
            return True
        except Exception as e:
            raise ApiError(f"Ollama validation failed: {e}")

    def get_model_info(self) -> dict:
        """Get information about the current model."""
        return {
            "provider": "ollama",
            "model": self.config.llm.ollama_model,
            "base_url": self.config.llm.ollama_base_url,
            "local": True
        }