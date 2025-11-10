"""Gemini LangChain provider implementation."""

from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI

from git_llm_tool.core.config import AppConfig
from git_llm_tool.core.exceptions import ApiError
from git_llm_tool.providers.langchain_base import LangChainProvider


class GeminiLangChainProvider(LangChainProvider):
    """Google Gemini provider using LangChain."""

    def __init__(self, config: AppConfig):
        """Initialize Gemini provider."""
        try:
            # Get API key
            api_key = config.llm.api_keys.get("google")
            if not api_key:
                raise ApiError("Google API key not found in configuration")

            # Determine model
            model = config.llm.default_model
            if not model.startswith("gemini-"):
                # Fallback to Gemini 1.5 Pro if model doesn't look like Gemini model
                model = "gemini-1.5-pro"

            # Initialize the Gemini LLM
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=0.1,  # Keep it deterministic for commit messages
                convert_system_message_to_human=True  # Required for Gemini
            )

            super().__init__(config, llm)

        except Exception as e:
            raise ApiError(f"Failed to initialize Gemini provider: {e}")

    def validate_config(self) -> bool:
        """Validate Gemini configuration."""
        try:
            # Test with a simple request
            response = self.llm.invoke("test")
            return True
        except Exception as e:
            raise ApiError(f"Gemini validation failed: {e}")

    def get_model_info(self) -> dict:
        """Get information about the current model."""
        return {
            "provider": "google",
            "model": self.config.llm.default_model,
            "service": "Google Generative AI",
            "local": False
        }