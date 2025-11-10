"""Azure OpenAI LangChain provider implementation."""

from langchain_openai import AzureChatOpenAI
from langchain_core.language_models import BaseLanguageModel

from git_llm_tool.core.config import AppConfig
from git_llm_tool.core.exceptions import ApiError
from git_llm_tool.providers.langchain_base import LangChainProvider


class AzureOpenAiLangChainProvider(LangChainProvider):
    """Azure OpenAI provider using LangChain with intelligent chunking support."""

    def _create_llm(self) -> BaseLanguageModel:
        """Create Azure OpenAI LangChain LLM instance."""
        # Get Azure OpenAI configuration
        azure_config = self.config.llm.azure_openai
        if not azure_config.get("endpoint"):
            raise ApiError("Azure OpenAI endpoint not found in configuration")

        api_key = self.config.llm.api_keys.get("azure_openai")
        if not api_key:
            raise ApiError("Azure OpenAI API key not found in configuration")

        # Default values for Azure OpenAI
        api_version = azure_config.get("api_version", "2024-02-15-preview")
        deployment_name = azure_config.get("deployment_name")

        # Determine model/deployment name
        if deployment_name:
            model = deployment_name
        else:
            # For Azure, we typically use deployment names instead of model names
            model = self.config.llm.default_model
            if model.startswith(("gpt-", "o1-")):
                model = model
            else:
                # Default to gpt-4o deployment if model doesn't look like OpenAI model
                model = "gpt-4o"

        try:
            # Create LangChain Azure OpenAI instance
            return AzureChatOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_config["endpoint"],
                deployment_name=model,  # In Azure, this is the deployment name
                temperature=0.7,
                max_tokens=500,  # Increased for better commit messages
                # LangChain will handle retries and error handling automatically
            )

        except Exception as e:
            raise ApiError(f"Failed to create Azure OpenAI LangChain instance: {e}")

    def __str__(self) -> str:
        """String representation for debugging."""
        deployment = self.config.llm.azure_openai.get("deployment_name", "unknown")
        return f"AzureOpenAiLangChainProvider(deployment={deployment})"