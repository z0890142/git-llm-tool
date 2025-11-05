"""LLM Provider factory for automatic provider selection."""

from git_llm_tool.core.config import AppConfig
from git_llm_tool.core.exceptions import ApiError
from git_llm_tool.providers.base import LlmProvider
from git_llm_tool.providers.openai import OpenAiProvider
from git_llm_tool.providers.azure_openai import AzureOpenAiProvider
from git_llm_tool.providers.anthropic import AnthropicProvider
from git_llm_tool.providers.gemini import GeminiProvider


def get_provider(config: AppConfig) -> LlmProvider:
    """Get appropriate LLM provider based on model name in config.

    Args:
        config: Application configuration

    Returns:
        Initialized LLM provider

    Raises:
        ApiError: If no suitable provider is found or API key is missing
    """
    model = config.llm.default_model.lower()

    # Check if Azure OpenAI is configured (highest priority for OpenAI-compatible models)
    if config.llm.azure_openai and config.llm.azure_openai.get("endpoint"):
        if model.startswith(("gpt-", "o1-")) or "azure" in model:
            if "azure_openai" not in config.llm.api_keys:
                raise ApiError("Azure OpenAI API key required for Azure OpenAI models")
            return AzureOpenAiProvider(config)

    # OpenAI models (regular OpenAI API)
    if model.startswith(("gpt-", "o1-")):
        if "openai" not in config.llm.api_keys:
            raise ApiError("OpenAI API key required for GPT models")
        return OpenAiProvider(config)

    # Anthropic models
    elif model.startswith("claude-"):
        if "anthropic" not in config.llm.api_keys:
            raise ApiError("Anthropic API key required for Claude models")
        return AnthropicProvider(config)

    # Google models
    elif model.startswith("gemini-"):
        if "google" not in config.llm.api_keys:
            raise ApiError("Google API key required for Gemini models")
        return GeminiProvider(config)

    # Fallback logic - try providers in order of preference
    else:
        # Try Azure OpenAI first if configured
        if config.llm.azure_openai and config.llm.azure_openai.get("endpoint") and "azure_openai" in config.llm.api_keys:
            return AzureOpenAiProvider(config)

        # Try OpenAI second (most common)
        elif "openai" in config.llm.api_keys:
            return OpenAiProvider(config)

        # Try Anthropic third
        elif "anthropic" in config.llm.api_keys:
            return AnthropicProvider(config)

        # Try Google last
        elif "google" in config.llm.api_keys:
            return GeminiProvider(config)

        # No API keys available
        else:
            raise ApiError(
                "No API keys configured. Please set at least one API key:\n"
                "  git-llm config set llm.api_keys.openai sk-your-key\n"
                "  git-llm config set llm.api_keys.azure_openai your-azure-key\n"
                "  git-llm config set llm.api_keys.anthropic sk-ant-your-key\n"
                "  git-llm config set llm.api_keys.google your-google-key"
            )