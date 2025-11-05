"""Anthropic Claude LLM provider implementation."""

from typing import Optional
import anthropic

from git_llm_tool.core.config import AppConfig
from git_llm_tool.core.exceptions import ApiError
from git_llm_tool.providers.base import LlmProvider


class AnthropicProvider(LlmProvider):
    """Anthropic Claude provider implementation."""

    def __init__(self, config: AppConfig):
        """Initialize Anthropic provider."""
        super().__init__(config)

        # Get API key
        api_key = config.llm.api_keys.get("anthropic")
        if not api_key:
            raise ApiError("Anthropic API key not found in configuration")

        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=api_key)

        # Determine model
        model = config.llm.default_model
        if not model.startswith("claude-"):
            # Fallback to Claude 3.5 Sonnet if model doesn't look like Anthropic model
            model = "claude-3-5-sonnet-20241024"
        self.model = model

    def generate_commit_message(
        self,
        diff: str,
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate commit message using Anthropic API."""
        prompt = self._build_commit_prompt(diff, jira_ticket, work_hours)
        return self._make_api_call(prompt, **kwargs)

    def generate_changelog(
        self,
        commit_messages: list[str],
        **kwargs
    ) -> str:
        """Generate changelog using Anthropic API."""
        prompt = self._build_changelog_prompt(commit_messages)
        return self._make_api_call(prompt, **kwargs)

    def _make_api_call(self, prompt: str, **kwargs) -> str:
        """Make API call to Anthropic."""
        try:
            # Default parameters
            api_params = {
                "model": self.model,
                "max_tokens": kwargs.get("max_tokens", 150),
                "temperature": kwargs.get("temperature", 0.7),
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }

            # Make API call
            response = self.client.messages.create(**api_params)

            # Extract response text
            if response.content and len(response.content) > 0:
                # Anthropic returns a list of content blocks
                content = response.content[0]
                if hasattr(content, 'text'):
                    return content.text.strip()

            raise ApiError("Empty response from Anthropic API")

        except anthropic.AuthenticationError:
            raise ApiError("Invalid Anthropic API key")
        except anthropic.RateLimitError:
            raise ApiError("Anthropic API rate limit exceeded")
        except anthropic.APIConnectionError:
            raise ApiError("Failed to connect to Anthropic API")
        except anthropic.APIError as e:
            raise ApiError(f"Anthropic API error: {e}")
        except Exception as e:
            raise ApiError(f"Unexpected error calling Anthropic API: {e}")