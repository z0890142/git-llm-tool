"""OpenAI LLM provider implementation."""

from typing import Optional
import openai

from git_llm_tool.core.config import AppConfig
from git_llm_tool.core.exceptions import ApiError
from git_llm_tool.providers.base import LlmProvider


class OpenAiProvider(LlmProvider):
    """OpenAI GPT provider implementation."""

    def __init__(self, config: AppConfig):
        """Initialize OpenAI provider."""
        super().__init__(config)

        # Get API key
        api_key = config.llm.api_keys.get("openai")
        if not api_key:
            raise ApiError("OpenAI API key not found in configuration")

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=api_key)

        # Determine model
        model = config.llm.default_model
        if not model.startswith(("gpt-", "o1-")):
            # Fallback to GPT-4o if model doesn't look like OpenAI model
            model = "gpt-4o"
        self.model = model

    def generate_commit_message(
        self,
        diff: str,
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate commit message using OpenAI API."""
        prompt = self._build_commit_prompt(diff, jira_ticket, work_hours)
        return self._make_api_call(prompt, **kwargs)

    def generate_changelog(
        self,
        commit_messages: list[str],
        **kwargs
    ) -> str:
        """Generate changelog using OpenAI API."""
        prompt = self._build_changelog_prompt(commit_messages)
        return self._make_api_call(prompt, **kwargs)

    def _make_api_call(self, prompt: str, **kwargs) -> str:
        """Make API call to OpenAI."""
        try:
            # Default parameters
            api_params = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates git commit messages and changelogs."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": kwargs.get("max_tokens", 150),
                "temperature": kwargs.get("temperature", 0.7),
            }

            # Make API call
            response = self.client.chat.completions.create(**api_params)

            # Extract response text
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    return content.strip()

            raise ApiError("Empty response from OpenAI API")

        except openai.AuthenticationError:
            raise ApiError("Invalid OpenAI API key")
        except openai.RateLimitError:
            raise ApiError("OpenAI API rate limit exceeded")
        except openai.APIConnectionError:
            raise ApiError("Failed to connect to OpenAI API")
        except openai.APIError as e:
            raise ApiError(f"OpenAI API error: {e}")
        except Exception as e:
            raise ApiError(f"Unexpected error calling OpenAI API: {e}")