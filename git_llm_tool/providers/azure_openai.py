"""Azure OpenAI LLM provider implementation."""

from typing import Optional
import openai

from git_llm_tool.core.config import AppConfig
from git_llm_tool.core.exceptions import ApiError
from git_llm_tool.providers.base import LlmProvider


class AzureOpenAiProvider(LlmProvider):
    """Azure OpenAI provider implementation."""

    def __init__(self, config: AppConfig):
        """Initialize Azure OpenAI provider."""
        super().__init__(config)

        # Get Azure OpenAI configuration
        azure_config = config.llm.azure_openai
        if not azure_config.get("endpoint"):
            raise ApiError("Azure OpenAI endpoint not found in configuration")

        api_key = config.llm.api_keys.get("azure_openai")
        if not api_key:
            raise ApiError("Azure OpenAI API key not found in configuration")

        # Default values for Azure OpenAI
        api_version = azure_config.get("api_version", "2024-02-15-preview")
        deployment_name = azure_config.get("deployment_name")

        # Initialize Azure OpenAI client
        self.client = openai.AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_config["endpoint"]
        )

        # Use deployment name if provided, otherwise use model name
        if deployment_name:
            self.model = deployment_name
        else:
            # For Azure, we typically use deployment names instead of model names
            model = config.llm.default_model
            if model.startswith(("gpt-", "o1-")):
                self.model = model
            else:
                # Default to gpt-4o deployment if model doesn't look like OpenAI model
                self.model = "gpt-4o"

    def generate_commit_message(
        self,
        diff: str,
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate commit message using Azure OpenAI API."""
        prompt = self._build_commit_prompt(diff, jira_ticket, work_hours)
        return self._make_api_call(prompt, **kwargs)

    def generate_changelog(
        self,
        commit_messages: list[str],
        **kwargs
    ) -> str:
        """Generate changelog using Azure OpenAI API."""
        prompt = self._build_changelog_prompt(commit_messages)
        return self._make_api_call(prompt, **kwargs)

    def _make_api_call(self, prompt: str, **kwargs) -> str:
        """Make API call to Azure OpenAI."""
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

            raise ApiError("Empty response from Azure OpenAI API")

        except openai.AuthenticationError:
            raise ApiError("Invalid Azure OpenAI API key")
        except openai.RateLimitError:
            raise ApiError("Azure OpenAI API rate limit exceeded")
        except openai.APIConnectionError:
            raise ApiError("Failed to connect to Azure OpenAI API")
        except openai.NotFoundError:
            raise ApiError(f"Azure OpenAI deployment '{self.model}' not found. Check your deployment name.")
        except openai.APIError as e:
            raise ApiError(f"Azure OpenAI API error: {e}")
        except Exception as e:
            raise ApiError(f"Unexpected error calling Azure OpenAI API: {e}")