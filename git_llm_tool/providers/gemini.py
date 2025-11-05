"""Google Gemini LLM provider implementation."""

from typing import Optional
import google.generativeai as genai

from git_llm_tool.core.config import AppConfig
from git_llm_tool.core.exceptions import ApiError
from git_llm_tool.providers.base import LlmProvider


class GeminiProvider(LlmProvider):
    """Google Gemini provider implementation."""

    def __init__(self, config: AppConfig):
        """Initialize Gemini provider."""
        super().__init__(config)

        # Get API key
        api_key = config.llm.api_keys.get("google")
        if not api_key:
            raise ApiError("Google API key not found in configuration")

        # Configure Gemini
        genai.configure(api_key=api_key)

        # Determine model
        model = config.llm.default_model
        if not model.startswith("gemini-"):
            # Fallback to Gemini Pro if model doesn't look like Google model
            model = "gemini-1.5-pro"
        self.model = genai.GenerativeModel(model)

    def generate_commit_message(
        self,
        diff: str,
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate commit message using Gemini API."""
        prompt = self._build_commit_prompt(diff, jira_ticket, work_hours)
        return self._make_api_call(prompt, **kwargs)

    def generate_changelog(
        self,
        commit_messages: list[str],
        **kwargs
    ) -> str:
        """Generate changelog using Gemini API."""
        prompt = self._build_changelog_prompt(commit_messages)
        return self._make_api_call(prompt, **kwargs)

    def _make_api_call(self, prompt: str, **kwargs) -> str:
        """Make API call to Gemini."""
        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=kwargs.get("max_tokens", 150),
                temperature=kwargs.get("temperature", 0.7),
            )

            # Make API call
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )

            # Extract response text
            if response.text:
                return response.text.strip()

            raise ApiError("Empty response from Gemini API")

        except Exception as e:
            # Gemini exceptions are not well documented, so catch all
            if "API_KEY" in str(e).upper():
                raise ApiError("Invalid Google API key")
            elif "QUOTA" in str(e).upper() or "RATE" in str(e).upper():
                raise ApiError("Google API rate limit exceeded")
            elif "CONNECTION" in str(e).upper() or "NETWORK" in str(e).upper():
                raise ApiError("Failed to connect to Google API")
            else:
                raise ApiError(f"Google API error: {e}")