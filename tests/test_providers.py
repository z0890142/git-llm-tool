"""Tests for LLM providers."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from git_llm_tool.core.config import AppConfig, LlmConfig, JiraConfig
from git_llm_tool.core.exceptions import ApiError
from git_llm_tool.providers.base import LlmProvider
from git_llm_tool.providers.openai import OpenAiProvider
from git_llm_tool.providers.anthropic import AnthropicProvider
from git_llm_tool.providers.gemini import GeminiProvider
from git_llm_tool.providers.factory import get_provider


class TestLlmProviderBase:
    """Test base LLM provider functionality."""

    def test_build_commit_prompt_basic(self):
        """Test building basic commit prompt."""
        config = AppConfig(
            llm=LlmConfig(language="en"),
            jira=JiraConfig()
        )

        class TestProvider(LlmProvider):
            def generate_commit_message(self, diff, **kwargs):
                return ""

            def generate_changelog(self, commit_messages, **kwargs):
                return ""

            def _make_api_call(self, prompt, **kwargs):
                return ""

        provider = TestProvider(config)
        diff = "diff --git a/file.py b/file.py\\n+new line"

        prompt = provider._build_commit_prompt(diff)

        assert "git diff" in prompt
        assert "conventional commit format" in prompt
        assert diff in prompt
        assert "en" in prompt

    def test_build_commit_prompt_with_jira(self):
        """Test building commit prompt with Jira information."""
        config = AppConfig(
            llm=LlmConfig(language="zh-TW"),
            jira=JiraConfig()
        )

        class TestProvider(LlmProvider):
            def generate_commit_message(self, diff, **kwargs):
                return ""

            def generate_changelog(self, commit_messages, **kwargs):
                return ""

            def _make_api_call(self, prompt, **kwargs):
                return ""

        provider = TestProvider(config)
        diff = "test diff"

        prompt = provider._build_commit_prompt(
            diff,
            jira_ticket="PROJ-123",
            work_hours="2h 30m"
        )

        assert "PROJ-123" in prompt
        assert "2h 30m" in prompt
        assert "zh-TW" in prompt

    def test_build_changelog_prompt(self):
        """Test building changelog prompt."""
        config = AppConfig(
            llm=LlmConfig(language="fr"),
            jira=JiraConfig()
        )

        class TestProvider(LlmProvider):
            def generate_commit_message(self, diff, **kwargs):
                return ""

            def generate_changelog(self, commit_messages, **kwargs):
                return ""

            def _make_api_call(self, prompt, **kwargs):
                return ""

        provider = TestProvider(config)
        commits = ["feat: add feature", "fix: bug fix", "docs: update readme"]

        prompt = provider._build_changelog_prompt(commits)

        assert "changelog" in prompt.lower()
        assert "features" in prompt.lower()
        assert "bug fixes" in prompt.lower()
        assert "fr" in prompt
        for commit in commits:
            assert commit in prompt


class TestOpenAiProvider:
    """Test OpenAI provider."""

    def test_init_success(self):
        """Test successful OpenAI provider initialization."""
        config = AppConfig(
            llm=LlmConfig(
                default_model="gpt-4o",
                api_keys={"openai": "sk-test-key"}
            ),
            jira=JiraConfig()
        )

        with patch('openai.OpenAI') as mock_openai:
            provider = OpenAiProvider(config)

            assert provider.model == "gpt-4o"
            mock_openai.assert_called_once_with(api_key="sk-test-key")

    def test_init_no_api_key(self):
        """Test OpenAI provider initialization without API key."""
        config = AppConfig(
            llm=LlmConfig(default_model="gpt-4o", api_keys={}),
            jira=JiraConfig()
        )

        with pytest.raises(ApiError, match="OpenAI API key not found"):
            OpenAiProvider(config)

    def test_model_fallback(self):
        """Test model fallback for non-OpenAI models."""
        config = AppConfig(
            llm=LlmConfig(
                default_model="claude-3-sonnet",
                api_keys={"openai": "sk-test-key"}
            ),
            jira=JiraConfig()
        )

        with patch('openai.OpenAI'):
            provider = OpenAiProvider(config)
            assert provider.model == "gpt-4o"  # fallback

    @patch('openai.OpenAI')
    def test_generate_commit_message_success(self, mock_openai):
        """Test successful commit message generation."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "feat: add new feature"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        config = AppConfig(
            llm=LlmConfig(api_keys={"openai": "sk-test-key"}),
            jira=JiraConfig()
        )

        provider = OpenAiProvider(config)
        result = provider.generate_commit_message("test diff")

        assert result == "feat: add new feature"
        mock_client.chat.completions.create.assert_called_once()

    @patch('openai.OpenAI')
    def test_api_error_handling(self, mock_openai):
        """Test API error handling."""
        import openai

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = openai.AuthenticationError("Invalid API key")
        mock_openai.return_value = mock_client

        config = AppConfig(
            llm=LlmConfig(api_keys={"openai": "sk-test-key"}),
            jira=JiraConfig()
        )

        provider = OpenAiProvider(config)

        with pytest.raises(ApiError, match="Invalid OpenAI API key"):
            provider.generate_commit_message("test diff")


class TestAnthropicProvider:
    """Test Anthropic provider."""

    def test_init_success(self):
        """Test successful Anthropic provider initialization."""
        config = AppConfig(
            llm=LlmConfig(
                default_model="claude-3-5-sonnet-20241024",
                api_keys={"anthropic": "sk-ant-test-key"}
            ),
            jira=JiraConfig()
        )

        with patch('anthropic.Anthropic') as mock_anthropic:
            provider = AnthropicProvider(config)

            assert provider.model == "claude-3-5-sonnet-20241024"
            mock_anthropic.assert_called_once_with(api_key="sk-ant-test-key")

    def test_init_no_api_key(self):
        """Test Anthropic provider initialization without API key."""
        config = AppConfig(
            llm=LlmConfig(default_model="claude-3-sonnet", api_keys={}),
            jira=JiraConfig()
        )

        with pytest.raises(ApiError, match="Anthropic API key not found"):
            AnthropicProvider(config)

    @patch('anthropic.Anthropic')
    def test_generate_commit_message_success(self, mock_anthropic):
        """Test successful commit message generation."""
        # Setup mock response
        mock_content = Mock()
        mock_content.text = "fix: resolve authentication issue"

        mock_response = Mock()
        mock_response.content = [mock_content]

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        config = AppConfig(
            llm=LlmConfig(api_keys={"anthropic": "sk-ant-test-key"}),
            jira=JiraConfig()
        )

        provider = AnthropicProvider(config)
        result = provider.generate_commit_message("test diff")

        assert result == "fix: resolve authentication issue"
        mock_client.messages.create.assert_called_once()


class TestGeminiProvider:
    """Test Gemini provider."""

    def test_init_success(self):
        """Test successful Gemini provider initialization."""
        config = AppConfig(
            llm=LlmConfig(
                default_model="gemini-1.5-pro",
                api_keys={"google": "test-google-key"}
            ),
            jira=JiraConfig()
        )

        with patch('google.generativeai.configure') as mock_configure:
            with patch('google.generativeai.GenerativeModel') as mock_model:
                provider = GeminiProvider(config)

                mock_configure.assert_called_once_with(api_key="test-google-key")
                mock_model.assert_called_once_with("gemini-1.5-pro")

    def test_init_no_api_key(self):
        """Test Gemini provider initialization without API key."""
        config = AppConfig(
            llm=LlmConfig(default_model="gemini-pro", api_keys={}),
            jira=JiraConfig()
        )

        with pytest.raises(ApiError, match="Google API key not found"):
            GeminiProvider(config)

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_generate_commit_message_success(self, mock_model_class, mock_configure):
        """Test successful commit message generation."""
        # Setup mock response
        mock_response = Mock()
        mock_response.text = "chore: update dependencies"

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        config = AppConfig(
            llm=LlmConfig(api_keys={"google": "test-google-key"}),
            jira=JiraConfig()
        )

        provider = GeminiProvider(config)
        result = provider.generate_commit_message("test diff")

        assert result == "chore: update dependencies"
        mock_model.generate_content.assert_called_once()


class TestProviderFactory:
    """Test provider factory."""

    def test_get_openai_provider(self):
        """Test getting OpenAI provider."""
        config = AppConfig(
            llm=LlmConfig(
                default_model="gpt-4o",
                api_keys={"openai": "sk-test-key"}
            ),
            jira=JiraConfig()
        )

        with patch('openai.OpenAI'):
            provider = get_provider(config)
            assert isinstance(provider, OpenAiProvider)

    def test_get_anthropic_provider(self):
        """Test getting Anthropic provider."""
        config = AppConfig(
            llm=LlmConfig(
                default_model="claude-3-sonnet",
                api_keys={"anthropic": "sk-ant-test-key"}
            ),
            jira=JiraConfig()
        )

        with patch('anthropic.Anthropic'):
            provider = get_provider(config)
            assert isinstance(provider, AnthropicProvider)

    def test_get_gemini_provider(self):
        """Test getting Gemini provider."""
        config = AppConfig(
            llm=LlmConfig(
                default_model="gemini-1.5-pro",
                api_keys={"google": "test-google-key"}
            ),
            jira=JiraConfig()
        )

        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                provider = get_provider(config)
                assert isinstance(provider, GeminiProvider)

    def test_fallback_to_available_provider(self):
        """Test fallback to available provider when model doesn't match."""
        config = AppConfig(
            llm=LlmConfig(
                default_model="unknown-model",
                api_keys={"openai": "sk-test-key"}
            ),
            jira=JiraConfig()
        )

        with patch('openai.OpenAI'):
            provider = get_provider(config)
            assert isinstance(provider, OpenAiProvider)

    def test_no_api_keys_error(self):
        """Test error when no API keys are configured."""
        config = AppConfig(
            llm=LlmConfig(default_model="gpt-4o", api_keys={}),
            jira=JiraConfig()
        )

        with pytest.raises(ApiError, match="No API keys configured"):
            get_provider(config)

    def test_missing_required_api_key(self):
        """Test error when required API key is missing."""
        config = AppConfig(
            llm=LlmConfig(
                default_model="gpt-4o",
                api_keys={"anthropic": "sk-ant-key"}  # OpenAI key missing
            ),
            jira=JiraConfig()
        )

        with pytest.raises(ApiError, match="OpenAI API key required"):
            get_provider(config)