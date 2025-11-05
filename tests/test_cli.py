"""Tests for CLI interface."""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, Mock

from git_llm_tool.cli import main
from git_llm_tool.core.exceptions import ConfigError, GitError, ApiError


class TestCLI:
    """Test CLI functionality."""

    def test_main_help(self):
        """Test main command help."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])

        assert result.exit_code == 0
        assert "AI-powered git commit message and changelog generator" in result.output
        assert "commit" in result.output
        assert "config" in result.output
        assert "changelog" in result.output

    def test_version(self):
        """Test version command."""
        runner = CliRunner()
        result = runner.invoke(main, ['--version'])

        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_commit_help(self):
        """Test commit command help."""
        runner = CliRunner()
        result = runner.invoke(main, ['commit', '--help'])

        assert result.exit_code == 0
        assert "Generate AI-powered commit message" in result.output
        assert "--apply" in result.output
        assert "--model" in result.output
        assert "--language" in result.output

    def test_changelog_help(self):
        """Test changelog command help."""
        runner = CliRunner()
        result = runner.invoke(main, ['changelog', '--help'])

        assert result.exit_code == 0
        assert "Generate changelog from git history" in result.output
        assert "--from" in result.output
        assert "--to" in result.output
        assert "--output" in result.output

    def test_config_help(self):
        """Test config command help."""
        runner = CliRunner()
        result = runner.invoke(main, ['config', '--help'])

        assert result.exit_code == 0
        assert "Configuration management" in result.output
        assert "get" in result.output
        assert "set" in result.output
        assert "init" in result.output


class TestConfigCommands:
    """Test configuration commands."""

    @patch('git_llm_tool.cli.ConfigLoader')
    def test_config_init_success(self, mock_loader):
        """Test successful config initialization."""
        mock_instance = Mock()
        mock_loader.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(main, ['config', 'init'])

        assert result.exit_code == 0
        assert "Configuration initialized" in result.output
        mock_instance.save_config.assert_called_once()

    @patch('git_llm_tool.cli.ConfigLoader')
    @patch('pathlib.Path.exists', return_value=True)
    def test_config_init_existing_file_confirm_yes(self, mock_exists, mock_loader):
        """Test config initialization with existing file - user confirms."""
        mock_instance = Mock()
        mock_loader.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(main, ['config', 'init'], input='y\\n')

        assert result.exit_code == 0
        assert "Configuration initialized" in result.output
        mock_instance.save_config.assert_called_once()

    @patch('git_llm_tool.cli.ConfigLoader')
    @patch('pathlib.Path.exists', return_value=True)
    def test_config_init_existing_file_confirm_no(self, mock_exists, mock_loader):
        """Test config initialization with existing file - user cancels."""
        runner = CliRunner()
        result = runner.invoke(main, ['config', 'init'], input='n\\n')

        assert result.exit_code == 0
        assert "Initialization cancelled" in result.output

    @patch('git_llm_tool.cli.ConfigLoader')
    def test_config_set_success(self, mock_loader):
        """Test successful config set."""
        mock_instance = Mock()
        mock_loader.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(main, ['config', 'set', 'llm.default_model', 'gpt-4-turbo'])

        assert result.exit_code == 0
        assert "Set llm.default_model = gpt-4-turbo" in result.output
        mock_instance.set_value.assert_called_once_with('llm.default_model', 'gpt-4-turbo')
        mock_instance.save_config.assert_called_once()

    @patch('git_llm_tool.cli.ConfigLoader')
    def test_config_set_error(self, mock_loader):
        """Test config set with error."""
        mock_instance = Mock()
        mock_instance.set_value.side_effect = ConfigError("Invalid key")
        mock_loader.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(main, ['config', 'set', 'invalid.key', 'value'])

        assert result.exit_code == 0
        assert "Configuration error: Invalid key" in result.output

    @patch('git_llm_tool.cli.get_config')
    def test_config_get_all(self, mock_get_config):
        """Test getting all configuration."""
        from git_llm_tool.core.config import AppConfig, LlmConfig, JiraConfig

        mock_config = AppConfig(
            llm=LlmConfig(
                default_model="gpt-4o",
                language="en",
                api_keys={"openai": "sk-1234567890abcdef"}
            ),
            jira=JiraConfig(enabled=False)
        )
        mock_get_config.return_value = mock_config

        runner = CliRunner()
        result = runner.invoke(main, ['config', 'get'])

        assert result.exit_code == 0
        assert "Current Configuration" in result.output
        assert "llm.default_model = gpt-4o" in result.output
        assert "llm.language = en" in result.output
        assert "openai = sk-12345..." in result.output  # masked
        assert "jira.enabled = False" in result.output

    @patch('git_llm_tool.cli.ConfigLoader')
    def test_config_get_specific_key(self, mock_loader):
        """Test getting specific configuration key."""
        mock_instance = Mock()
        mock_instance.get_value.return_value = "gpt-4-turbo"
        mock_loader.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(main, ['config', 'get', 'llm.default_model'])

        assert result.exit_code == 0
        assert "llm.default_model = gpt-4-turbo" in result.output
        mock_instance.get_value.assert_called_once_with('llm.default_model')


class TestCommitCommand:
    """Test commit command."""

    @patch('git_llm_tool.commands.commit_cmd.execute_commit')
    def test_commit_basic(self, mock_execute):
        """Test basic commit command."""
        runner = CliRunner()
        result = runner.invoke(main, ['commit'])

        assert result.exit_code == 0
        mock_execute.assert_called_once_with(
            apply=False,
            model=None,
            language=None,
            verbose=False
        )

    @patch('git_llm_tool.commands.commit_cmd.execute_commit')
    def test_commit_with_options(self, mock_execute):
        """Test commit command with options."""
        runner = CliRunner()
        result = runner.invoke(main, [
            '--verbose', 'commit',
            '--apply',
            '--model', 'gpt-4-turbo',
            '--language', 'zh-TW'
        ])

        assert result.exit_code == 0
        mock_execute.assert_called_once_with(
            apply=True,
            model='gpt-4-turbo',
            language='zh-TW',
            verbose=True
        )

    @patch('git_llm_tool.commands.commit_cmd.get_config')
    @patch('git_llm_tool.commands.commit_cmd.GitHelper')
    def test_commit_no_staged_changes(self, mock_git_helper, mock_get_config):
        """Test commit command with no staged changes."""
        from git_llm_tool.core.config import AppConfig, LlmConfig, JiraConfig

        mock_config = AppConfig(llm=LlmConfig(), jira=JiraConfig())
        mock_get_config.return_value = mock_config

        mock_helper = Mock()
        mock_helper.get_staged_diff.side_effect = GitError("No staged changes found")
        mock_git_helper.return_value = mock_helper

        runner = CliRunner()
        result = runner.invoke(main, ['commit'])

        assert result.exit_code == 0
        assert "No staged changes found" in result.output
        assert "Use 'git add' to stage files" in result.output

    @patch('git_llm_tool.commands.commit_cmd.get_config')
    @patch('git_llm_tool.commands.commit_cmd.get_provider')
    def test_commit_no_api_key(self, mock_get_provider, mock_get_config):
        """Test commit command with no API key."""
        from git_llm_tool.core.config import AppConfig, LlmConfig, JiraConfig

        mock_config = AppConfig(llm=LlmConfig(), jira=JiraConfig())
        mock_get_config.return_value = mock_config

        mock_get_provider.side_effect = ApiError("No API keys configured")

        runner = CliRunner()
        result = runner.invoke(main, ['commit'])

        assert result.exit_code == 0
        assert "No API keys configured" in result.output


class TestChangelogCommand:
    """Test changelog command."""

    def test_changelog_basic(self):
        """Test basic changelog command."""
        runner = CliRunner()
        result = runner.invoke(main, ['changelog'])

        assert result.exit_code == 0
        assert "Generating changelog" in result.output

    def test_changelog_with_options(self):
        """Test changelog command with options."""
        runner = CliRunner()
        result = runner.invoke(main, [
            'changelog',
            '--from', 'v1.0.0',
            '--to', 'HEAD',
            '--output', 'CHANGELOG.md',
            '--force'
        ])

        assert result.exit_code == 0
        assert "Generating changelog" in result.output