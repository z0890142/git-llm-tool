"""Tests for configuration management."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open

from git_llm_tool.core.config import ConfigLoader, AppConfig, LlmConfig, JiraConfig
from git_llm_tool.core.exceptions import ConfigError


class TestConfigLoader:
    """Test ConfigLoader functionality."""

    def setup_method(self):
        """Reset ConfigLoader singleton before each test."""
        ConfigLoader._reset_instance()

    def test_default_config(self):
        """Test default configuration values."""
        with patch('pathlib.Path.exists', return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                loader = ConfigLoader()
                config = loader.config

                assert isinstance(config, AppConfig)
                assert config.llm.default_model == "gpt-4o"
                assert config.llm.language == "en"
                assert config.llm.api_keys == {}
                assert config.jira.enabled is False
                assert config.jira.branch_regex is None

    def test_environment_variables(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            'OPENAI_API_KEY': 'sk-test-openai',
            'ANTHROPIC_API_KEY': 'sk-test-anthropic',
            'GOOGLE_API_KEY': 'test-google',
            'GIT_LLM_MODEL': 'gpt-4-turbo',
            'GIT_LLM_LANGUAGE': 'zh-TW'
        }

        with patch('pathlib.Path.exists', return_value=False):
            with patch.dict(os.environ, env_vars, clear=True):
                loader = ConfigLoader()
                config = loader.config

                assert config.llm.default_model == "gpt-4-turbo"
                assert config.llm.language == "zh-TW"
                assert config.llm.api_keys["openai"] == "sk-test-openai"
                assert config.llm.api_keys["anthropic"] == "sk-test-anthropic"
                assert config.llm.api_keys["google"] == "test-google"

    def test_yaml_config_loading(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
llm:
  default_model: claude-3-sonnet
  language: fr
  api_keys:
    openai: sk-yaml-openai
    anthropic: sk-yaml-anthropic

jira:
  enabled: true
  branch_regex: "feature/(JIRA-\\\\d+)-.*"
"""

        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=yaml_content)):
                with patch.dict(os.environ, {}, clear=True):
                    loader = ConfigLoader()
                    config = loader.config

                    assert config.llm.default_model == "claude-3-sonnet"
                    assert config.llm.language == "fr"
                    assert config.llm.api_keys["openai"] == "sk-yaml-openai"
                    assert config.llm.api_keys["anthropic"] == "sk-yaml-anthropic"
                    assert config.jira.enabled is True
                    assert config.jira.branch_regex == "feature/(JIRA-\\\\d+)-.*"

    def test_config_hierarchy(self):
        """Test configuration hierarchy: env vars override file config."""
        yaml_content = """
llm:
  default_model: gpt-3.5-turbo
  language: en
  api_keys:
    openai: sk-yaml-key

jira:
  enabled: false
"""

        env_vars = {
            'GIT_LLM_MODEL': 'gpt-4-turbo',
            'OPENAI_API_KEY': 'sk-env-key'
        }

        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=yaml_content)):
                with patch.dict(os.environ, env_vars, clear=True):
                    loader = ConfigLoader()
                    config = loader.config

                    # Environment should override file
                    assert config.llm.default_model == "gpt-4-turbo"
                    assert config.llm.api_keys["openai"] == "sk-env-key"
                    # File values should remain
                    assert config.llm.language == "en"
                    assert config.jira.enabled is False

    def test_set_and_get_value(self):
        """Test setting and getting configuration values."""
        with patch('pathlib.Path.exists', return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                loader = ConfigLoader()

                # Test setting values
                loader.set_value("llm.default_model", "claude-3-haiku")
                loader.set_value("llm.language", "ja")
                loader.set_value("llm.api_keys.openai", "sk-new-key")
                loader.set_value("jira.enabled", "true")
                loader.set_value("jira.branch_regex", "test-regex")

                # Test getting values
                assert loader.get_value("llm.default_model") == "claude-3-haiku"
                assert loader.get_value("llm.language") == "ja"
                assert loader.get_value("llm.api_keys.openai") == "sk-new-key"
                assert loader.get_value("jira.enabled") is True
                assert loader.get_value("jira.branch_regex") == "test-regex"

    def test_invalid_key_path(self):
        """Test error handling for invalid key paths."""
        with patch('pathlib.Path.exists', return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                loader = ConfigLoader()

                with pytest.raises(ConfigError, match="Invalid key path"):
                    loader.set_value("invalid", "value")

                with pytest.raises(ConfigError, match="Unknown configuration key"):
                    loader.set_value("unknown.key", "value")

    def test_boolean_conversion(self):
        """Test boolean value conversion for jira.enabled."""
        with patch('pathlib.Path.exists', return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                loader = ConfigLoader()

                # Test various true values
                for true_value in ["true", "True", "1", "yes", "on"]:
                    loader.set_value("jira.enabled", true_value)
                    assert loader.get_value("jira.enabled") is True

                # Test false values
                for false_value in ["false", "False", "0", "no", "off"]:
                    loader.set_value("jira.enabled", false_value)
                    assert loader.get_value("jira.enabled") is False

    def test_invalid_yaml(self):
        """Test error handling for invalid YAML."""
        invalid_yaml = """
llm:
  default_model: gpt-4o
  invalid: yaml: content:
"""

        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=invalid_yaml)):
                with pytest.raises(ConfigError, match="Invalid YAML"):
                    ConfigLoader()

    def test_save_config(self):
        """Test saving configuration to file."""
        with patch('pathlib.Path.exists', return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                loader = ConfigLoader()
                loader.set_value("llm.default_model", "gpt-4-turbo")
                loader.set_value("llm.api_keys.openai", "sk-test")
                loader.set_value("jira.enabled", "true")

                # Mock file operations
                mock_file = mock_open()
                with patch('builtins.open', mock_file):
                    with patch('pathlib.Path.mkdir'):
                        loader.save_config()

                # Verify file was opened for writing
                mock_file.assert_called_once()
                # Verify YAML content was written
                written_content = ''.join(call.args[0] for call in mock_file().write.call_args_list)
                assert 'default_model: gpt-4-turbo' in written_content
                assert 'openai: sk-test' in written_content
                assert 'enabled: true' in written_content

    def test_singleton_behavior(self):
        """Test that ConfigLoader is a singleton."""
        with patch('pathlib.Path.exists', return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                loader1 = ConfigLoader()
                loader2 = ConfigLoader()

                assert loader1 is loader2
                assert loader1.config is loader2.config