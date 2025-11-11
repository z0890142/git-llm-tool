"""Configuration management for git-llm-tool."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from git_llm_tool.core.exceptions import ConfigError


@dataclass
class LlmConfig:
    """LLM configuration settings."""
    default_model: str = "gpt-4o"
    language: str = "en"
    api_keys: Dict[str, str] = field(default_factory=dict)
    azure_openai: Dict[str, str] = field(default_factory=dict)  # endpoint, api_version, deployment_name

    # Processing configuration
    chunking_threshold: int = 12000  # Token threshold to trigger chunking + parallel processing

    # Ollama configuration for hybrid processing
    use_ollama_for_chunks: bool = True  # Use Ollama for chunk processing (map phase)
    ollama_model: str = "phi3:mini"  # Ollama model for chunk processing
    ollama_base_url: str = "http://localhost:11434"  # Ollama API base URL

    # Parallel processing configuration
    max_parallel_chunks: int = 4  # Maximum concurrent chunks for remote APIs
    ollama_max_parallel_chunks: int = 16  # Maximum concurrent chunks for Ollama (local)

    # Chunking configuration
    chunk_size: int = 2048  # Maximum chunk size in tokens
    chunk_overlap: int = 150  # Overlap between chunks in tokens

    # Internal constants (not user-configurable)
    _chunk_size: int = 2048  # Maximum chunk size in characters
    _chunk_overlap: int = 300  # Overlap between chunks to maintain context
    _max_parallel_chunks: int = 4  # Maximum number of chunks to process in parallel (remote APIs)
    _ollama_max_parallel_chunks: int = 4  # Maximum number of chunks to process in parallel (Ollama local)
    _chunk_processing_timeout: float = 120.0  # Timeout for each chunk processing (seconds)
    _max_retries: int = 5  # Maximum number of retries
    _initial_delay: float = 1.0  # Initial retry delay in seconds
    _max_delay: float = 60.0  # Maximum retry delay in seconds
    _backoff_multiplier: float = 2.0  # Exponential backoff multiplier
    _rate_limit_delay: float = 0.5  # Minimum delay between requests
    _max_context_lines: int = 3  # Maximum context lines to keep
    _max_tokens: int = 8000  # Maximum tokens before truncation


@dataclass
class JiraConfig:
    """Jira integration configuration."""
    enabled: bool = False
    ticket_pattern: Optional[str] = None  # Jira ticket regex pattern


@dataclass
class EditorConfig:
    """Editor configuration settings."""
    preferred_editor: Optional[str] = None  # e.g., "vi", "nano", "code", etc.


@dataclass
class AppConfig:
    """Main application configuration."""
    llm: LlmConfig = field(default_factory=LlmConfig)
    jira: JiraConfig = field(default_factory=JiraConfig)
    editor: EditorConfig = field(default_factory=EditorConfig)


class ConfigLoader:
    """Singleton configuration loader with hierarchical configuration support."""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not getattr(self, '_initialized', False):
            self._config = self._load_config()
            self._initialized = True

    @property
    def config(self) -> AppConfig:
        """Get the loaded configuration."""
        return self._config

    def _load_config(self) -> AppConfig:
        """Load configuration from multiple sources in hierarchical order."""
        config_data = {}

        # 1. Load global config
        global_config_path = Path.home() / ".git-llm-tool" / "config.yaml"
        if global_config_path.exists():
            config_data.update(self._load_yaml_file(global_config_path))

        # 2. Load project config (override global)
        project_config_path = Path(".git-llm-tool.yaml")
        if project_config_path.exists():
            project_config = self._load_yaml_file(project_config_path)
            config_data = self._merge_configs(config_data, project_config)

        # 3. Load environment variables (override file configs)
        env_config = self._load_env_config()
        config_data = self._merge_configs(config_data, env_config)

        return self._create_app_config(config_data)

    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data if data is not None else {}
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {file_path}: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to read config file {file_path}: {e}")

    def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}

        # API keys from environment
        api_keys = {}
        if openai_key := os.getenv("OPENAI_API_KEY"):
            api_keys["openai"] = openai_key
        if anthropic_key := os.getenv("ANTHROPIC_API_KEY"):
            api_keys["anthropic"] = anthropic_key
        if google_key := os.getenv("GOOGLE_API_KEY"):
            api_keys["google"] = google_key

        # Azure OpenAI configuration from environment
        azure_openai = {}
        if azure_endpoint := os.getenv("AZURE_OPENAI_ENDPOINT"):
            azure_openai["endpoint"] = azure_endpoint
        if azure_key := os.getenv("AZURE_OPENAI_API_KEY"):
            api_keys["azure_openai"] = azure_key
        if azure_version := os.getenv("AZURE_OPENAI_API_VERSION"):
            azure_openai["api_version"] = azure_version
        if azure_deployment := os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"):
            azure_openai["deployment_name"] = azure_deployment

        # Set up LLM config
        if api_keys or azure_openai:
            config["llm"] = {}
            if api_keys:
                config["llm"]["api_keys"] = api_keys
            if azure_openai:
                config["llm"]["azure_openai"] = azure_openai

        # Other environment variables
        if model := os.getenv("GIT_LLM_MODEL"):
            config.setdefault("llm", {})["default_model"] = model

        if language := os.getenv("GIT_LLM_LANGUAGE"):
            config.setdefault("llm", {})["language"] = language

        return config

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries recursively."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _create_app_config(self, config_data: Dict[str, Any]) -> AppConfig:
        """Create AppConfig instance from configuration data."""
        # Create LLM config
        llm_data = config_data.get("llm", {})
        llm_config = LlmConfig(
            default_model=llm_data.get("default_model", "gpt-4o"),
            language=llm_data.get("language", "en"),
            api_keys=llm_data.get("api_keys", {}),
            azure_openai=llm_data.get("azure_openai", {}),
            # Processing settings
            chunking_threshold=llm_data.get("chunking_threshold", 12000),
            # Ollama settings
            use_ollama_for_chunks=llm_data.get("use_ollama_for_chunks", True),
            ollama_model=llm_data.get("ollama_model", "phi3:mini"),
            ollama_base_url=llm_data.get("ollama_base_url", "http://localhost:11434"),
            # Parallel processing settings
            max_parallel_chunks=llm_data.get("max_parallel_chunks", 4),
            ollama_max_parallel_chunks=llm_data.get("ollama_max_parallel_chunks", 16),
            # Chunking settings
            chunk_size=llm_data.get("chunk_size", 6000),
            chunk_overlap=llm_data.get("chunk_overlap", 300)
        )

        # Create Jira config
        jira_data = config_data.get("jira", {})
        jira_config = JiraConfig(
            enabled=jira_data.get("enabled", False),
            ticket_pattern=jira_data.get("ticket_pattern")
        )

        # Create Editor config
        editor_data = config_data.get("editor", {})
        editor_config = EditorConfig(
            preferred_editor=editor_data.get("preferred_editor")
        )

        return AppConfig(llm=llm_config, jira=jira_config, editor=editor_config)

    def save_config(self, config_path: Optional[Path] = None) -> None:
        """Save current configuration to file."""
        if config_path is None:
            # Save to global config by default
            config_path = Path.home() / ".git-llm-tool" / "config.yaml"

        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert config to dict
        config_dict = {
            "llm": {
                "default_model": self._config.llm.default_model,
                "language": self._config.llm.language,
                "api_keys": self._config.llm.api_keys,
                "azure_openai": self._config.llm.azure_openai,
                "chunking_threshold": self._config.llm.chunking_threshold,
                "use_ollama_for_chunks": self._config.llm.use_ollama_for_chunks,
                "ollama_model": self._config.llm.ollama_model,
                "ollama_base_url": self._config.llm.ollama_base_url
            },
            "jira": {
                "enabled": self._config.jira.enabled,
                "ticket_pattern": self._config.jira.ticket_pattern
            },
            "editor": {
                "preferred_editor": self._config.editor.preferred_editor
            }
        }

        # Remove empty sections to keep config clean
        if not config_dict["llm"]["api_keys"]:
            del config_dict["llm"]["api_keys"]
        if not config_dict["llm"]["azure_openai"]:
            del config_dict["llm"]["azure_openai"]

        # Remove None values from jira config
        if config_dict["jira"]["ticket_pattern"] is None:
            del config_dict["jira"]["ticket_pattern"]

        # Remove None values from editor config
        if config_dict["editor"]["preferred_editor"] is None:
            del config_dict["editor"]["preferred_editor"]
        if not config_dict["editor"]:
            del config_dict["editor"]

        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ConfigError(f"Failed to save config to {config_path}: {e}")

    def set_value(self, key_path: str, value: str) -> None:
        """Set a configuration value using dot notation (e.g., 'llm.default_model')."""
        keys = key_path.split('.')

        if len(keys) < 2:
            raise ConfigError(f"Invalid key path: {key_path}")

        # Handle llm.default_model
        if keys[0] == "llm" and keys[1] == "default_model":
            self._config.llm.default_model = value
        # Handle llm.language
        elif keys[0] == "llm" and keys[1] == "language":
            self._config.llm.language = value
        # Handle llm.use_ollama_for_chunks
        elif keys[0] == "llm" and keys[1] == "use_ollama_for_chunks":
            self._config.llm.use_ollama_for_chunks = value.lower() in ("true", "1", "yes", "on")
        # Handle llm.ollama_model
        elif keys[0] == "llm" and keys[1] == "ollama_model":
            self._config.llm.ollama_model = value
        # Handle llm.ollama_base_url
        elif keys[0] == "llm" and keys[1] == "ollama_base_url":
            self._config.llm.ollama_base_url = value
        # Handle llm.api_keys.*
        elif keys[0] == "llm" and keys[1] == "api_keys" and len(keys) == 3:
            self._config.llm.api_keys[keys[2]] = value
        # Handle llm.azure_openai.*
        elif keys[0] == "llm" and keys[1] == "azure_openai" and len(keys) == 3:
            self._config.llm.azure_openai[keys[2]] = value
        # Handle jira.enabled
        elif keys[0] == "jira" and keys[1] == "enabled":
            self._config.jira.enabled = value.lower() in ("true", "1", "yes", "on")
        # Handle jira.ticket_pattern
        elif keys[0] == "jira" and keys[1] == "ticket_pattern":
            self._config.jira.ticket_pattern = value
        # Handle editor.preferred_editor
        elif keys[0] == "editor" and keys[1] == "preferred_editor":
            self._config.editor.preferred_editor = value
        else:
            raise ConfigError(f"Unknown configuration key: {key_path}")

    def get_value(self, key_path: str) -> Any:
        """Get a configuration value using dot notation."""
        keys = key_path.split('.')

        if len(keys) < 2:
            raise ConfigError(f"Invalid key path: {key_path}")

        # Handle llm.default_model
        if keys[0] == "llm" and keys[1] == "default_model":
            return self._config.llm.default_model
        # Handle llm.language
        elif keys[0] == "llm" and keys[1] == "language":
            return self._config.llm.language
        # Handle llm.use_ollama_for_chunks
        elif keys[0] == "llm" and keys[1] == "use_ollama_for_chunks":
            return self._config.llm.use_ollama_for_chunks
        # Handle llm.ollama_model
        elif keys[0] == "llm" and keys[1] == "ollama_model":
            return self._config.llm.ollama_model
        # Handle llm.ollama_base_url
        elif keys[0] == "llm" and keys[1] == "ollama_base_url":
            return self._config.llm.ollama_base_url
        # Handle llm.api_keys.*
        elif keys[0] == "llm" and keys[1] == "api_keys" and len(keys) == 3:
            return self._config.llm.api_keys.get(keys[2])
        # Handle llm.azure_openai.*
        elif keys[0] == "llm" and keys[1] == "azure_openai" and len(keys) == 3:
            return self._config.llm.azure_openai.get(keys[2])
        # Handle jira.enabled
        elif keys[0] == "jira" and keys[1] == "enabled":
            return self._config.jira.enabled
        # Handle jira.ticket_pattern
        elif keys[0] == "jira" and keys[1] == "ticket_pattern":
            return self._config.jira.ticket_pattern
        # Handle editor.preferred_editor
        elif keys[0] == "editor" and keys[1] == "preferred_editor":
            return self._config.editor.preferred_editor
        else:
            raise ConfigError(f"Unknown configuration key: {key_path}")

    def reload(self) -> None:
        """Reload configuration from files."""
        self._config = self._load_config()

    @classmethod
    def _reset_instance(cls) -> None:
        """Reset singleton instance for testing."""
        cls._instance = None
        cls._config = None


def get_config() -> AppConfig:
    """Get the application configuration."""
    return ConfigLoader().config