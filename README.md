# Git-LLM-Tool

AI-powered git commit message and changelog generator using LLM APIs.

## Features

- 🤖 Automatically generate commit messages from git diff
- 📝 Generate structured changelogs from git history
- 🔧 Support multiple LLM providers (OpenAI, Anthropic, Gemini)
- ⚙️ Hierarchical configuration system
- 🎯 Jira integration for ticket tracking
- 🌐 Multi-language support

## Installation

```bash
pip install git-llm-tool
```

## Quick Start

```bash
# Configure your API key
git-llm config set llm.api_keys.openai sk-your-key-here

# Generate a commit message
git add .
git-llm commit

# Generate a changelog
git-llm changelog --from v1.0.0 --to HEAD
```

## Configuration

The tool uses a hierarchical configuration system:
1. CLI flags (highest priority)
2. Project config `.git-llm-tool.yaml`
3. Global config `~/.git-llm-tool/config.yaml`

Example configuration:
```yaml
llm:
  default_model: 'gpt-4o'
  language: 'en'
  api_keys:
    openai: 'sk-...'
    anthropic: 'sk-...'
    google: '...'

jira:
  enabled: false
  branch_regex: 'feature/(JIRA-\\d+)-.*'
```

## Development

```bash
# Install with poetry
poetry install

# Run tests
poetry run pytest

# Format code
poetry run black .
poetry run isort .
```

## License

MIT License