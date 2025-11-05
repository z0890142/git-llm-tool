# Git-LLM-Tool

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

AI-powered git commit message and changelog generator using LLM APIs.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [CLI Commands Reference](#cli-commands-reference)
- [Environment Variables](#environment-variables)
- [Usage Examples](#usage-examples)
- [Supported Models](#supported-models)
- [Development](#development)
- [Contributing](#contributing)
- [Git Custom Command Integration](#git-custom-command-integration)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

- 🤖 **Smart Commit Messages**: Automatically generate commit messages from git diff using AI
- 📝 **Changelog Generation**: Generate structured changelogs from git history
- 🔧 **Multiple LLM Providers**: Support for OpenAI, Anthropic Claude, Google Gemini, and Azure OpenAI
- ⚙️ **Hierarchical Configuration**: Project-level and global configuration support
- 🎯 **Jira Integration**: Automatic ticket detection and work hours tracking
- 🌐 **Multi-language Support**: Generate messages in different languages
- ✏️ **Editor Integration**: Configurable editor support for reviewing commit messages
- 🚀 **Easy Setup**: Simple installation and configuration

## Installation

### From PyPI (Coming Soon)
```bash
pip install git-llm-tool
```

### From Source
```bash
git clone https://github.com/your-username/git-llm-tool.git
cd git-llm-tool
poetry install
```

## Quick Start

### 1. Initialize Configuration
```bash
git-llm config init
```

### 2. Configure Your API Key
Choose one of the supported providers:

```bash
# OpenAI
git-llm config set llm.api_keys.openai sk-your-openai-key-here

# Anthropic Claude
git-llm config set llm.api_keys.anthropic sk-ant-your-key-here

# Google Gemini
git-llm config set llm.api_keys.google your-gemini-key-here

# Azure OpenAI
git-llm config set llm.api_keys.azure_openai your-azure-key
git-llm config set llm.azure_openai.endpoint https://your-resource.openai.azure.com/
git-llm config set llm.azure_openai.deployment_name gpt-4o
```

### 3. Generate Commit Messages
```bash
# Stage your changes
git add .

# Generate and review commit message (opens editor)
git-llm commit

# Or apply directly without review
git-llm commit --apply
```

### 4. Generate Changelogs
```bash
# Generate changelog from last tag to HEAD
git-llm changelog

# Generate changelog for specific range
git-llm changelog --from v1.0.0 --to v2.0.0
```

## Configuration

### Configuration Hierarchy

The tool uses a hierarchical configuration system (highest to lowest priority):
1. **CLI flags** (highest priority)
2. **Project config** `.git-llm-tool.yaml`
3. **Global config** `~/.git-llm-tool/config.yaml`
4. **Environment variables**
5. **Default values**

### Configuration Options

#### LLM Settings
```bash
# Set default model
git-llm config set llm.default_model gpt-4o

# Set output language (en, zh, ja, etc.)
git-llm config set llm.language en

# API Keys
git-llm config set llm.api_keys.openai sk-your-key
git-llm config set llm.api_keys.anthropic sk-ant-your-key
git-llm config set llm.api_keys.google your-key

# Azure OpenAI specific settings
git-llm config set llm.azure_openai.endpoint https://your-resource.openai.azure.com/
git-llm config set llm.azure_openai.api_version 2024-12-01-preview
git-llm config set llm.azure_openai.deployment_name gpt-4o
```

#### Editor Configuration
```bash
# Set preferred editor for commit message review
git-llm config set editor.preferred_editor vi
git-llm config set editor.preferred_editor nano
git-llm config set editor.preferred_editor "code --wait"  # VS Code
git-llm config set editor.preferred_editor "subl --wait"  # Sublime Text
```

**Editor Priority (highest to lowest):**
1. `editor.preferred_editor` config
2. `git config core.editor`
3. Environment variables (`GIT_EDITOR`, `VISUAL`, `EDITOR`)
4. System defaults (`nano`, `vim`, `vi`)

#### Jira Integration
```bash
# Enable Jira integration
git-llm config set jira.enabled true

# Set branch regex pattern for ticket extraction
git-llm config set jira.branch_regex '^(feat|fix|chore)\/([A-Z]+-\d+)\/.+$'
```

### Example Configuration File

Global config (`~/.git-llm-tool/config.yaml`):
```yaml
llm:
  default_model: 'gpt-4o'
  language: 'en'
  api_keys:
    openai: 'sk-your-openai-key'
    anthropic: 'sk-ant-your-key'
    google: 'your-gemini-key'
  azure_openai:
    endpoint: 'https://your-resource.openai.azure.com/'
    api_version: '2024-12-01-preview'
    deployment_name: 'gpt-4o'

editor:
  preferred_editor: 'vi'

jira:
  enabled: true
  branch_regex: '^(feat|fix|chore)\/([A-Z]+-\d+)\/.+$'
```

### View Configuration
```bash
# View all configuration
git-llm config get

# View specific setting
git-llm config get llm.default_model
git-llm config get editor.preferred_editor
```

## CLI Commands Reference

### Commit Command
```bash
git-llm commit [OPTIONS]

Options:
  -a, --apply          Apply commit message directly without opening editor
  -m, --model TEXT     Override LLM model (e.g., gpt-4, claude-3-sonnet)
  -l, --language TEXT  Override output language (e.g., en, zh, ja)
  -v, --verbose        Enable verbose output
  --help               Show help message
```

### Changelog Command
```bash
git-llm changelog [OPTIONS]

Options:
  --from TEXT     Starting reference (default: last tag)
  --to TEXT       Ending reference (default: HEAD)
  -o, --output TEXT  Output file (default: stdout)
  -f, --force        Force overwrite existing output file
  --help             Show help message
```

### Config Commands
```bash
git-llm config init                    # Initialize configuration
git-llm config get [KEY]              # Get configuration value(s)
git-llm config set KEY VALUE          # Set configuration value
```

## Environment Variables

You can also configure the tool using environment variables:

```bash
# LLM API Keys
export OPENAI_API_KEY="sk-your-openai-key"
export ANTHROPIC_API_KEY="sk-ant-your-key"
export GOOGLE_API_KEY="your-gemini-key"

# Azure OpenAI
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o"

# Override default model
export GIT_LLM_MODEL="gpt-4o"
export GIT_LLM_LANGUAGE="en"
```

## Usage Examples

### Basic Workflow
```bash
# 1. Make changes to your code
echo "console.log('Hello World');" > app.js

# 2. Stage changes
git add app.js

# 3. Generate commit message with review
git-llm commit
# Opens your editor with AI-generated message for review

# 4. Or apply directly
git-llm commit --apply
```

### Using Different Models
```bash
# Use specific model for this commit
git-llm commit --model claude-3-sonnet

# Use different language
git-llm commit --language zh
```

### Project-specific Configuration
Create `.git-llm-tool.yaml` in your project root:
```yaml
llm:
  default_model: 'claude-3-sonnet'
  language: 'zh'
editor:
  preferred_editor: 'code --wait'
jira:
  enabled: true
  branch_regex: '^(feat|fix|docs)\/([A-Z]+-\d+)\/.+$'
```

## Supported Models

### OpenAI
- `gpt-4o` (recommended)
- `gpt-4o-mini`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

### Anthropic Claude
- `claude-3-5-sonnet-20241022` (recommended)
- `claude-3-5-haiku-20241022`
- `claude-3-opus-20240229`

### Google Gemini
- `gemini-1.5-pro`
- `gemini-1.5-flash`

### Azure OpenAI
- Any deployment of the above OpenAI models

## Development

### Setup Development Environment
```bash
# Clone repository
git clone https://github.com/your-username/git-llm-tool.git
cd git-llm-tool

# Install dependencies
poetry install

# Install pre-commit hooks
poetry run pre-commit install
```

### Running Tests
```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=git_llm_tool

# Run specific test file
poetry run pytest tests/test_config.py
```

### Code Formatting
```bash
# Format code
poetry run black .
poetry run isort .

# Check formatting
poetry run black --check .
poetry run flake8 .
```

### Building and Publishing
```bash
# Build package
poetry build

# Publish to PyPI (maintainers only)
poetry publish
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`poetry run pytest`)
6. Format code (`poetry run black . && poetry run isort .`)
7. Commit your changes (`git-llm commit` 😉)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Git Custom Command Integration

You can integrate git-llm as a native git subcommand, allowing you to use `git llm` instead of `git-llm`.

### Method 1: Git Aliases (Recommended)

Add aliases to your git configuration:

```bash
# Add git aliases for all commands
git config --global alias.llm-commit '!git-llm commit'
git config --global alias.llm-changelog '!git-llm changelog'
git config --global alias.llm-config '!git-llm config'

# Or create a general alias
git config --global alias.llm '!git-llm'
```

Now you can use:
```bash
git llm commit              # Instead of git-llm commit
git llm changelog           # Instead of git-llm changelog
git llm config get          # Instead of git-llm config get

# Or with specific aliases
git llm-commit              # Direct alias to git-llm commit
git llm-changelog           # Direct alias to git-llm changelog
```

### Method 2: Shell Aliases

Add to your shell profile (`.bashrc`, `.zshrc`, etc.):

```bash
# Simple alias
alias gllm='git-llm'

# Or git-style aliases
alias gllmc='git-llm commit'
alias gllmcl='git-llm changelog'
alias gllmcfg='git-llm config'
```

Usage:
```bash
gllm commit                 # git-llm commit
gllmc                       # git-llm commit
gllmcl                      # git-llm changelog
```

### Method 3: Custom Git Script

Create a custom git command script:

```bash
# Create git-llm script in your PATH
sudo tee /usr/local/bin/git-llm > /dev/null << 'EOF'
#!/bin/bash
# Git-LLM integration script
exec git-llm "$@"
EOF

sudo chmod +x /usr/local/bin/git-llm
```

Now you can use:
```bash
git llm commit              # Calls git-llm commit
git llm changelog           # Calls git-llm changelog
```

### Recommended Git Workflow

With git aliases configured, your workflow becomes:

```bash
# Make changes
echo "console.log('Hello');" > app.js

# Stage changes
git add .

# Generate AI commit message (opens editor)
git llm commit

# Or commit directly
git llm commit --apply

# Generate changelog
git llm changelog

# Check configuration
git llm config get
```

## Requirements

- Python 3.12+
- Git
- At least one LLM provider API key

## Troubleshooting

### Common Issues

**"No suitable editor found"**
- Set your preferred editor: `git-llm config set editor.preferred_editor vi`
- Or set git editor: `git config --global core.editor vi`

**"No staged changes found"**
- Stage your changes first: `git add .`

**"API Error: Invalid API key"**
- Check your API key configuration: `git-llm config get`
- Ensure the key is correctly set: `git-llm config set llm.api_keys.openai sk-your-key`

**"No commits found in range"**
- Make sure you have commits in the specified range
- Check git log: `git log --oneline`

## License

MIT License