"""Custom exceptions for git-llm-tool."""


class GitLlmError(Exception):
    """Base exception for git-llm-tool."""
    pass


class ConfigError(GitLlmError):
    """Configuration-related errors."""
    pass


class GitError(GitLlmError):
    """Git operation errors."""
    pass


class ApiError(GitLlmError):
    """LLM API-related errors."""
    pass


class JiraError(GitLlmError):
    """Jira integration errors."""
    pass