"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from git_llm_tool.core.config import AppConfig


class LlmProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: AppConfig):
        """Initialize the provider with configuration."""
        self.config = config

    @abstractmethod
    def generate_commit_message(
        self,
        diff: str,
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate commit message from git diff and optional Jira information.

        Args:
            diff: Git diff output
            jira_ticket: Jira ticket number (optional)
            work_hours: Work hours spent (optional)
            **kwargs: Additional provider-specific arguments

        Returns:
            Generated commit message

        Raises:
            ApiError: If API call fails
        """
        pass

    @abstractmethod
    def generate_changelog(self, commit_messages: list[str], **kwargs) -> str:
        """Generate changelog from commit messages.

        Args:
            commit_messages: List of commit messages
            **kwargs: Additional provider-specific arguments

        Returns:
            Generated changelog in markdown format

        Raises:
            ApiError: If API call fails
        """
        pass

    def _build_commit_prompt(
        self,
        diff: str,
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None,
    ) -> str:
        """Build prompt for commit message generation."""

        # If we have Jira ticket, use Jira format, otherwise use conventional commit format
        if jira_ticket:
            base_prompt = f"""Based on the following git diff, generate a concise commit message in {self.config.llm.language} using the JIRA format:
1. Use **JIRA format**:
    <ISSUE_KEY> <summary_comment> #time <time_spent>

    Where:
        - <ISSUE_KEY>: {jira_ticket}
        - <summary_comment>: concise description
        - #time: time tracking (e.g., #time 2h, #time 45m)

2. Also include **Conventional Commit format**:
    type: description where type can be:
    - feat: new feature
    - fix: bug fix
    - docs: documentation changes
    - style: formatting, missing semicolons, etc
    - refactor: code restructuring without changing functionality
    - test: adding or modifying tests
    - chore: maintenance tasks

Git diff:
```
{diff}
```

JIRA ticket: {jira_ticket}"""

            if work_hours:
                base_prompt += f"\nTime spent: {work_hours}"
                base_prompt += f"\n\nGenerate ONLY the commit message in this exact format: {jira_ticket} <description> #time {work_hours}"
            else:
                base_prompt += (
                    "\nTime spent: (estimate appropriate time based on the changes)"
                )
                base_prompt += f"\n\nGenerate ONLY the commit message in this format: {jira_ticket} <description> #time <estimated_time>"
        else:
            # Use conventional commit format when no Jira ticket
            base_prompt = f"""Based on the following git diff, generate a concise and descriptive commit message in {self.config.llm.language}.

Follow conventional commit format (type: description) where type can be:
- feat: new feature
- fix: bug fix
- docs: documentation changes
- style: formatting, missing semicolons, etc
- refactor: code restructuring without changing functionality
- test: adding or modifying tests
- chore: maintenance tasks

Git diff:
```
{diff}
```

Generate ONLY the commit message, no additional text or explanation."""

        return base_prompt

    def _build_changelog_prompt(self, commit_messages: list[str]) -> str:
        """Build prompt for changelog generation."""
        commits_text = "\n".join([f"- {msg}" for msg in commit_messages])

        return f"""Generate a structured changelog in {self.config.llm.language} from the following commit messages.

Organize by categories:
✨ **Features**
🐛 **Bug Fixes**
📚 **Documentation**
🎨 **Style**
♻️ **Refactoring**
🧪 **Tests**
🔧 **Chores**
💥 **Breaking Changes**

Only include categories that have items. Use markdown format.

Commit messages:
{commits_text}

Generate the changelog:"""

    @abstractmethod
    def _make_api_call(self, prompt: str, **kwargs) -> str:
        """Make API call to the LLM provider.

        Args:
            prompt: The prompt to send
            **kwargs: Provider-specific arguments

        Returns:
            Generated text response

        Raises:
            ApiError: If API call fails
        """
        pass
