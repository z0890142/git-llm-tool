"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from git_llm_tool.core.config import AppConfig


class PromptTemplates:
    """Centralized prompt templates for better code readability."""

    # Base prompt with conventional commit types
    BASE_COMMIT_PROMPT = """Based on the following git diff, generate a concise commit message in {language}.

**Conventional Commit types**:
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
```"""

    # Format for commits without Jira tickets
    NO_JIRA_FORMAT = """
Generate the commit message in this **exact format**:
<summary>
- feat: description of new features
- fix: description of bug fixes
- docs: description of documentation changes
(include only the types that apply to your changes)

Example format:
Implement user authentication system
- feat: add user authentication endpoints
- fix: resolve login validation issue"""

    # Format for commits with Jira tickets
    JIRA_FORMAT = """
**Jira ticket found**: {jira_ticket}

Generate the commit message in this **exact format**:
{jira_ticket} <summary> #time <time_spent>
- feat: detailed description of new features
- fix: detailed description of bug fixes
- docs: detailed description of documentation changes
(include only the types that apply to your changes)

Where:
- First line: {jira_ticket} <brief_summary> #time <time_spent>
- Following lines: List each change type with "- type: description" format
- Only include the conventional commit types that actually apply to the changes"""

    # Time tracking instructions
    EXACT_TIME_INSTRUCTION = """
**Use exact time**: #time {work_hours}

Example format:
{jira_ticket} Implement user authentication system #time {work_hours}
- feat: add login and registration endpoints
- feat: implement JWT token validation
- docs: update API documentation"""

    ESTIMATE_TIME_INSTRUCTION = """

Example format:
{jira_ticket} Implement user authentication system
- feat: add login and registration endpoints
- feat: implement JWT token validation
- docs: update API documentation"""

    # Final instruction
    FINAL_INSTRUCTION = "\n\nGenerate ONLY the commit message in the specified format, no additional text or explanation."


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

        # Prepare all template variables
        template_vars = {
            "language": self.config.llm.language,
            "diff": diff,
            "jira_ticket": jira_ticket or "",
            "work_hours": work_hours or "",
        }

        # Build prompt components
        prompt_parts = [PromptTemplates.BASE_COMMIT_PROMPT]

        # Add format-specific instructions
        if jira_ticket:
            prompt_parts.append(PromptTemplates.JIRA_FORMAT)

            # Add time tracking instructions
            if work_hours:
                prompt_parts.append(PromptTemplates.EXACT_TIME_INSTRUCTION)
            else:
                prompt_parts.append(PromptTemplates.ESTIMATE_TIME_INSTRUCTION)
        else:
            prompt_parts.append(PromptTemplates.NO_JIRA_FORMAT)

        # Add final instruction
        prompt_parts.append(PromptTemplates.FINAL_INSTRUCTION)

        # Combine and format all parts at once
        full_template = "".join(prompt_parts)
        return full_template.format(**template_vars)

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
