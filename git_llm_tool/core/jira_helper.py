"""Jira integration helper for git-llm-tool."""

import re
import click
from typing import Optional, Tuple

from git_llm_tool.core.config import AppConfig
from git_llm_tool.core.git_helper import GitHelper
from git_llm_tool.core.exceptions import JiraError


class JiraHelper:
    """Helper class for Jira integration."""

    def __init__(self, config: AppConfig, git_helper: GitHelper):
        """Initialize Jira helper.

        Args:
            config: Application configuration
            git_helper: Git helper instance
        """
        self.config = config
        self.git_helper = git_helper

    def get_jira_context(
        self, verbose: bool = False
    ) -> Tuple[Optional[str], Optional[str]]:
        """Get Jira ticket and work hours context.

        Args:
            verbose: Enable verbose output

        Returns:
            Tuple of (jira_ticket, work_hours)
        """
        if not self.config.jira.enabled:
            if verbose:
                click.echo("🔒 Jira integration is disabled")
            return None, None

        # Try to extract ticket from branch name
        jira_ticket = self._extract_ticket_from_branch()

        if jira_ticket:
            if verbose:
                click.echo(f"🎯 Auto-detected Jira ticket: {jira_ticket}")
        else:
            # Interactive prompt for ticket
            jira_ticket = self._prompt_for_ticket()

        # Interactive prompt for work hours
        work_hours = self._prompt_for_work_hours()

        return jira_ticket, work_hours

    def _extract_ticket_from_branch(self) -> Optional[str]:
        """Extract Jira ticket from current branch name using regex.

        Returns:
            Jira ticket number if found, None otherwise
        """
        try:
            branch_name = self.git_helper.get_current_branch()

            # Use ticket pattern to extract Jira ticket
            if self.config.jira.ticket_pattern:
                match = re.search(self.config.jira.ticket_pattern, branch_name)
                if match:
                    # If the pattern has capture groups, use the first one
                    if match.groups():
                        return match.group(1)
                    else:
                        # If no capture groups, use the whole match
                        return match.group(0)

        except Exception:
            # Ignore any errors in regex matching or git operations
            pass

        return None

    def _is_jira_ticket_format(self, text: str) -> bool:
        """Check if text matches typical Jira ticket format.

        Args:
            text: Text to check

        Returns:
            True if text looks like a Jira ticket (e.g., PROJECT-123)
        """
        import re

        # Common Jira ticket format: UPPERCASE-DIGITS
        return bool(re.match(r"^[A-Z]+-\d+$", text))

    def _prompt_for_ticket(self) -> Optional[str]:
        """Interactively prompt user for Jira ticket.

        Returns:
            Jira ticket number or None if skipped
        """
        click.echo("\n🎫 Jira Integration")
        ticket = click.prompt(
            "Enter Jira ticket number (or press Enter to skip)",
            default="",
            show_default=False,
        ).strip()

        if not ticket:
            return None

        # Basic validation - should look like a Jira ticket
        if not re.match(r"^[A-Z]+-\d+$", ticket.upper()):
            click.echo(
                "⚠️  Warning: Ticket format doesn't look like standard Jira format (e.g., PROJ-123)"
            )
            if not click.confirm("Continue anyway?"):
                return None

        return ticket.upper()

    def _prompt_for_work_hours(self) -> Optional[str]:
        """Interactively prompt user for work hours.

        Returns:
            Work hours string or None if skipped
        """
        work_hours = click.prompt(
            "Enter work hours (e.g., '1h 30m', '2h', '45m', '1d 2h', '1w 3d 4h 30m') or press Enter to skip",
            default="",
            show_default=False,
        ).strip()

        if not work_hours:
            return None

        # Basic validation for work hours format
        if not re.match(
            r"^(\d+w\s*)?(\d+d\s*)?(\d+h\s*)?(\d+m)?$",
            work_hours.lower().replace(" ", ""),
        ):
            click.echo(
                "⚠️  Warning: Work hours format should be like '1h 30m', '2h', '45m', '1d 2h', or '1w 2d 3h 30m'"
            )
            if not click.confirm("Continue anyway?"):
                return None

        # Normalize the work hours format
        normalized_hours = self._normalize_work_hours(work_hours)
        return normalized_hours

    def _normalize_work_hours(self, work_hours: str) -> str:
        """Normalize work hours to standard format: 0w 0d 0h 0m.

        Args:
            work_hours: Input work hours string (e.g., '1h 30m', '2h', '45m')

        Returns:
            Normalized work hours string in format '0w 0d 0h 0m'
        """
        # Initialize all time units to 0
        weeks = 0
        days = 0
        hours = 0
        minutes = 0

        # Clean the input and make it lowercase
        clean_input = work_hours.lower().replace(" ", "")

        # Extract weeks
        week_match = re.search(r"(\d+)w", clean_input)
        if week_match:
            weeks = int(week_match.group(1))

        # Extract days
        day_match = re.search(r"(\d+)d", clean_input)
        if day_match:
            days = int(day_match.group(1))

        # Extract hours
        hour_match = re.search(r"(\d+)h", clean_input)
        if hour_match:
            hours = int(hour_match.group(1))

        # Extract minutes
        minute_match = re.search(r"(\d+)m", clean_input)
        if minute_match:
            minutes = int(minute_match.group(1))

        # Return in standard format
        return f"{weeks}w {days}d {hours}h {minutes}m"

    def format_jira_info(
        self, jira_ticket: Optional[str], work_hours: Optional[str]
    ) -> str:
        """Format Jira information for display.

        Args:
            jira_ticket: Jira ticket number
            work_hours: Work hours

        Returns:
            Formatted string for display
        """
        info_parts = []

        if jira_ticket:
            info_parts.append(f"🎫 Ticket: {jira_ticket}")

        if work_hours:
            info_parts.append(f"⏱️  Time: {work_hours}")

        return " | ".join(info_parts) if info_parts else "No Jira information"

    def validate_config(self) -> bool:
        """Validate Jira configuration.

        Returns:
            True if configuration is valid

        Raises:
            JiraError: If configuration is invalid
        """
        if not self.config.jira.enabled:
            return True

        # Validate ticket pattern regex if provided
        if self.config.jira.ticket_pattern:
            try:
                re.compile(self.config.jira.ticket_pattern)
            except re.error as e:
                raise JiraError(f"Invalid ticket pattern regex: {e}")

        return True
