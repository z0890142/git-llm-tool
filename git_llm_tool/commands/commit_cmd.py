"""Commit command implementation."""

import click
from typing import Optional

from git_llm_tool.core.config import get_config
from git_llm_tool.core.git_helper import GitHelper
from git_llm_tool.core.jira_helper import JiraHelper
from git_llm_tool.core.exceptions import GitError, ApiError, ConfigError, JiraError
from git_llm_tool.providers import get_provider


def execute_commit(
    apply: bool = False,
    model: Optional[str] = None,
    language: Optional[str] = None,
    verbose: bool = False
) -> None:
    """Execute the commit command logic.

    Args:
        apply: Whether to apply commit directly without editor
        model: Override model from config
        language: Override language from config
        verbose: Enable verbose output
    """
    try:
        # Load configuration
        config = get_config()

        # Override config with CLI parameters
        if model:
            config.llm.default_model = model
        if language:
            config.llm.language = language

        if verbose:
            click.echo(f"📄 Using model: {config.llm.default_model}")
            click.echo(f"🌐 Using language: {config.llm.language}")

        # Initialize Git helper
        git_helper = GitHelper()

        # Get staged diff
        if verbose:
            click.echo("📊 Getting staged changes...")

        try:
            diff = git_helper.get_staged_diff()
        except GitError as e:
            click.echo(f"❌ {e}", err=True)
            click.echo("💡 Tip: Use 'git add' to stage files before committing", err=True)
            return

        if verbose:
            click.echo(f"📝 Found {len(diff.splitlines())} lines of changes")

        # Get LLM provider
        try:
            provider = get_provider(config)
            if verbose:
                click.echo(f"🤖 Using provider: {provider.__class__.__name__}")
        except ApiError as e:
            click.echo(f"❌ {e}", err=True)
            return

        # Initialize Jira helper and get Jira context
        jira_helper = JiraHelper(config, git_helper)

        try:
            jira_helper.validate_config()
            jira_ticket, work_hours = jira_helper.get_jira_context(verbose=verbose)

            if jira_ticket or work_hours:
                jira_info = jira_helper.format_jira_info(jira_ticket, work_hours)
                click.echo(f"📋 Jira Info: {jira_info}")

        except JiraError as e:
            click.echo(f"⚠️  Jira Error: {e}", err=True)
            # Continue without Jira info
            jira_ticket = None
            work_hours = None

        # Generate commit message
        click.echo("🤖 Generating commit message...")

        try:
            commit_message = provider.generate_commit_message(
                diff=diff,
                jira_ticket=jira_ticket,
                work_hours=work_hours
            )
        except ApiError as e:
            click.echo(f"❌ API Error: {e}", err=True)
            return

        if verbose:
            click.echo(f"✨ Generated message: {commit_message}")

        # Apply commit or open editor
        if apply:
            # Direct commit
            try:
                git_helper.commit_with_message(commit_message)
                click.echo("✅ Commit applied successfully!")
                click.echo(f"📝 Message: {commit_message}")
            except GitError as e:
                click.echo(f"❌ Commit failed: {e}", err=True)
        else:
            # Open editor for review
            click.echo("📝 Opening editor for review...")
            try:
                committed = git_helper.open_commit_editor(commit_message, config)
                if committed:
                    click.echo("✅ Commit created successfully!")
                else:
                    click.echo("❌ Commit cancelled by user")
            except GitError as e:
                click.echo(f"❌ Editor error: {e}", err=True)

    except ConfigError as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)