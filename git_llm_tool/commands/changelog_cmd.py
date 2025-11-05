"""Changelog command implementation."""

import click
import os
from datetime import datetime
from typing import Optional

from git_llm_tool.core.config import get_config
from git_llm_tool.core.git_helper import GitHelper
from git_llm_tool.core.exceptions import GitError, ApiError, ConfigError
from git_llm_tool.providers import get_provider


def _manage_changelog_file(new_content: str, verbose: bool = False) -> str:
    """Manage the changelog.md file in the repository root.

    Args:
        new_content: New changelog content to add
        verbose: Enable verbose output

    Returns:
        Path to the changelog file
    """
    # Get repository root
    git_helper = GitHelper()
    repo_info = git_helper.get_repository_info()
    repo_root = repo_info.get('repository_root', os.getcwd())

    changelog_path = os.path.join(repo_root, 'changelog.md')

    # Get current date
    current_date = datetime.now().strftime('%Y-%m-%d')

    # Create header with date
    header = f"\n## {current_date}\n\n"

    # Clean the new content (remove any duplicate titles)
    cleaned_content = new_content.strip()
    if cleaned_content.startswith('# Changelog'):
        # Remove the duplicate title line
        lines = cleaned_content.split('\n')
        lines = [line for line in lines[1:] if line.strip()]  # Skip title and empty lines
        cleaned_content = '\n'.join(lines)

    # Prepare content to add
    content_to_add = header + cleaned_content + "\n"

    if os.path.exists(changelog_path):
        if verbose:
            click.echo(f"📝 Found existing changelog at {changelog_path}")

        # Read existing content
        try:
            with open(changelog_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        except IOError as e:
            raise Exception(f"Failed to read existing changelog: {e}")

        # Check if we're at the beginning of the file or need to add after title
        if existing_content.strip().startswith('# '):
            # Find the end of the title line
            lines = existing_content.split('\n')
            title_line = lines[0]
            rest_content = '\n'.join(lines[1:])

            # Insert new content after title
            final_content = title_line + '\n' + content_to_add + rest_content
        else:
            # Prepend to existing content
            final_content = content_to_add + existing_content

    else:
        if verbose:
            click.echo(f"📄 Creating new changelog at {changelog_path}")

        # Create new changelog with header
        final_content = f"# Changelog\n{content_to_add}"

    # Write the file
    try:
        with open(changelog_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        if verbose:
            click.echo(f"✅ Updated changelog at {changelog_path}")
    except IOError as e:
        raise Exception(f"Failed to write changelog: {e}")

    return changelog_path


def execute_changelog(
    from_ref: Optional[str] = None,
    to_ref: str = "HEAD",
    output: Optional[str] = None,
    force: bool = False,
    verbose: bool = False
) -> None:
    """Execute the changelog command logic.

    Args:
        from_ref: Starting reference (default: last tag)
        to_ref: Ending reference (default: HEAD)
        output: Output file path
        force: Force overwrite existing file
        verbose: Enable verbose output
    """
    try:
        # Load configuration
        config = get_config()

        if verbose:
            click.echo(f"📄 Using model: {config.llm.default_model}")
            click.echo(f"🌐 Using language: {config.llm.language}")

        # Initialize Git helper
        git_helper = GitHelper()

        # Get commit messages in range
        if verbose:
            click.echo("📊 Getting commit messages...")

        try:
            commit_messages = git_helper.get_commit_messages(from_ref, to_ref)
        except GitError as e:
            click.echo(f"❌ {e}", err=True)
            return

        if verbose:
            click.echo(f"📝 Found {len(commit_messages)} commits")

        # Get LLM provider
        try:
            provider = get_provider(config)
            if verbose:
                click.echo(f"🤖 Using provider: {provider.__class__.__name__}")
        except ApiError as e:
            click.echo(f"❌ {e}", err=True)
            return

        # Generate changelog
        click.echo("🤖 Generating changelog...")

        try:
            changelog = provider.generate_changelog(commit_messages)
        except ApiError as e:
            click.echo(f"❌ API Error: {e}", err=True)
            return

        if verbose:
            click.echo(f"✨ Generated changelog ({len(changelog)} characters)")

        # Output changelog
        if output:
            # Custom output file specified
            if os.path.exists(output) and not force:
                if not click.confirm(f"File {output} exists. Overwrite?"):
                    click.echo("❌ Changelog generation cancelled.")
                    return

            # Write to custom file
            try:
                with open(output, 'w', encoding='utf-8') as f:
                    f.write(changelog)
                click.echo(f"✅ Changelog saved to {output}")
            except IOError as e:
                click.echo(f"❌ Failed to write to {output}: {e}", err=True)
        else:
            # Auto-manage changelog.md in repository root
            try:
                changelog_path = _manage_changelog_file(changelog, verbose)
                click.echo(f"✅ Changelog updated in {changelog_path}")

                # Also show the generated content
                if verbose:
                    click.echo("\n" + "="*60)
                    click.echo("📋 Generated Content:")
                    click.echo("="*60)
                    click.echo(changelog)
                    click.echo("="*60)
            except Exception as e:
                click.echo(f"❌ Failed to update changelog.md: {e}", err=True)

    except ConfigError as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)