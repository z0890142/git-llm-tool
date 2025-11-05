"""Main CLI interface for git-llm-tool."""

import click
from pathlib import Path

from git_llm_tool import __version__
from git_llm_tool.core.config import ConfigLoader, get_config
from git_llm_tool.core.exceptions import ConfigError
from git_llm_tool.commands.commit_cmd import execute_commit


@click.group()
@click.version_option(version=__version__)
@click.option(
    "--verbose", "-v", is_flag=True, help="Enable verbose output"
)
@click.pass_context
def main(ctx, verbose):
    """AI-powered git commit message and changelog generator."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose


@main.command()
@click.option(
    "--apply", "-a", is_flag=True,
    help="Apply the commit message directly without opening editor"
)
@click.option(
    "--model", "-m",
    help="LLM model to use (overrides config)"
)
@click.option(
    "--language", "-l",
    help="Output language (overrides config)"
)
@click.pass_context
def commit(ctx, apply, model, language):
    """Generate AI-powered commit message from staged changes."""
    verbose = ctx.obj.get('verbose', False) if ctx.obj else False
    execute_commit(apply=apply, model=model, language=language, verbose=verbose)


@main.command()
@click.option(
    "--from", "from_ref",
    help="Starting reference (default: last tag)"
)
@click.option(
    "--to", "to_ref", default="HEAD",
    help="Ending reference (default: HEAD)"
)
@click.option(
    "--output", "-o",
    help="Output file (default: stdout)"
)
@click.option(
    "--force", "-f", is_flag=True,
    help="Force overwrite existing output file"
)
@click.pass_context
def changelog(ctx, from_ref, to_ref, output, force):
    """Generate changelog from git history."""
    click.echo("📋 Generating changelog...")

    # This will be implemented in subsequent tasks
    if output:
        click.echo(f"📄 Changelog saved to {output}")
    else:
        click.echo("📄 Changelog output to stdout")


@main.group()
def config():
    """Configuration management commands."""
    pass


@config.command()
@click.argument("key")
@click.argument("value")
def set(key, value):
    """Set configuration value."""
    try:
        config_loader = ConfigLoader()
        config_loader.set_value(key, value)

        # Save to global config
        config_loader.save_config()

        click.echo(f"✅ Set {key} = {value}")
    except ConfigError as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)


@config.command()
@click.argument("key", required=False)
def get(key):
    """Get configuration value(s)."""
    try:
        config = get_config()

        if key:
            # Get specific key
            config_loader = ConfigLoader()
            value = config_loader.get_value(key)
            click.echo(f"{key} = {value}")
        else:
            # Show all configuration
            click.echo("📋 Current Configuration:")
            click.echo(f"  llm.default_model = {config.llm.default_model}")
            click.echo(f"  llm.language = {config.llm.language}")

            if config.llm.api_keys:
                click.echo("  llm.api_keys:")
                for provider, key_value in config.llm.api_keys.items():
                    # Hide API key for security
                    masked_key = key_value[:8] + "..." if len(key_value) > 8 else "***"
                    click.echo(f"    {provider} = {masked_key}")

            if config.llm.azure_openai:
                click.echo("  llm.azure_openai:")
                for key, value in config.llm.azure_openai.items():
                    click.echo(f"    {key} = {value}")

            click.echo(f"  jira.enabled = {config.jira.enabled}")
            if config.jira.branch_regex:
                click.echo(f"  jira.branch_regex = {config.jira.branch_regex}")

    except ConfigError as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)


@config.command()
def init():
    """Initialize configuration file."""
    try:
        config_path = Path.home() / ".git-llm-tool" / "config.yaml"

        if config_path.exists():
            if not click.confirm(f"Configuration file already exists at {config_path}. Overwrite?"):
                click.echo("❌ Initialization cancelled.")
                return

        # Create default configuration
        config_loader = ConfigLoader()
        config_loader.save_config(config_path)

        click.echo(f"✅ Configuration initialized at {config_path}")
        click.echo("💡 You can now set API keys with:")
        click.echo("   git-llm config set llm.api_keys.openai sk-your-key-here")

    except Exception as e:
        click.echo(f"❌ Failed to initialize configuration: {e}", err=True)


if __name__ == "__main__":
    main()