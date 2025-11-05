"""Git operations helper for git-llm-tool."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from git_llm_tool.core.exceptions import GitError


class GitHelper:
    """Helper class for Git operations."""

    def __init__(self):
        """Initialize Git helper."""
        self._verify_git_repo()

    def _verify_git_repo(self) -> None:
        """Verify that we're in a git repository."""
        try:
            self._run_git_command(["git", "rev-parse", "--git-dir"])
        except GitError:
            raise GitError("Not in a git repository")

    def _run_git_command(self, command: List[str]) -> str:
        """Run a git command and return output.

        Args:
            command: Git command as list of strings

        Returns:
            Command output as string

        Raises:
            GitError: If command fails
        """
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                cwd=os.getcwd()
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.strip() if e.stderr else "Unknown error"
            raise GitError(f"Git command failed: {' '.join(command)}\n{stderr}")
        except FileNotFoundError:
            raise GitError("Git command not found. Is git installed?")

    def get_staged_diff(self) -> str:
        """Get diff of staged changes.

        Returns:
            Git diff output of staged changes

        Raises:
            GitError: If no staged changes or git command fails
        """
        diff = self._run_git_command(["git", "diff", "--cached"])

        if not diff.strip():
            raise GitError("No staged changes found. Use 'git add' to stage files first.")

        return diff

    def get_current_branch(self) -> str:
        """Get current branch name.

        Returns:
            Current branch name

        Raises:
            GitError: If git command fails
        """
        return self._run_git_command(["git", "symbolic-ref", "--short", "HEAD"])

    def get_commit_messages(self, from_ref: Optional[str] = None, to_ref: str = "HEAD") -> List[str]:
        """Get commit messages in a range.

        Args:
            from_ref: Starting reference (if None, uses last tag)
            to_ref: Ending reference

        Returns:
            List of commit messages

        Raises:
            GitError: If git command fails
        """
        if from_ref is None:
            # Try to get last tag
            try:
                from_ref = self._run_git_command(["git", "describe", "--tags", "--abbrev=0"])
            except GitError:
                # If no tags exist, use initial commit
                from_ref = self._run_git_command(["git", "rev-list", "--max-parents=0", "HEAD"])

        # Get commit messages in range
        commit_range = f"{from_ref}..{to_ref}"
        log_output = self._run_git_command([
            "git", "log", commit_range, "--pretty=format:%s"
        ])

        if not log_output.strip():
            raise GitError(f"No commits found in range {commit_range}")

        return [msg.strip() for msg in log_output.split('\n') if msg.strip()]

    def commit_with_message(self, message: str) -> None:
        """Create a commit with the given message.

        Args:
            message: Commit message

        Raises:
            GitError: If commit fails
        """
        try:
            self._run_git_command(["git", "commit", "-m", message])
        except GitError as e:
            if "nothing to commit" in str(e).lower():
                raise GitError("No staged changes to commit")
            raise

    def open_commit_editor(self, message: str) -> bool:
        """Open commit message in editor for review.

        Args:
            message: Initial commit message

        Returns:
            True if commit was made, False if cancelled

        Raises:
            GitError: If git operations fail
        """
        # Create temporary file with commit message
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(message)
            temp_file_path = temp_file.name

        try:
            # Get git editor
            editor = self._get_git_editor()

            # Open editor
            result = subprocess.run([editor, temp_file_path])

            if result.returncode != 0:
                raise GitError("Editor exited with non-zero status")

            # Read edited message
            with open(temp_file_path, 'r') as f:
                edited_message = f.read().strip()

            # Check if message was cleared (user cancelled)
            if not edited_message:
                return False

            # Create commit with edited message
            self.commit_with_message(edited_message)
            return True

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass

    def _get_git_editor(self) -> str:
        """Get the configured git editor.

        Returns:
            Editor command

        Raises:
            GitError: If no editor is configured
        """
        # Try git config
        try:
            editor = self._run_git_command(["git", "config", "--get", "core.editor"])
            if editor:
                return editor
        except GitError:
            pass

        # Try environment variables
        for env_var in ["GIT_EDITOR", "VISUAL", "EDITOR"]:
            editor = os.environ.get(env_var)
            if editor:
                return editor

        # Default editors by platform
        if os.name == 'nt':
            # Windows
            return "notepad"
        else:
            # Unix-like systems
            for default_editor in ["nano", "vim", "vi"]:
                try:
                    subprocess.run(["which", default_editor],
                                 capture_output=True, check=True)
                    return default_editor
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue

        raise GitError("No suitable editor found. Please set core.editor in git config")

    def get_repository_info(self) -> dict:
        """Get basic repository information.

        Returns:
            Dictionary with repository info
        """
        try:
            return {
                "branch": self.get_current_branch(),
                "has_staged_changes": bool(self._run_git_command(["git", "diff", "--cached", "--name-only"])),
                "has_unstaged_changes": bool(self._run_git_command(["git", "diff", "--name-only"])),
                "repository_root": self._run_git_command(["git", "rev-parse", "--show-toplevel"])
            }
        except GitError:
            return {}

    def is_clean_workspace(self) -> bool:
        """Check if workspace has no uncommitted changes.

        Returns:
            True if workspace is clean
        """
        try:
            staged = self._run_git_command(["git", "diff", "--cached", "--name-only"])
            unstaged = self._run_git_command(["git", "diff", "--name-only"])
            untracked = self._run_git_command(["git", "ls-files", "--others", "--exclude-standard"])

            return not (staged or unstaged or untracked)
        except GitError:
            return False