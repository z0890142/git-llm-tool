"""Tests for Git helper functionality."""

import pytest
from unittest.mock import patch, MagicMock
import subprocess

from git_llm_tool.core.git_helper import GitHelper
from git_llm_tool.core.exceptions import GitError


class TestGitHelper:
    """Test GitHelper functionality."""

    @patch('git_llm_tool.core.git_helper.subprocess.run')
    def test_verify_git_repo_success(self, mock_run):
        """Test successful git repository verification."""
        mock_run.return_value = MagicMock(
            stdout=".git",
            returncode=0
        )

        # Should not raise exception
        helper = GitHelper()
        assert helper is not None

    @patch('git_llm_tool.core.git_helper.subprocess.run')
    def test_verify_git_repo_failure(self, mock_run):
        """Test git repository verification failure."""
        mock_run.side_effect = subprocess.CalledProcessError(
            128, ['git', 'rev-parse', '--git-dir'], stderr="Not a git repository"
        )

        with pytest.raises(GitError, match="Not in a git repository"):
            GitHelper()

    @patch('git_llm_tool.core.git_helper.subprocess.run')
    def test_get_staged_diff_success(self, mock_run):
        """Test getting staged diff successfully."""
        # Mock initial git repo check
        mock_run.side_effect = [
            MagicMock(stdout=".git", returncode=0),  # repo verification
            MagicMock(stdout="diff --git a/file.py b/file.py\\n+new line", returncode=0)  # diff
        ]

        helper = GitHelper()
        diff = helper.get_staged_diff()

        assert diff == "diff --git a/file.py b/file.py\\n+new line"
        mock_run.assert_called_with(
            ["git", "diff", "--cached"],
            capture_output=True,
            text=True,
            check=True,
            cwd=mock_run.call_args[1]['cwd']
        )

    @patch('git_llm_tool.core.git_helper.subprocess.run')
    def test_get_staged_diff_no_changes(self, mock_run):
        """Test getting staged diff with no changes."""
        mock_run.side_effect = [
            MagicMock(stdout=".git", returncode=0),  # repo verification
            MagicMock(stdout="", returncode=0)  # empty diff
        ]

        helper = GitHelper()

        with pytest.raises(GitError, match="No staged changes found"):
            helper.get_staged_diff()

    @patch('git_llm_tool.core.git_helper.subprocess.run')
    def test_get_current_branch(self, mock_run):
        """Test getting current branch name."""
        mock_run.side_effect = [
            MagicMock(stdout=".git", returncode=0),  # repo verification
            MagicMock(stdout="feature/test-branch", returncode=0)  # branch name
        ]

        helper = GitHelper()
        branch = helper.get_current_branch()

        assert branch == "feature/test-branch"

    @patch('git_llm_tool.core.git_helper.subprocess.run')
    def test_get_commit_messages_with_tag(self, mock_run):
        """Test getting commit messages from last tag."""
        mock_run.side_effect = [
            MagicMock(stdout=".git", returncode=0),  # repo verification
            MagicMock(stdout="v1.0.0", returncode=0),  # last tag
            MagicMock(stdout="feat: add new feature\\nfix: bug fix\\nchore: update deps", returncode=0)  # log
        ]

        helper = GitHelper()
        messages = helper.get_commit_messages()

        assert messages == ["feat: add new feature", "fix: bug fix", "chore: update deps"]

    @patch('git_llm_tool.core.git_helper.subprocess.run')
    def test_get_commit_messages_no_tags(self, mock_run):
        """Test getting commit messages when no tags exist."""
        mock_run.side_effect = [
            MagicMock(stdout=".git", returncode=0),  # repo verification
            subprocess.CalledProcessError(128, ['git', 'describe'], stderr="No tags"),  # no tags
            MagicMock(stdout="abc123", returncode=0),  # initial commit
            MagicMock(stdout="feat: initial commit\\nfeat: add feature", returncode=0)  # log
        ]

        helper = GitHelper()
        messages = helper.get_commit_messages()

        assert messages == ["feat: initial commit", "feat: add feature"]

    @patch('git_llm_tool.core.git_helper.subprocess.run')
    def test_commit_with_message(self, mock_run):
        """Test creating commit with message."""
        mock_run.side_effect = [
            MagicMock(stdout=".git", returncode=0),  # repo verification
            MagicMock(stdout="", returncode=0)  # commit
        ]

        helper = GitHelper()
        helper.commit_with_message("test commit message")

        # Verify commit command was called
        commit_call = mock_run.call_args_list[1]
        assert commit_call[0][0] == ["git", "commit", "-m", "test commit message"]

    @patch('git_llm_tool.core.git_helper.subprocess.run')
    def test_commit_with_message_nothing_to_commit(self, mock_run):
        """Test commit failure when nothing to commit."""
        mock_run.side_effect = [
            MagicMock(stdout=".git", returncode=0),  # repo verification
            subprocess.CalledProcessError(1, ['git', 'commit'], stderr="nothing to commit")
        ]

        helper = GitHelper()

        with pytest.raises(GitError, match="No staged changes to commit"):
            helper.commit_with_message("test message")

    @patch('git_llm_tool.core.git_helper.subprocess.run')
    def test_get_repository_info(self, mock_run):
        """Test getting repository information."""
        mock_run.side_effect = [
            MagicMock(stdout=".git", returncode=0),  # repo verification
            MagicMock(stdout="main", returncode=0),  # current branch
            MagicMock(stdout="file1.py\\nfile2.py", returncode=0),  # staged changes
            MagicMock(stdout="file3.py", returncode=0),  # unstaged changes
            MagicMock(stdout="/path/to/repo", returncode=0)  # repo root
        ]

        helper = GitHelper()
        info = helper.get_repository_info()

        expected = {
            "branch": "main",
            "has_staged_changes": True,
            "has_unstaged_changes": True,
            "repository_root": "/path/to/repo"
        }

        assert info == expected

    @patch('git_llm_tool.core.git_helper.subprocess.run')
    def test_is_clean_workspace(self, mock_run):
        """Test checking if workspace is clean."""
        mock_run.side_effect = [
            MagicMock(stdout=".git", returncode=0),  # repo verification
            MagicMock(stdout="", returncode=0),  # no staged changes
            MagicMock(stdout="", returncode=0),  # no unstaged changes
            MagicMock(stdout="", returncode=0)   # no untracked files
        ]

        helper = GitHelper()
        is_clean = helper.is_clean_workspace()

        assert is_clean is True

    @patch('git_llm_tool.core.git_helper.subprocess.run')
    def test_is_dirty_workspace(self, mock_run):
        """Test checking workspace with uncommitted changes."""
        mock_run.side_effect = [
            MagicMock(stdout=".git", returncode=0),  # repo verification
            MagicMock(stdout="staged_file.py", returncode=0),  # staged changes
            MagicMock(stdout="", returncode=0),  # no unstaged changes
            MagicMock(stdout="", returncode=0)   # no untracked files
        ]

        helper = GitHelper()
        is_clean = helper.is_clean_workspace()

        assert is_clean is False

    @patch('git_llm_tool.core.git_helper.subprocess.run')
    def test_git_command_not_found(self, mock_run):
        """Test error when git command is not found."""
        mock_run.side_effect = FileNotFoundError("git command not found")

        with pytest.raises(GitError, match="Git command not found"):
            GitHelper()

    @patch('git_llm_tool.core.git_helper.subprocess.run')
    def test_git_command_error(self, mock_run):
        """Test handling of git command errors."""
        mock_run.side_effect = [
            MagicMock(stdout=".git", returncode=0),  # repo verification
            subprocess.CalledProcessError(1, ['git', 'diff'], stderr="Permission denied")
        ]

        helper = GitHelper()

        with pytest.raises(GitError, match="Git command failed"):
            helper.get_staged_diff()