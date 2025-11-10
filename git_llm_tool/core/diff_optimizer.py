"""Diff optimization strategies to reduce token usage."""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DiffStats:
    """Statistics about diff optimization."""
    original_size: int
    optimized_size: int
    compression_ratio: float
    files_processed: int
    lines_removed: int


class DiffOptimizer:
    """Optimize git diffs to reduce token usage while preserving meaning."""

    def __init__(self, max_context_lines: int = 3):
        self.max_context_lines = max_context_lines

    def optimize_diff(self, diff: str, aggressive: bool = False) -> Tuple[str, DiffStats]:
        """Optimize diff to reduce token usage."""
        original_size = len(diff)
        lines = diff.split('\n')

        optimized_lines = []
        files_processed = 0
        lines_removed = 0

        i = 0
        while i < len(lines):
            line = lines[i]

            # Keep file headers
            if line.startswith('diff --git'):
                files_processed += 1
                optimized_lines.append(line)
                i += 1
                continue

            # Keep index and mode lines (compressed)
            if line.startswith('index ') or line.startswith('new file mode') or line.startswith('deleted file mode'):
                if not aggressive:
                    optimized_lines.append(line)
                else:
                    lines_removed += 1
                i += 1
                continue

            # Keep file paths but simplify
            if line.startswith('--- ') or line.startswith('+++ '):
                if aggressive:
                    # Simplify path names
                    simplified = re.sub(r'^(---|\+\+\+) [ab]/', r'\1 ', line)
                    optimized_lines.append(simplified)
                else:
                    optimized_lines.append(line)
                i += 1
                continue

            # Process hunk headers
            if line.startswith('@@'):
                optimized_lines.append(line)
                i += 1

                # Process the content of this hunk
                hunk_lines, removed_count = self._process_hunk(lines[i:], aggressive)
                optimized_lines.extend(hunk_lines)
                lines_removed += removed_count
                i += len(hunk_lines)
                continue

            # Keep other lines as-is
            optimized_lines.append(line)
            i += 1

        optimized_diff = '\n'.join(optimized_lines)
        optimized_size = len(optimized_diff)

        stats = DiffStats(
            original_size=original_size,
            optimized_size=optimized_size,
            compression_ratio=optimized_size / original_size if original_size > 0 else 1.0,
            files_processed=files_processed,
            lines_removed=lines_removed
        )

        return optimized_diff, stats

    def _process_hunk(self, lines: List[str], aggressive: bool) -> Tuple[List[str], int]:
        """Process a hunk to optimize context and changes."""
        result = []
        removed_count = 0

        # Group consecutive context lines
        context_buffer = []

        for line in lines:
            # Stop at next hunk or file
            if line.startswith('@@') or line.startswith('diff --git'):
                break

            if line.startswith(' '):  # Context line
                context_buffer.append(line)
            else:  # Added or removed line
                # Flush context buffer with limit
                if context_buffer:
                    if aggressive and len(context_buffer) > self.max_context_lines * 2:
                        # Keep first and last few context lines
                        keep_start = context_buffer[:self.max_context_lines]
                        keep_end = context_buffer[-self.max_context_lines:]
                        result.extend(keep_start)
                        if len(context_buffer) > self.max_context_lines * 2:
                            result.append(f' ... ({len(context_buffer) - len(keep_start) - len(keep_end)} context lines omitted)')
                        result.extend(keep_end)
                        removed_count += len(context_buffer) - len(keep_start) - len(keep_end)
                    else:
                        result.extend(context_buffer)
                    context_buffer = []

                # Keep change lines (but can compress whitespace-only changes)
                if aggressive and self._is_whitespace_only_change(line):
                    result.append(f'{line[0]} (whitespace change)')
                    removed_count += 1
                else:
                    result.append(line)

        # Handle remaining context
        if context_buffer:
            if aggressive and len(context_buffer) > self.max_context_lines:
                result.extend(context_buffer[:self.max_context_lines])
                result.append(f' ... ({len(context_buffer) - self.max_context_lines} trailing context lines omitted)')
                removed_count += len(context_buffer) - self.max_context_lines
            else:
                result.extend(context_buffer)

        return result, removed_count

    def _is_whitespace_only_change(self, line: str) -> bool:
        """Check if a line represents only whitespace changes."""
        if len(line) < 2:
            return False

        content = line[1:]  # Remove +/- prefix
        return content.strip() == '' or re.match(r'^\s+$', content)

    def smart_truncate(self, diff: str, max_tokens: int = 8000) -> str:
        """Smart truncation that preserves important parts of the diff."""
        # Rough estimation: 1 token ≈ 4 characters
        max_chars = max_tokens * 4

        if len(diff) <= max_chars:
            return diff

        lines = diff.split('\n')
        important_lines = []
        regular_lines = []

        for line in lines:
            if self._is_important_line(line):
                important_lines.append(line)
            else:
                regular_lines.append(line)

        # Always keep important lines
        result = important_lines[:]
        current_size = sum(len(line) + 1 for line in result)  # +1 for newline

        # Add regular lines until we hit the limit
        for line in regular_lines:
            if current_size + len(line) + 1 > max_chars:
                result.append('... (diff truncated to fit token limit)')
                break
            result.append(line)
            current_size += len(line) + 1

        return '\n'.join(result)

    def _is_important_line(self, line: str) -> bool:
        """Determine if a line is important and should be preserved."""
        return (
            line.startswith('diff --git') or
            line.startswith('+++') or
            line.startswith('---') or
            line.startswith('@@') or
            (line.startswith('+') and not self._is_whitespace_only_change(line)) or
            (line.startswith('-') and not self._is_whitespace_only_change(line))
        )

    def get_summary_stats(self, diff: str) -> dict:
        """Get summary statistics about a diff."""
        lines = diff.split('\n')

        stats = {
            'total_lines': len(lines),
            'files_changed': len([l for l in lines if l.startswith('diff --git')]),
            'lines_added': len([l for l in lines if l.startswith('+') and not l.startswith('+++')]),
            'lines_removed': len([l for l in lines if l.startswith('-') and not l.startswith('---')]),
            'context_lines': len([l for l in lines if l.startswith(' ')]),
            'estimated_tokens': len(diff) // 4  # Rough estimation
        }

        return stats