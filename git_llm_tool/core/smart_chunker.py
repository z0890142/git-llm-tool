"""Smart chunking strategies for git diffs."""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
from langchain_core.documents import Document
from git_llm_tool.core.token_counter import TokenCounter


@dataclass
class ChunkInfo:
    """Information about a chunk."""
    content: str
    file_path: Optional[str]
    chunk_type: str  # 'file', 'hunk', 'size-based'
    size: int
    is_complete_file: bool


class SmartChunker:
    """Smart chunker that prioritizes file-based splitting over size-based."""

    def __init__(self, chunk_size: int = 3000, chunk_overlap: int = 150, model_name: str = "gpt-4o"):
        self.chunk_size = chunk_size  # Now in tokens
        self.chunk_overlap = chunk_overlap  # Now in tokens
        self.token_counter = TokenCounter(model_name)

        # For backward compatibility, also keep character-based limits as fallback
        self.char_chunk_size = chunk_size * 3  # Rough estimate: 1 token ≈ 3 chars
        self.char_chunk_overlap = chunk_overlap * 3

    def chunk_diff(self, diff: str) -> List[ChunkInfo]:
        """
        Intelligently chunk a git diff.

        Strategy:
        1. First try to split by files
        2. If files are too large, split by hunks
        3. If hunks are still too large, apply size-based splitting to oversized hunks
        4. As last resort, use pure size-based splitting
        """
        chunks = []

        # Split diff into files
        file_sections = self._split_by_files(diff)

        for file_path, file_content in file_sections:
            # Use token-based size checking
            file_tokens = self.token_counter.count_tokens(file_content)

            if file_tokens <= self.chunk_size:
                # File fits in one chunk
                chunks.append(ChunkInfo(
                    content=file_content,
                    file_path=file_path,
                    chunk_type='file',
                    size=len(file_content),  # Keep character size for backward compatibility
                    is_complete_file=True
                ))
            else:
                # File is too large, try to split by hunks
                hunk_chunks = self._split_file_by_hunks(file_content, file_path)
                if hunk_chunks:
                    # Check if any hunk chunks are still too large (token-based)
                    final_chunks = []
                    for hunk_chunk in hunk_chunks:
                        hunk_tokens = self.token_counter.count_tokens(hunk_chunk.content)
                        if hunk_tokens <= self.chunk_size:
                            # Hunk chunk is reasonable size
                            final_chunks.append(hunk_chunk)
                        else:
                            # Hunk chunk is still too large, apply token-based splitting
                            oversized_chunks = self._split_by_tokens(hunk_chunk.content, file_path)
                            # Update chunk types to indicate mixed strategy
                            for chunk in oversized_chunks:
                                chunk.chunk_type = 'hunk-token-based'
                            final_chunks.extend(oversized_chunks)
                    chunks.extend(final_chunks)
                else:
                    # Fallback to pure token-based splitting
                    token_chunks = self._split_by_tokens(file_content, file_path)
                    chunks.extend(token_chunks)

        return chunks

    def _split_by_files(self, diff: str) -> List[Tuple[Optional[str], str]]:
        """Split diff by files."""
        files = []
        current_file = []
        current_path = None

        lines = diff.split('\n')

        for line in lines:
            if line.startswith('diff --git'):
                # Start of new file
                if current_file:
                    files.append((current_path, '\n'.join(current_file)))

                current_file = [line]
                # Extract file path
                match = re.search(r'diff --git a/(.+?) b/', line)
                current_path = match.group(1) if match else None
            else:
                current_file.append(line)

        # Add last file
        if current_file:
            files.append((current_path, '\n'.join(current_file)))

        return files

    def _split_file_by_hunks(self, file_content: str, file_path: Optional[str]) -> List[ChunkInfo]:
        """Split a large file by hunks."""
        chunks = []
        lines = file_content.split('\n')

        # Keep file header
        header_lines = []
        content_start = 0
        found_hunk = False

        for i, line in enumerate(lines):
            if line.startswith('@@'):
                content_start = i
                found_hunk = True
                break
            header_lines.append(line)

        # Check if we found any hunks - this is the key fix
        if not found_hunk:
            return []  # No hunk markers found, not a proper git diff format

        header = '\n'.join(header_lines)

        # Split by hunks
        current_hunk = []
        hunks = []

        for line in lines[content_start:]:
            if line.startswith('@@') and current_hunk:
                # Start of new hunk, save current
                hunks.append('\n'.join(current_hunk))
                current_hunk = [line]
            else:
                current_hunk.append(line)

        # Add last hunk
        if current_hunk:
            hunks.append('\n'.join(current_hunk))

        # Create chunks from hunks
        current_chunk_lines = header_lines[:]
        current_size = len(header)

        for hunk in hunks:
            hunk_size = len(hunk)

            if current_size + hunk_size <= self.chunk_size:
                # Add hunk to current chunk
                current_chunk_lines.extend(hunk.split('\n'))
                current_size += hunk_size
            else:
                # Save current chunk and start new one
                if len(current_chunk_lines) > len(header_lines):
                    chunks.append(ChunkInfo(
                        content='\n'.join(current_chunk_lines),
                        file_path=file_path,
                        chunk_type='hunk',
                        size=current_size,
                        is_complete_file=False
                    ))

                # Start new chunk with header + current hunk
                current_chunk_lines = header_lines[:] + hunk.split('\n')
                current_size = len(header) + hunk_size

        # Add final chunk
        if len(current_chunk_lines) > len(header_lines):
            chunks.append(ChunkInfo(
                content='\n'.join(current_chunk_lines),
                file_path=file_path,
                chunk_type='hunk',
                size=current_size,
                is_complete_file=False
            ))

        return chunks

    def _split_by_size(self, content: str, file_path: Optional[str]) -> List[ChunkInfo]:
        """Fallback size-based splitting."""
        chunks = []
        lines = content.split('\n')

        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1  # +1 for newline

            if current_size + line_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(ChunkInfo(
                    content='\n'.join(current_chunk),
                    file_path=file_path,
                    chunk_type='size-based',
                    size=current_size,
                    is_complete_file=False
                ))

                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    overlap_lines = current_chunk[-self.chunk_overlap//50:]  # Rough estimation
                    current_chunk = overlap_lines + [line]
                    current_size = sum(len(l) + 1 for l in current_chunk)
                else:
                    current_chunk = [line]
                    current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        # Add final chunk
        if current_chunk:
            chunks.append(ChunkInfo(
                content='\n'.join(current_chunk),
                file_path=file_path,
                chunk_type='size-based',
                size=current_size,
                is_complete_file=False
            ))

        return chunks

    def _split_by_tokens(self, content: str, file_path: Optional[str]) -> List[ChunkInfo]:
        """Token-based splitting using accurate token counting."""
        chunks = []
        lines = content.split('\n')

        current_lines = []
        current_tokens = 0

        for line in lines:
            line_with_newline = line + '\n'
            line_tokens = self.token_counter.count_tokens(line_with_newline)

            # Check if adding this line would exceed the limit
            if current_tokens + line_tokens > self.chunk_size and current_lines:
                # Save current chunk
                chunk_content = '\n'.join(current_lines)
                chunks.append(ChunkInfo(
                    content=chunk_content,
                    file_path=file_path,
                    chunk_type='token-based',
                    size=len(chunk_content),
                    is_complete_file=False
                ))

                # Start new chunk with overlap
                if self.chunk_overlap > 0 and len(current_lines) > 1:
                    # Calculate overlap lines based on token count
                    overlap_lines = []
                    overlap_tokens = 0

                    # Add lines from the end until we reach overlap limit
                    for overlap_line in reversed(current_lines):
                        overlap_line_tokens = self.token_counter.count_tokens(overlap_line + '\n')
                        if overlap_tokens + overlap_line_tokens <= self.chunk_overlap:
                            overlap_lines.insert(0, overlap_line)
                            overlap_tokens += overlap_line_tokens
                        else:
                            break

                    current_lines = overlap_lines + [line]
                    current_tokens = overlap_tokens + line_tokens
                else:
                    current_lines = [line]
                    current_tokens = line_tokens
            else:
                current_lines.append(line)
                current_tokens += line_tokens

        # Add final chunk
        if current_lines:
            chunk_content = '\n'.join(current_lines)
            chunks.append(ChunkInfo(
                content=chunk_content,
                file_path=file_path,
                chunk_type='token-based',
                size=len(chunk_content),
                is_complete_file=False
            ))

        return chunks

    def chunks_to_documents(self, chunks: List[ChunkInfo]) -> List[Document]:
        """Convert ChunkInfo to LangChain Documents."""
        documents = []

        for i, chunk in enumerate(chunks):
            metadata = {
                'chunk_id': i,
                'file_path': chunk.file_path,
                'chunk_type': chunk.chunk_type,
                'size': chunk.size,
                'is_complete_file': chunk.is_complete_file
            }

            documents.append(Document(
                page_content=chunk.content,
                metadata=metadata
            ))

        return documents

    def get_chunking_stats(self, chunks: List[ChunkInfo]) -> dict:
        """Get statistics about the chunking process."""
        total_size = sum(chunk.size for chunk in chunks)
        file_chunks = len([c for c in chunks if c.chunk_type == 'file'])
        hunk_chunks = len([c for c in chunks if c.chunk_type == 'hunk'])
        size_chunks = len([c for c in chunks if c.chunk_type == 'size-based'])
        token_chunks = len([c for c in chunks if c.chunk_type == 'token-based'])
        mixed_chunks = len([c for c in chunks if c.chunk_type == 'hunk-size-based'])
        mixed_token_chunks = len([c for c in chunks if c.chunk_type == 'hunk-token-based'])

        # Calculate token statistics
        total_tokens = sum(self.token_counter.count_tokens(chunk.content) for chunk in chunks)
        avg_tokens_per_chunk = total_tokens // len(chunks) if chunks else 0

        return {
            'total_chunks': len(chunks),
            'total_size': total_size,
            'total_tokens': total_tokens,
            'file_chunks': file_chunks,
            'hunk_chunks': hunk_chunks,
            'size_based_chunks': size_chunks,
            'token_based_chunks': token_chunks,
            'mixed_hunk_size_chunks': mixed_chunks,
            'mixed_hunk_token_chunks': mixed_token_chunks,
            'average_chunk_size': total_size // len(chunks) if chunks else 0,
            'average_tokens_per_chunk': avg_tokens_per_chunk,
            'complete_files': len([c for c in chunks if c.is_complete_file])
        }