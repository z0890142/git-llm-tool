"""Smart chunking strategies for git diffs."""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
from langchain_core.documents import Document


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

    def __init__(self, chunk_size: int = 10000, chunk_overlap: int = 300):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

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
            if len(file_content) <= self.chunk_size:
                # File fits in one chunk
                chunks.append(ChunkInfo(
                    content=file_content,
                    file_path=file_path,
                    chunk_type='file',
                    size=len(file_content),
                    is_complete_file=True
                ))
            else:
                # File is too large, try to split by hunks
                hunk_chunks = self._split_file_by_hunks(file_content, file_path)
                if hunk_chunks:
                    # Check if any hunk chunks are still too large
                    final_chunks = []
                    for hunk_chunk in hunk_chunks:
                        if len(hunk_chunk.content) <= self.chunk_size:
                            # Hunk chunk is reasonable size
                            final_chunks.append(hunk_chunk)
                        else:
                            # Hunk chunk is still too large, apply size-based splitting
                            oversized_chunks = self._split_by_size(hunk_chunk.content, file_path)
                            # Update chunk types to indicate mixed strategy
                            for chunk in oversized_chunks:
                                chunk.chunk_type = 'hunk-size-based'
                            final_chunks.extend(oversized_chunks)
                    chunks.extend(final_chunks)
                else:
                    # Fallback to pure size-based splitting
                    size_chunks = self._split_by_size(file_content, file_path)
                    chunks.extend(size_chunks)

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
        mixed_chunks = len([c for c in chunks if c.chunk_type == 'hunk-size-based'])

        return {
            'total_chunks': len(chunks),
            'total_size': total_size,
            'file_chunks': file_chunks,
            'hunk_chunks': hunk_chunks,
            'size_based_chunks': size_chunks,
            'mixed_hunk_size_chunks': mixed_chunks,
            'average_chunk_size': total_size // len(chunks) if chunks else 0,
            'complete_files': len([c for c in chunks if c.is_complete_file])
        }