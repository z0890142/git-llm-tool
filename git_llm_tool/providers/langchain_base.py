"""LangChain base provider implementation."""

from abc import abstractmethod
from typing import Optional, List
from dataclasses import dataclass
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from halo import Halo
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.language_models import BaseLanguageModel
from langchain_ollama import OllamaLLM

from git_llm_tool.core.config import AppConfig
from git_llm_tool.core.exceptions import ApiError
from git_llm_tool.providers.base import LlmProvider
from git_llm_tool.core.rate_limiter import RateLimiter, RateLimitConfig
from git_llm_tool.core.diff_optimizer import DiffOptimizer
from git_llm_tool.core.smart_chunker import SmartChunker
from git_llm_tool.core.token_counter import TokenCounter


@dataclass
class ChunkStats:
    """Statistics about diff chunking process."""
    original_size: int
    num_chunks: int
    chunking_used: bool
    processing_time: float


class LangChainProvider(LlmProvider):
    """Base class for LangChain-based LLM providers."""

    def __init__(self, config: AppConfig, llm: Optional[BaseLanguageModel] = None):
        """Initialize the LangChain provider."""
        super().__init__(config)

        # Set the main LLM (provided by subclass or create one)
        self.llm = llm if llm is not None else self._create_llm()

        # Initialize Ollama LLM for chunk processing if enabled
        self.ollama_llm = None
        if config.llm.use_ollama_for_chunks:
            try:
                self.ollama_llm = OllamaLLM(
                    model=config.llm.ollama_model,
                    base_url=config.llm.ollama_base_url,
                    temperature=0.1
                )
            except Exception as e:
                # If Ollama is not available, fall back to main LLM
                print(f"⚠️ Ollama not available, using main LLM for chunks: {e}")
                self.ollama_llm = None

        # Always initialize rate limiter (simplified - always enabled)
        rate_config = RateLimitConfig(
            max_retries=config.llm._max_retries,
            initial_delay=config.llm._initial_delay,
            max_delay=config.llm._max_delay,
            backoff_multiplier=config.llm._backoff_multiplier,
            rate_limit_delay=config.llm._rate_limit_delay
        )
        self.rate_limiter = RateLimiter(rate_config)

        # Always initialize diff optimizer (simplified - always enabled)
        self.diff_optimizer = DiffOptimizer(
            max_context_lines=config.llm._max_context_lines
        )

        # Always initialize token counter
        self.token_counter = TokenCounter(config.llm.default_model)

        # Always initialize smart chunker (will be used based on threshold)
        self.smart_chunker = SmartChunker(
            chunk_size=config.llm._chunk_size,
            chunk_overlap=config.llm._chunk_overlap
        )

        # Keep fallback text splitter for edge cases
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.llm._chunk_size,
            chunk_overlap=config.llm._chunk_overlap,
            separators=[
                "\n\ndiff --git",  # Git diff file separators
                "\n@@",           # Git diff hunk separators
                "\n+",            # Added lines
                "\n-",            # Removed lines
                "\n",             # General newlines
                " ",              # Spaces
                "",               # Characters
            ]
        )

    @abstractmethod
    def _create_llm(self) -> BaseLanguageModel:
        """Create the specific LangChain LLM instance.

        Returns:
            Configured LangChain LLM instance

        Raises:
            ApiError: If LLM creation fails
        """
        pass

    def generate_commit_message(
        self,
        diff: str,
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate commit message using LangChain with optimizations."""
        start_time = time.time()

        try:
            # Step 1: Pre-optimize diff if enabled
            optimized_diff = diff
            # Simple decision: chunking threshold determines everything
            diff_tokens = self.token_counter.count_tokens(diff)
            will_chunk = diff_tokens > self.config.llm.chunking_threshold

            # Use lighter optimization if we will chunk (preserve diff structure)
            # Use more aggressive optimization if we won't chunk (save tokens)
            use_aggressive = not will_chunk

            optimized_diff, diff_stats = self.diff_optimizer.optimize_diff(
                diff,
                aggressive=use_aggressive
            )

            if kwargs.get("verbose", False):
                print(f"📊 Diff pre-optimization (aggressive={use_aggressive}):")
                print(f"   Original size: {diff_stats.original_size:,} chars")
                print(f"   Optimized size: {diff_stats.optimized_size:,} chars")
                print(f"   Compression ratio: {diff_stats.compression_ratio:.2f}")
                print(f"   Files processed: {diff_stats.files_processed}")

            # If not chunking and still too large, apply smart truncation using accurate token count
            current_tokens = self.token_counter.count_tokens(optimized_diff)
            if not will_chunk and current_tokens > self.config.llm._max_tokens:
                if kwargs.get("verbose", False):
                    print(f"   Current tokens: {current_tokens:,} (exceeds limit: {self.config.llm._max_tokens:,})")

                optimized_diff = self.token_counter.truncate_to_tokens(
                    optimized_diff,
                    self.config.llm._max_tokens
                )
                new_tokens = self.token_counter.count_tokens(optimized_diff)
                if kwargs.get("verbose", False):
                    print(f"   Truncated to: {new_tokens:,} tokens ({len(optimized_diff):,} chars)")

            # Step 2: Simple decision - use threshold to decide processing strategy
            optimized_tokens = self.token_counter.count_tokens(optimized_diff)
            should_chunk = optimized_tokens > self.config.llm.chunking_threshold

            if should_chunk:
                if kwargs.get("verbose", False):
                    print(f"🔄 Using intelligent chunking strategy for large diff")
                    print(f"   Diff tokens: {optimized_tokens:,} (threshold: {self.config.llm.chunking_threshold:,})")
                result = self._generate_with_smart_chunking(optimized_diff, jira_ticket, work_hours, **kwargs)
            else:
                if kwargs.get("verbose", False):
                    print(f"✨ Using direct processing")
                    print(f"   Diff tokens: {optimized_tokens:,} (under threshold: {self.config.llm.chunking_threshold:,})")
                result = self._generate_simple(optimized_diff, jira_ticket, work_hours, **kwargs)

            processing_time = time.time() - start_time
            if kwargs.get("verbose", False):
                print(f"⏱️  Total processing time: {processing_time:.2f}s")

            return result

        except Exception as e:
            raise ApiError(f"LangChain provider error: {e}")

    def _generate_simple(
        self,
        diff: str,
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate commit message without chunking."""
        prompt = self._build_commit_prompt(diff, jira_ticket, work_hours)

        def _make_llm_call():
            """Internal function to make the LLM call."""
            return self.llm.invoke(prompt, **kwargs)

        try:
            # Show progress for simple processing too
            with Halo(text="🤖 Generating commit message...", spinner="dots") as spinner:
                # Use rate limiter if available
                if self.rate_limiter:
                    response = self.rate_limiter.retry_with_backoff(_make_llm_call)
                else:
                    response = _make_llm_call()

                spinner.succeed("✅ Commit message generated successfully")

            # Handle different response types
            if hasattr(response, 'content'):
                return response.content.strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()

        except Exception as e:
            raise ApiError(f"Failed to generate commit message: {e}")

    def _generate_with_smart_chunking(
        self,
        diff: str,
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate commit message using smart chunking and rate limiting."""
        verbose = kwargs.get("verbose", False)

        try:
            # Show chunking progress
            with Halo(text="🔄 Analyzing diff and creating intelligent chunks...", spinner="dots") as spinner:
                chunk_infos = self.smart_chunker.chunk_diff(diff)
                docs = self.smart_chunker.chunks_to_documents(chunk_infos)
                spinner.succeed(f"✅ Created {len(docs)} intelligent chunks")

            if verbose:
                stats = self.smart_chunker.get_chunking_stats(chunk_infos)
                print(f"📄 Smart chunking stats:")
                print(f"   Total chunks: {stats['total_chunks']}")
                print(f"   File chunks: {stats['file_chunks']}")
                print(f"   Hunk chunks: {stats['hunk_chunks']}")
                print(f"   Size-based chunks: {stats['size_based_chunks']}")
                print(f"   Complete files: {stats['complete_files']}")
                print(f"   Average chunk size: {stats['average_chunk_size']:,} chars")

                # Show hybrid processing info
                if self.ollama_llm is not None:
                    print(f"🔄 Hybrid processing mode:")
                    print(f"   Map phase (chunks): Ollama ({self.config.llm.ollama_model})")
                    print(f"   Reduce phase (final): {self.config.llm.default_model}")
                else:
                    print(f"🔄 Standard processing mode:")
                    print(f"   All phases: {self.config.llm.default_model}")

            # Use manual map-reduce to avoid dict concatenation issues
            if len(docs) == 1:
                # Single chunk, use direct processing
                with Halo(text="🤖 Generating commit message for single chunk...", spinner="dots") as spinner:
                    prompt = self._build_commit_prompt(docs[0].page_content, jira_ticket, work_hours)

                    def _make_single_call():
                        return self.llm.invoke(prompt)

                    if self.rate_limiter:
                        response = self.rate_limiter.retry_with_backoff(_make_single_call)
                    else:
                        response = _make_single_call()

                    # Handle response
                    if hasattr(response, 'content'):
                        result = response.content.strip()
                    elif isinstance(response, str):
                        result = response.strip()
                    else:
                        result = str(response).strip()

                    spinner.succeed("✅ Commit message generated successfully")
                    return result
            else:
                # Multiple chunks, use manual map-reduce
                return self._manual_map_reduce(docs, jira_ticket, work_hours, **kwargs)

        except Exception as e:
            raise ApiError(f"Failed to generate commit message with smart chunking: {e}")

    def _manual_map_reduce(
        self,
        docs: List[Document],
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None,
        **kwargs
    ) -> str:
        """Manual map-reduce implementation with parallel processing."""
        try:
            # Improved logic: use parallel if Ollama is enabled OR we have multiple docs
            use_parallel = (self.ollama_llm is not None) or (len(docs) > 1)

            if use_parallel:
                return self._parallel_map_reduce(docs, jira_ticket, work_hours, **kwargs)
            else:
                return self._sequential_map_reduce(docs, jira_ticket, work_hours, **kwargs)

        except Exception as e:
            raise ApiError(f"Manual map-reduce failed: {e}")

    def _parallel_map_reduce(
        self,
        docs: List[Document],
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None,
        **kwargs
    ) -> str:
        """Parallel map-reduce implementation for faster processing."""
        verbose = kwargs.get("verbose", False)

        try:
            # Map phase: Process chunks in parallel
            summaries = [""] * len(docs)  # Pre-allocate to maintain order

            def process_chunk(index_doc_pair):
                """Process a single chunk."""
                i, doc = index_doc_pair
                try:
                    if kwargs.get("verbose", False):
                        print(f"📝 Processing chunk {i+1}/{len(docs)} in parallel...")

                    # Create map prompt for this chunk
                    map_prompt = self._create_simple_map_prompt(doc.page_content)

                    def _make_map_call():
                        # Use Ollama for chunk processing if available, otherwise use main LLM
                        if self.ollama_llm is not None:
                            return self.ollama_llm.invoke(map_prompt)
                        else:
                            return self.llm.invoke(map_prompt)

                    # Execute with rate limiting
                    if self.rate_limiter:
                        response = self.rate_limiter.retry_with_backoff(_make_map_call)
                    else:
                        response = _make_map_call()

                    # Extract text from response
                    if hasattr(response, 'content'):
                        summary = response.content.strip()
                    elif isinstance(response, str):
                        summary = response.strip()
                    else:
                        summary = str(response).strip()

                    if kwargs.get("verbose", False):
                        print(f"   ✅ Chunk {i+1} completed ({len(summary)} chars)")

                    return i, summary

                except Exception as e:
                    if kwargs.get("verbose", False):
                        print(f"   ❌ Chunk {i+1} failed: {e}")
                    return i, f"Error processing chunk: {str(e)}"

            # Execute parallel processing with configurable worker count
            if self.ollama_llm is not None:
                # Ollama is local, use configured Ollama concurrency
                max_workers = min(self.config.llm.ollama_max_parallel_chunks, len(docs))
            else:
                # Remote API, use configured remote API concurrency
                max_workers = min(self.config.llm.max_parallel_chunks, len(docs))
            completed_chunks = 0

            with Halo(text=f"🚀 Processing {len(docs)} chunks in parallel (0/{len(docs)} completed)...", spinner="dots") as spinner:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    future_to_index = {
                        executor.submit(process_chunk, (i, doc)): i
                        for i, doc in enumerate(docs)
                    }

                    # Collect results as they complete
                    for future in as_completed(future_to_index, timeout=self.config.llm.chunk_processing_timeout):
                        try:
                            index, summary = future.result()
                            summaries[index] = summary
                            completed_chunks += 1

                            # Update spinner text with progress
                            spinner.text = f"🚀 Processing {len(docs)} chunks in parallel ({completed_chunks}/{len(docs)} completed)..."

                            if verbose and not summary.startswith("Error"):
                                spinner.text += f" ✅ Chunk {index+1}"

                        except Exception as e:
                            # Handle individual chunk failures
                            index = future_to_index[future]
                            summaries[index] = f"Chunk processing failed: {str(e)}"
                            completed_chunks += 1

                            spinner.text = f"🚀 Processing {len(docs)} chunks in parallel ({completed_chunks}/{len(docs)} completed)..."
                            if verbose:
                                spinner.text += f" ❌ Chunk {index+1} failed"

                successful_chunks = len([s for s in summaries if not s.startswith("Error") and not s.startswith("Chunk processing failed")])
                spinner.succeed(f"✅ Parallel processing completed: {successful_chunks}/{len(docs)} chunks successful")

            # Reduce phase: Combine summaries into final commit message
            with Halo(text=f"🔄 Combining {len(summaries)} summaries into final commit message...", spinner="dots") as spinner:
                combined_summary = "\n\n".join([f"Part {i+1}: {summary}" for i, summary in enumerate(summaries)])
                combine_prompt = self._create_combine_prompt(combined_summary, jira_ticket, work_hours)

                def _make_combine_call():
                    return self.llm.invoke(combine_prompt)

                # Execute final combination with rate limiting
                if self.rate_limiter:
                    final_response = self.rate_limiter.retry_with_backoff(_make_combine_call)
                else:
                    final_response = _make_combine_call()

                spinner.succeed("✅ Final commit message generated successfully")

            # Extract final result
            if hasattr(final_response, 'content'):
                return final_response.content.strip()
            elif isinstance(final_response, str):
                return final_response.strip()
            else:
                return str(final_response).strip()

        except Exception as e:
            if kwargs.get("verbose", False):
                print(f"❌ Parallel processing failed, falling back to sequential: {e}")
            return self._sequential_map_reduce(docs, jira_ticket, work_hours, **kwargs)

    def _sequential_map_reduce(
        self,
        docs: List[Document],
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None,
        **kwargs
    ) -> str:
        """Sequential map-reduce implementation (fallback)."""
        try:
            # Map phase: Summarize each chunk sequentially
            summaries = []

            with Halo(text=f"⏳ Processing {len(docs)} chunks sequentially (0/{len(docs)} completed)...", spinner="dots") as spinner:
                for i, doc in enumerate(docs):
                    spinner.text = f"⏳ Processing chunk {i+1}/{len(docs)} sequentially..."

                    # Create map prompt for this chunk
                    map_prompt = self._create_simple_map_prompt(doc.page_content)

                    def _make_map_call():
                        # Use Ollama for chunk processing if available, otherwise use main LLM
                        if self.ollama_llm is not None:
                            return self.ollama_llm.invoke(map_prompt)
                        else:
                            return self.llm.invoke(map_prompt)

                    # Execute with rate limiting
                    if self.rate_limiter:
                        response = self.rate_limiter.retry_with_backoff(_make_map_call)
                    else:
                        response = _make_map_call()

                    # Extract text from response
                    if hasattr(response, 'content'):
                        summary = response.content.strip()
                    elif isinstance(response, str):
                        summary = response.strip()
                    else:
                        summary = str(response).strip()

                    summaries.append(summary)
                    spinner.text = f"⏳ Processing {len(docs)} chunks sequentially ({i+1}/{len(docs)} completed)..."

                spinner.succeed(f"✅ Sequential processing completed for {len(docs)} chunks")

            # Reduce phase: Combine summaries into final commit message
            with Halo(text=f"🔄 Combining {len(summaries)} summaries into final commit message...", spinner="dots") as spinner:
                combined_summary = "\n\n".join([f"Part {i+1}: {summary}" for i, summary in enumerate(summaries)])
                combine_prompt = self._create_combine_prompt(combined_summary, jira_ticket, work_hours)

                def _make_combine_call():
                    return self.llm.invoke(combine_prompt)

                # Execute final combination with rate limiting
                if self.rate_limiter:
                    final_response = self.rate_limiter.retry_with_backoff(_make_combine_call)
                else:
                    final_response = _make_combine_call()

                spinner.succeed("✅ Final commit message generated successfully")

            # Extract final result
            if hasattr(final_response, 'content'):
                return final_response.content.strip()
            elif isinstance(final_response, str):
                return final_response.strip()
            else:
                return str(final_response).strip()

        except Exception as e:
            raise ApiError(f"Sequential map-reduce failed: {e}")

    def _create_simple_map_prompt(self, chunk_content: str) -> str:
        """Create a simple prompt for analyzing a diff chunk."""
        return f"""Analyze this part of a git diff and summarize the changes in {self.config.llm.language}.

Focus on:
- What type of changes (feat, fix, docs, style, refactor, test, chore)
- Key modifications made
- Files or components affected

Git diff part:
{chunk_content}

Summary of changes in this part:"""

    def _create_combine_prompt(
        self,
        summaries: str,
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None
    ) -> str:
        """Create prompt for combining summaries into final commit message."""

        # Determine the output format based on Jira ticket
        if jira_ticket:
            format_instructions = f"""
Generate the commit message in this **exact format**:
{jira_ticket} <summary>""" + (f" #time {work_hours}" if work_hours else "") + """
- feat: detailed description of new features
- fix: detailed description of bug fixes
- docs: detailed description of documentation changes
(include only the types that apply to your changes)"""
        else:
            format_instructions = """
Generate the commit message in this **exact format**:
<summary>
- feat: description of new features
- fix: description of bug fixes
- docs: description of documentation changes
(include only the types that apply to your changes)"""

        return f"""Based on the following summaries of different parts of a git diff,
generate a concise commit message in {self.config.llm.language}.

**Conventional Commit types**:
- feat: new feature
- fix: bug fix
- docs: documentation changes
- style: formatting, missing semicolons, etc
- refactor: code restructuring without changing functionality
- test: adding or modifying tests
- chore: maintenance tasks

Part summaries:
{summaries}

{format_instructions}

Generate ONLY the commit message in the specified format, no additional text or explanation."""



    def generate_changelog(
        self,
        commit_messages: list[str],
        **kwargs
    ) -> str:
        """Generate changelog using LangChain."""
        prompt = self._build_changelog_prompt(commit_messages)

        try:
            response = self.llm.invoke(prompt, **kwargs)

            if hasattr(response, 'content'):
                return response.content.strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()

        except Exception as e:
            raise ApiError(f"Failed to generate changelog: {e}")

    def _should_chunk(self, text: str) -> bool:
        """Determine if text should be chunked based on token count and configuration."""
        text_tokens = self.token_counter.count_tokens(text)
        return text_tokens > self.config.llm.chunking_threshold

    def _make_api_call(self, prompt: str, **kwargs) -> str:
        """Make API call using LangChain's unified interface.

        This method is required by the base LlmProvider class but is handled
        internally by the LangChain LLM instances in this implementation.
        """
        try:
            response = self.llm.invoke(prompt, **kwargs)

            if hasattr(response, 'content'):
                return response.content.strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()

        except Exception as e:
            raise ApiError(f"LangChain API call failed: {e}")