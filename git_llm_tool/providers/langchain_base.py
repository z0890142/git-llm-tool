"""LangChain base provider implementation."""

from abc import abstractmethod
from typing import Optional, List, Any, Callable
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

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
        self.llm = llm if llm is not None else self._create_llm()
        self.ollama_llm = self._init_ollama()
        self.rate_limiter = self._init_rate_limiter()
        self.diff_optimizer = DiffOptimizer(max_context_lines=config.llm._max_context_lines)
        self.token_counter = TokenCounter(config.llm.default_model)
        self.smart_chunker = SmartChunker(
            chunk_size=config.llm.chunk_size,
            chunk_overlap=config.llm.chunk_overlap,
            model_name=config.llm.default_model
        )

    def _init_ollama(self) -> Optional[OllamaLLM]:
        """Initialize the Ollama LLM if configured."""
        if not self.config.llm.use_ollama_for_chunks:
            return None
        try:
            return OllamaLLM(
                model=self.config.llm.ollama_model,
                base_url=self.config.llm.ollama_base_url,
                temperature=0.1, top_p=0.9, top_k=40,
                num_predict=100, timeout=30.0
            )
        except Exception as e:
            print(f"⚠️ Ollama not available, using main LLM for chunks: {e}")
            return None

    def _init_rate_limiter(self) -> RateLimiter:
        """Initialize the rate limiter."""
        rate_config = RateLimitConfig(
            max_retries=self.config.llm._max_retries,
            initial_delay=self.config.llm._initial_delay,
            max_delay=self.config.llm._max_delay,
            backoff_multiplier=self.config.llm._backoff_multiplier,
            rate_limit_delay=self.config.llm._rate_limit_delay
        )
        return RateLimiter(rate_config)

    @abstractmethod
    def _create_llm(self) -> BaseLanguageModel:
        """Create the specific LangChain LLM instance."""
        pass

    def _extract_response_content(self, response: Any) -> str:
        """Extracts and strips string content from various LLM response types."""
        if hasattr(response, 'content'):
            return response.content.strip()
        if isinstance(response, str):
            return response.strip()
        return str(response).strip()

    def _log_verbose(self, message: str, verbose: bool):
        """Logs a message if verbose mode is enabled."""
        if verbose:
            print(message)

    def generate_commit_message(
        self,
        diff: str,
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate commit message using LangChain with optimizations."""
        start_time = time.time()
        verbose = kwargs.get("verbose", False)

        try:
            # Step 1: Optimize diff
            # A less aggressive optimization is used initially to preserve structure for chunking.
            optimized_diff, diff_stats = self.diff_optimizer.optimize_diff(diff, aggressive=False)
            optimized_tokens = self.token_counter.count_tokens(optimized_diff)

            self._log_verbose(
                f"📊 Diff pre-optimization (aggressive=False):\n"
                f"   Original size: {diff_stats.original_size:,} chars\n"
                f"   Optimized size: {diff_stats.optimized_size:,} chars\n"
                f"   Compression ratio: {diff_stats.compression_ratio:.2f}",
                verbose
            )

            # Step 2: Decide processing strategy based on token count
            if optimized_tokens > self.config.llm.chunking_threshold:
                self._log_verbose(
                    f"🔄 Using intelligent chunking strategy for large diff.\n"
                    f"   Diff tokens: {optimized_tokens:,} (threshold: {self.config.llm.chunking_threshold:,})",
                    verbose
                )
                result = self._generate_with_smart_chunking(optimized_diff, jira_ticket, work_hours, **kwargs)
            else:
                # If not chunking, we can be more aggressive with optimizations.
                optimized_diff, _ = self.diff_optimizer.optimize_diff(diff, aggressive=True)
                current_tokens = self.token_counter.count_tokens(optimized_diff)

                # Truncate if it's still too large for a single call.
                if current_tokens > self.config.llm._max_tokens:
                    self._log_verbose(f"   Tokens {current_tokens:,} exceed limit {self.config.llm._max_tokens:,}. Truncating...", verbose)
                    optimized_diff = self.token_counter.truncate_to_tokens(optimized_diff, self.config.llm._max_tokens)
                    self._log_verbose(f"   Truncated to: {self.token_counter.count_tokens(optimized_diff):,} tokens", verbose)

                self._log_verbose(
                    f"✨ Using direct processing.\n"
                    f"   Final diff tokens: {self.token_counter.count_tokens(optimized_diff):,}",
                    verbose
                )
                result = self._generate_simple(optimized_diff, jira_ticket, work_hours, **kwargs)

            self._log_verbose(f"⏱️  Total processing time: {time.time() - start_time:.2f}s", verbose)
            return result

        except Exception as e:
            raise ApiError(f"LangChain provider error: {e}") from e

    def _generate_simple(
        self,
        diff: str,
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate a commit message for a single piece of text (no chunking)."""
        prompt = self._build_commit_prompt(diff, jira_ticket, work_hours)
        
        def _make_llm_call():
            return self.llm.invoke(prompt, **kwargs)

        try:
            with Halo(text="🤖 Generating commit message...", spinner="dots") as spinner:
                response = self.rate_limiter.retry_with_backoff(_make_llm_call)
                spinner.succeed("✅ Commit message generated successfully")
            return self._extract_response_content(response)
        except Exception as e:
            raise ApiError(f"Failed to generate commit message: {e}") from e

    def _generate_with_smart_chunking(
        self,
        diff: str,
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate a commit message using a map-reduce strategy on intelligent chunks."""
        verbose = kwargs.get("verbose", False)
        try:
            with Halo(text="🔄 Analyzing diff and creating intelligent chunks...", spinner="dots") as spinner:
                chunk_infos = self.smart_chunker.chunk_diff(diff)
                docs = self.smart_chunker.chunks_to_documents(chunk_infos)
                spinner.succeed(f"✅ Created {len(docs)} intelligent chunks")

            if verbose:
                stats = self.smart_chunker.get_chunking_stats(chunk_infos)
                print(f"📄 Smart chunking stats:\n"
                      f"   Total chunks: {stats['total_chunks']}\n"
                      f"   File/Hunk/Size-based chunks: {stats['file_chunks']}/{stats['hunk_chunks']}/{stats['size_based_chunks']}\n"
                      f"   Average chunk size: {stats['average_chunk_size']:,} chars")
                
                mode = "Hybrid" if self.ollama_llm else "Standard"
                map_model = self.config.llm.ollama_model if self.ollama_llm else self.config.llm.default_model
                print(f"🔄 {mode} processing mode:\n"
                      f"   Map phase (chunks): {map_model}\n"
                      f"   Reduce phase (final): {self.config.llm.default_model}")

            if len(docs) == 1:
                self._log_verbose("Single chunk created, using direct generation.", verbose)
                return self._generate_simple(docs[0].page_content, jira_ticket, work_hours, **kwargs)
            
            return self._manual_map_reduce(docs, jira_ticket, work_hours, **kwargs)

        except Exception as e:
            raise ApiError(f"Failed to generate commit message with smart chunking: {e}") from e

    def _manual_map_reduce(
        self,
        docs: List[Document],
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None,
        **kwargs
    ) -> str:
        """Orchestrates the map-reduce process, choosing between parallel and sequential execution."""
        # Simple flag to control parallel execution. Can be tied to config.
        use_parallel = self.ollama_llm is not None or len(docs) > 1

        try:
            if use_parallel:
                return self._parallel_map_reduce(docs, jira_ticket, work_hours, **kwargs)
            return self._sequential_map_reduce(docs, jira_ticket, work_hours, **kwargs)
        except Exception as e:
            raise ApiError(f"Map-reduce process failed: {e}") from e

    def _process_chunk_for_map(self, doc: Document) -> str:
        """Processes a single document chunk for the map phase."""
        map_prompt = self._create_simple_map_prompt(doc.page_content)
        
        def _make_map_call():
            llm_for_chunk = self.ollama_llm or self.llm
            return llm_for_chunk.invoke(map_prompt)

        response = self.rate_limiter.retry_with_backoff(_make_map_call)
        return self._extract_response_content(response)

    def _parallel_map_reduce(
        self,
        docs: List[Document],
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None,
        **kwargs
    ) -> str:
        """Parallel map-reduce implementation."""
        verbose = kwargs.get("verbose", False)
        summaries = [""] * len(docs)
        
        if self.ollama_llm:
            max_workers = min(getattr(self.config.llm, "ollama_max_parallel_chunks", 2), len(docs))
        else:
            max_workers = min(getattr(self.config.llm, "max_parallel_chunks", 4), len(docs))

        print(f"🚀 Launching {max_workers} parallel workers for {len(docs)} chunks...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(self._process_chunk_for_map, doc): i for i, doc in enumerate(docs)}
            
            completed_count = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    summary = future.result()
                    summaries[index] = summary
                    self._log_verbose(f"✅ [Worker] Chunk {index + 1} done.", verbose)
                except Exception as e:
                    summaries[index] = f"⚠️ Error processing chunk {index + 1}: {e}"
                    self._log_verbose(f"❌ [Worker] Chunk {index + 1} failed: {e}", verbose)
                
                completed_count += 1
                print(f"Progress: {completed_count}/{len(docs)} ({completed_count / len(docs):.1%})")

        return self._reduce_phase(summaries, jira_ticket, work_hours)

    def _sequential_map_reduce(
        self,
        docs: List[Document],
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None,
        **kwargs
    ) -> str:
        """Sequential map-reduce implementation (fallback)."""
        summaries = []
        with Halo(text=f"⏳ Processing {len(docs)} chunks sequentially...", spinner="dots") as spinner:
            for i, doc in enumerate(docs):
                spinner.text = f"⏳ Processing chunk {i+1}/{len(docs)}..."
                try:
                    summary = self._process_chunk_for_map(doc)
                    summaries.append(summary)
                except Exception as e:
                    summaries.append(f"⚠️ Error processing chunk {i + 1}: {e}")
            spinner.succeed(f"✅ All {len(docs)} chunks processed.")
        
        return self._reduce_phase(summaries, jira_ticket, work_hours)

    def _reduce_phase(
        self,
        summaries: List[str],
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None
    ) -> str:
        """Combines summaries from the map phase into a final commit message."""
        successful_chunks = len([s for s in summaries if not s.startswith("⚠️")])
        print(f"✅ Map phase done: {successful_chunks}/{len(summaries)} chunks successful.")

        combined_summary = "\n\n".join(
            [f"Part {i+1}: {summary}" for i, summary in enumerate(summaries)]
        )
        combine_prompt = self._create_combine_prompt(combined_summary, jira_ticket, work_hours)

        def _make_combine_call():
            return self.llm.invoke(combine_prompt)

        with Halo(text="🔄 Combining summaries into the final message...", spinner="dots") as spinner:
            final_response = self.rate_limiter.retry_with_backoff(_make_combine_call)
            spinner.succeed("✅ Final commit message generated.")
        
        return self._extract_response_content(final_response)

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
        if jira_ticket:
            time_log = f" #time {work_hours}" if work_hours else ""
            format_instructions = f"""
Generate the commit message in this **exact format**:
{jira_ticket} <summary>{time_log}
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
            return self._extract_response_content(response)
        except Exception as e:
            raise ApiError(f"Failed to generate changelog: {e}") from e