"""LangChain base provider implementation."""

from abc import abstractmethod
from typing import Optional, List, Any
from dataclasses import dataclass

from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel

from git_llm_tool.core.config import AppConfig
from git_llm_tool.core.exceptions import ApiError
from git_llm_tool.providers.base import LlmProvider, PromptTemplates


@dataclass
class ChunkStats:
    """Statistics about diff chunking process."""
    original_size: int
    num_chunks: int
    chunking_used: bool
    processing_time: float


class LangChainProvider(LlmProvider):
    """Base class for LangChain-based LLM providers."""

    def __init__(self, config: AppConfig):
        """Initialize the LangChain provider."""
        super().__init__(config)
        self.llm = self._create_llm()

        # Initialize text splitter for chunking
        if config.llm.enable_chunking:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.llm.chunk_size,
                chunk_overlap=config.llm.chunk_overlap,
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
        """Generate commit message using LangChain with intelligent chunking."""
        import time
        start_time = time.time()

        try:
            # Check if we should use chunking
            should_chunk = (
                self.config.llm.enable_chunking and
                len(diff) > self.config.llm.chunking_threshold
            )

            if should_chunk:
                result = self._generate_with_chunking(diff, jira_ticket, work_hours, **kwargs)
                processing_time = time.time() - start_time

                # Log chunking stats for debugging
                if kwargs.get("verbose", False):
                    print(f"🔄 Used chunking: {len(diff)} chars -> multiple chunks")
                    print(f"⏱️  Processing time: {processing_time:.2f}s")

                return result
            else:
                result = self._generate_simple(diff, jira_ticket, work_hours, **kwargs)
                processing_time = time.time() - start_time

                if kwargs.get("verbose", False):
                    print(f"✨ Direct processing: {len(diff)} chars")
                    print(f"⏱️  Processing time: {processing_time:.2f}s")

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

        try:
            # Use LangChain's unified invoke interface
            response = self.llm.invoke(prompt, **kwargs)

            # Handle different response types
            if hasattr(response, 'content'):
                return response.content.strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()

        except Exception as e:
            raise ApiError(f"Failed to generate commit message: {e}")

    def _generate_with_chunking(
        self,
        diff: str,
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate commit message using document chunking and map-reduce."""
        try:
            # Split the diff into chunks
            docs = self.text_splitter.create_documents([diff])

            if kwargs.get("verbose", False):
                print(f"📄 Split diff into {len(docs)} chunks")

            # Create prompts for map-reduce chain
            map_prompt = self._create_map_prompt(jira_ticket, work_hours)
            combine_prompt = self._create_combine_prompt(jira_ticket, work_hours)

            # Create and run summarize chain
            chain = load_summarize_chain(
                self.llm,
                chain_type="map_reduce",
                map_prompt=map_prompt,
                combine_prompt=combine_prompt,
                verbose=kwargs.get("verbose", False)
            )

            result = chain.run(docs)
            return result.strip()

        except Exception as e:
            raise ApiError(f"Failed to generate commit message with chunking: {e}")

    def _create_map_prompt(
        self,
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None
    ) -> PromptTemplate:
        """Create prompt template for the map phase of chunking."""

        # Build context based on Jira ticket presence
        context_info = ""
        if jira_ticket:
            context_info = f"\n**Jira ticket**: {jira_ticket}"
            if work_hours:
                context_info += f"\n**Time spent**: {work_hours}"

        template = f"""Analyze this part of a git diff and summarize the changes in {self.config.llm.language}.

{context_info}

Focus on:
- What type of changes (feat, fix, docs, style, refactor, test, chore)
- Key modifications made
- Files or components affected

Git diff part:
{{text}}

Summary of changes in this part:"""

        return PromptTemplate.from_template(template)

    def _create_combine_prompt(
        self,
        jira_ticket: Optional[str] = None,
        work_hours: Optional[str] = None
    ) -> PromptTemplate:
        """Create prompt template for the combine phase of chunking."""

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

        template = f"""Based on the following summaries of different parts of a git diff,
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
{{text}}

{format_instructions}

Generate ONLY the commit message in the specified format, no additional text or explanation."""

        return PromptTemplate.from_template(template)

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
        """Determine if text should be chunked based on size and configuration."""
        return (
            self.config.llm.enable_chunking and
            len(text) > self.config.llm.chunking_threshold
        )

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