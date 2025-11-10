"""Accurate token counting using tiktoken."""

import tiktoken
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class TokenStats:
    """Token counting statistics."""
    text_length: int
    token_count: int
    tokens_per_char: float
    model_used: str


class TokenCounter:
    """Accurate token counter using tiktoken."""

    # Model to encoding mapping
    MODEL_ENCODINGS = {
        # OpenAI models
        'gpt-4': 'cl100k_base',
        'gpt-4-turbo': 'cl100k_base',
        'gpt-4o': 'o200k_base',
        'gpt-4o-mini': 'o200k_base',
        'gpt-3.5-turbo': 'cl100k_base',
        'text-embedding-3-small': 'cl100k_base',
        'text-embedding-3-large': 'cl100k_base',

        # Anthropic models (use OpenAI compatible encoding as approximation)
        'claude-3-sonnet': 'cl100k_base',
        'claude-3-haiku': 'cl100k_base',
        'claude-3-opus': 'cl100k_base',
        'claude-3.5-sonnet': 'cl100k_base',

        # Fallback
        'default': 'cl100k_base'
    }

    def __init__(self, model_name: str = "gpt-4o"):
        """Initialize token counter for specific model."""
        self.model_name = model_name
        self.encoding_name = self._get_encoding_name(model_name)

        try:
            self.encoding = tiktoken.get_encoding(self.encoding_name)
        except Exception:
            # Fallback to default encoding
            self.encoding = tiktoken.get_encoding('cl100k_base')
            self.encoding_name = 'cl100k_base'

    def _get_encoding_name(self, model_name: str) -> str:
        """Get appropriate encoding name for model."""
        # Try exact match first
        if model_name in self.MODEL_ENCODINGS:
            return self.MODEL_ENCODINGS[model_name]

        # Try partial matches
        model_lower = model_name.lower()
        for model_key in self.MODEL_ENCODINGS:
            if model_key in model_lower or model_lower in model_key:
                return self.MODEL_ENCODINGS[model_key]

        # Default fallback
        return self.MODEL_ENCODINGS['default']

    def count_tokens(self, text: str) -> int:
        """Count tokens in text accurately."""
        if not text:
            return 0

        try:
            return len(self.encoding.encode(text))
        except Exception:
            # Fallback to rough estimation if encoding fails
            return len(text) // 4

    def get_token_stats(self, text: str) -> TokenStats:
        """Get detailed token statistics."""
        token_count = self.count_tokens(text)
        text_length = len(text)

        return TokenStats(
            text_length=text_length,
            token_count=token_count,
            tokens_per_char=token_count / text_length if text_length > 0 else 0,
            model_used=f"{self.model_name} ({self.encoding_name})"
        )

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to specific token count."""
        if not text:
            return text

        try:
            tokens = self.encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text

            # Truncate and decode
            truncated_tokens = tokens[:max_tokens]
            return self.encoding.decode(truncated_tokens)
        except Exception:
            # Fallback to character-based truncation
            estimated_chars = max_tokens * 4
            return text[:estimated_chars] if len(text) > estimated_chars else text

    def split_by_tokens(self, text: str, max_tokens: int, overlap: int = 0) -> list[str]:
        """Split text into chunks by token count."""
        if not text:
            return []

        try:
            tokens = self.encoding.encode(text)
            if len(tokens) <= max_tokens:
                return [text]

            chunks = []
            start = 0

            while start < len(tokens):
                end = min(start + max_tokens, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = self.encoding.decode(chunk_tokens)
                chunks.append(chunk_text)

                # Move start position with overlap
                start = end - overlap
                if start >= len(tokens):
                    break

            return chunks
        except Exception:
            # Fallback to character-based splitting
            estimated_chars = max_tokens * 4
            overlap_chars = overlap * 4

            chunks = []
            start = 0

            while start < len(text):
                end = min(start + estimated_chars, len(text))
                chunks.append(text[start:end])
                start = end - overlap_chars
                if start >= len(text):
                    break

            return chunks

    def estimate_cost(self, text: str, input_cost_per_1k: float = 0.0, output_cost_per_1k: float = 0.0) -> dict:
        """Estimate API cost based on token count."""
        token_count = self.count_tokens(text)

        return {
            'tokens': token_count,
            'input_cost': (token_count / 1000) * input_cost_per_1k,
            'output_cost': (token_count / 1000) * output_cost_per_1k,
            'total_cost': (token_count / 1000) * (input_cost_per_1k + output_cost_per_1k)
        }

    def is_within_limit(self, text: str, max_tokens: int) -> bool:
        """Check if text is within token limit."""
        return self.count_tokens(text) <= max_tokens

    @classmethod
    def create_for_model(cls, model_name: str) -> 'TokenCounter':
        """Factory method to create counter for specific model."""
        return cls(model_name)