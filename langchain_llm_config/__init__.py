"""
Langchain LLM Config - A comprehensive LLM configuration package

This package provides a unified interface for working with multiple LLM providers
including OpenAI, VLLM, Gemini, and Infinity for both chat assistants and embeddings.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

# Import base classes for extensibility
from .assistant.base import Assistant
from .assistant.chat_streaming import ChatStreaming
from .assistant.multimodal import create_image_content, create_multimodal_query

# Import provider classes
from .assistant.providers.vllm import VLLMAssistant

# Import configuration functions
from .config import (
    get_default_config_path,
    init_config,
    load_config,
)
from .embeddings.base import BaseEmbeddingProvider
from .embeddings.providers.openai import OpenAIEmbeddingProvider
from .embeddings.providers.vllm import VLLMEmbeddingProvider

# Optional imports - only available if dependencies are installed
if TYPE_CHECKING:
    from .assistant.providers.gemini import GeminiAssistant
    from .embeddings.providers.infinity import InfinityEmbeddingProvider
else:
    try:
        from .assistant.providers.gemini import GeminiAssistant
    except ImportError:
        GeminiAssistant = None  # type: ignore[misc,assignment]

    try:
        from .embeddings.providers.infinity import InfinityEmbeddingProvider
    except ImportError:
        InfinityEmbeddingProvider = None  # type: ignore[misc,assignment]

# Import main factory functions
from .factory import (
    create_assistant,
    create_chat_streaming,
    create_embedding_provider,
)

# Get version from package metadata
try:
    from importlib.metadata import version

    __version__ = version("langchain-llm-config")
except ImportError:
    # Fallback for Python < 3.8
    __version__ = "0.1.6"
__author__ = "Xingbang Liu"
__email__ = "xingbangliu48@gmail.com"

# Define the tiktoken cache directory path
TIKTOKEN_CACHE_DIR = str(Path(__file__).parent / ".tiktoken_cache")

# Build __all__ list dynamically based on available imports
__all__ = [
    # Constants
    "TIKTOKEN_CACHE_DIR",
    # Factory functions
    "create_assistant",
    "create_chat_streaming",
    "create_embedding_provider",
    # Configuration functions
    "load_config",
    "init_config",
    "get_default_config_path",
    # Base classes
    "Assistant",
    "ChatStreaming",
    "BaseEmbeddingProvider",
    # Multimodal helper functions
    "create_image_content",
    "create_multimodal_query",
    # Provider classes (always available)
    "VLLMAssistant",
    "OpenAIEmbeddingProvider",
    "VLLMEmbeddingProvider",
]

# Add optional providers if available
if GeminiAssistant is not None:
    __all__.append("GeminiAssistant")
if InfinityEmbeddingProvider is not None:
    __all__.append("InfinityEmbeddingProvider")
