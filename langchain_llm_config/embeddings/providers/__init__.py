"""
嵌入提供者实现
"""

from typing import TYPE_CHECKING, Optional, Type

from .openai import OpenAIEmbeddingProvider
from .vllm import VLLMEmbeddingProvider

# Optional imports with proper typing
if TYPE_CHECKING:
    from .gemini import GeminiEmbeddingProvider
    from .infinity import InfinityEmbeddingProvider
else:
    try:
        from .gemini import GeminiEmbeddingProvider
    except ImportError:
        GeminiEmbeddingProvider = None  # type: ignore[misc,assignment]

    try:
        from .infinity import InfinityEmbeddingProvider
    except ImportError:
        InfinityEmbeddingProvider = None  # type: ignore[misc,assignment]

# Build __all__ list dynamically
__all__ = [
    "OpenAIEmbeddingProvider",
    "VLLMEmbeddingProvider",
]

if GeminiEmbeddingProvider is not None:
    __all__.append("GeminiEmbeddingProvider")
if InfinityEmbeddingProvider is not None:
    __all__.append("InfinityEmbeddingProvider")
