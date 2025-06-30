"""
Basic tests for langchain-llm-config package
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_llm_config import (
    create_assistant,
    create_embedding_provider,
    load_config,
    init_config,
    get_default_config_path,
)


class TestConfig:
    """Test configuration functions"""
    
    def test_get_default_config_path(self):
        """Test default config path resolution"""
        path = get_default_config_path()
        assert isinstance(path, Path)
        assert path.name == "api.yaml"
    
    def test_init_config(self, tmp_path):
        """Test configuration initialization"""
        config_path = tmp_path / "test_api.yaml"
        result_path = init_config(str(config_path))
        
        assert result_path.exists()
        assert result_path.name == "test_api.yaml"
        
        # Verify config structure
        config = load_config(str(result_path))
        assert "llm" in config
        assert "default" in config["llm"]
        assert "openai" in config["llm"]
        assert "vllm" in config["llm"]


class TestFactory:
    """Test factory functions"""
    
    @pytest.mark.asyncio
    async def test_create_assistant_mock(self):
        """Test assistant creation with mocked dependencies"""
        from pydantic import BaseModel, Field
        
        class TestResponse(BaseModel):
            result: str = Field(..., description="Test result")
        
        # Mock the assistant class
        mock_assistant = MagicMock()
        mock_assistant.ask = MagicMock(return_value=TestResponse(result="test"))
        
        with patch("langchain_llm_config.factory.Assistant", return_value=mock_assistant):
            with patch("langchain_llm_config.factory.load_config") as mock_load:
                mock_load.return_value = {
                    "default": {"chat_provider": "openai"},
                    "openai": {
                        "chat": {
                            "model_name": "gpt-3.5-turbo",
                            "api_key": "test-key",
                            "temperature": 0.7,
                            "max_tokens": 2000,
                        }
                    }
                }
                
                assistant = create_assistant(
                    response_model=TestResponse,
                    provider="openai"
                )
                
                assert assistant is not None
    
    @pytest.mark.asyncio
    async def test_create_embedding_provider_mock(self):
        """Test embedding provider creation with mocked dependencies"""
        # Mock the embedding provider class
        mock_provider = MagicMock()
        mock_provider.embed_texts = MagicMock(return_value=[[0.1, 0.2, 0.3]])
        
        with patch("langchain_llm_config.factory.OpenAIEmbeddingProvider", return_value=mock_provider):
            with patch("langchain_llm_config.factory.load_config") as mock_load:
                mock_load.return_value = {
                    "default": {"embedding_provider": "openai"},
                    "openai": {
                        "embeddings": {
                            "model_name": "text-embedding-ada-002",
                            "api_key": "test-key",
                        }
                    }
                }
                
                provider = create_embedding_provider(provider="openai")
                
                assert provider is not None


class TestImports:
    """Test that all imports work correctly"""
    
    def test_main_imports(self):
        """Test that main package imports work"""
        from langchain_llm_config import (
            create_assistant,
            create_chat_streaming,
            create_embedding_provider,
            load_config,
            init_config,
            get_default_config_path,
        )
        
        # Just verify imports work
        assert all([
            create_assistant,
            create_chat_streaming,
            create_embedding_provider,
            load_config,
            init_config,
            get_default_config_path,
        ])
    
    def test_provider_imports(self):
        """Test that provider imports work"""
        from langchain_llm_config import (
            VLLMAssistant,
            GeminiAssistant,
            OpenAIEmbeddingProvider,
            VLLMEmbeddingProvider,
            InfinityEmbeddingProvider,
        )
        
        # Just verify imports work
        assert all([
            VLLMAssistant,
            GeminiAssistant,
            OpenAIEmbeddingProvider,
            VLLMEmbeddingProvider,
            InfinityEmbeddingProvider,
        ])


if __name__ == "__main__":
    pytest.main([__file__]) 