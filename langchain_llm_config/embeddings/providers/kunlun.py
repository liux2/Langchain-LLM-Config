import asyncio
import time
from typing import Any, Dict, List

import httpx
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from ..base import BaseEmbeddingProvider

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None  # type: ignore[assignment,misc]


class KunlunEmbeddingProvider(BaseEmbeddingProvider):
    """Kunlun embedding provider (OpenAI-compatible with bearer token authentication)"""

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        """
        初始化Kunlun嵌入提供者

        Args:
            config: 配置字典，包含如下键:
                - model_name: 模型名称
                - api_base: API基础URL
                - bearer_token: Bearer token用于认证
                - dimensions: 嵌入维度（可选）
                - timeout: 超时时间（可选）
            **kwargs: 额外参数
        """
        if OpenAIEmbeddings is None:
            raise ImportError(
                "OpenAIEmbeddings not found. Please install langchain-openai: "
                "pip install langchain-openai"
            )

        # Validate bearer token
        bearer_token = config.get("bearer_token", "")
        if not bearer_token or bearer_token.strip() == "":
            raise ValueError(
                "Kunlun bearer token is required. "
                "Set KUNLUN_BEARER_TOKEN environment variable or provide it in config."
            )

        # Disable SSL verification for Kunlun platform (self-signed certificate)
        http_client = httpx.Client(verify=False)
        http_async_client = httpx.AsyncClient(verify=False)

        embedding_params = {
            "model": config["model_name"],
            "dimensions": config.get("dimensions"),
            "api_key": SecretStr(bearer_token),  # Use bearer token as api_key
            "base_url": config.get("api_base"),
            "timeout": config.get("timeout", 60),
            "http_client": http_client,
            "http_async_client": http_async_client,
        }

        # 记录初始化信息（隐藏敏感信息）
        safe_params = embedding_params.copy()
        if "api_key" in safe_params:
            safe_params["api_key"] = "******"

        # 添加其他kwargs
        embedding_params.update(kwargs)

        self._embeddings = OpenAIEmbeddings(**embedding_params)
        self._max_retries = 3
        self._retry_delay = 1.0  # 初始重试延迟（秒）

    @property
    def embedding_model(self) -> Embeddings:
        """获取嵌入模型"""
        return self._embeddings

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        嵌入文本列表（同步版本）

        Args:
            texts: 要嵌入的文本列表

        Returns:
            嵌入向量列表
        """
        if not texts:
            return []

        retry_count = 0
        last_error = None

        while retry_count < self._max_retries:
            try:
                result = self._embeddings.embed_documents(texts)
                return result
            except Exception as e:
                retry_count += 1
                last_error = e

                if retry_count < self._max_retries:
                    # 指数退避重试
                    wait_time = self._retry_delay * (2 ** (retry_count - 1))
                    time.sleep(wait_time)

        # 所有重试都失败，报告错误
        raise Exception(f"Kunlun嵌入文本失败: {str(last_error)}")

    async def embed_texts_async(self, texts: List[str]) -> List[List[float]]:
        """
        嵌入文本列表（异步版本）

        Args:
            texts: 要嵌入的文本列表

        Returns:
            嵌入向量列表
        """
        if not texts:
            return []

        retry_count = 0
        last_error = None

        while retry_count < self._max_retries:
            try:
                result = await self._embeddings.aembed_documents(texts)
                return result
            except Exception as e:
                retry_count += 1
                last_error = e

                if retry_count < self._max_retries:
                    # 指数退避重试
                    wait_time = self._retry_delay * (2 ** (retry_count - 1))
                    await asyncio.sleep(wait_time)

        # 所有重试都失败，报告错误
        raise Exception(f"Kunlun嵌入文本失败: {str(last_error)}")
