"""
使用示例：如何使用新的助手和嵌入提供者
"""

import asyncio
from typing import List
from pydantic import BaseModel, Field

from .factory import create_assistant, create_embedding_provider


# 示例响应模型
class ArticleAnalysisResponse(BaseModel):
    """文章分析响应"""

    summary: str = Field(..., description="文章摘要")
    keywords: List[str] = Field(..., description="关键词列表")
    sentiment: str = Field(..., description="情感分析")


async def assistant_example():
    """助手使用示例"""
    # 创建助手
    assistant = create_assistant(
        response_model=ArticleAnalysisResponse,
        system_prompt="你是一个专业的文章分析助手，可以提取摘要、关键词并进行情感分析。",
    )

    # 分析文章
    article = """
    人工智能正在迅速改变我们的世界。从自动驾驶汽车到智能助手，AI技术正在各个领域取得突破。
    近年来，深度学习和大型语言模型的发展尤为显著，它们在自然语言处理、计算机视觉和语音识别方面取得了惊人的进步。
    尽管如此，我们也必须关注AI发展带来的伦理问题和可能的社会影响。数据隐私、算法偏见和就业变化是需要解决的重要挑战。
    总体而言，如果谨慎发展和适当监管，AI有望为人类带来巨大福祉。
    """

    result = await assistant.ask(article)

    print("\n=== 文章分析结果 ===")
    print(f"摘要: {result['summary']}")
    print(f"关键词: {', '.join(result['keywords'])}")
    print(f"情感: {result['sentiment']}")


async def embedding_example():
    """嵌入提供者使用示例"""
    # 创建嵌入提供者
    embedding_provider = create_embedding_provider()

    # 嵌入文本
    texts = [
        "人工智能正在迅速改变我们的世界",
        "深度学习模型在各个领域取得了突破",
        "数据隐私是AI发展中的重要问题",
    ]

    # 使用新的embed_texts方法
    embeddings = await embedding_provider.embed_texts(texts)

    print("\n=== 嵌入结果 ===")
    print(f"嵌入了 {len(embeddings)} 个文本")
    print(f"每个嵌入向量维度: {len(embeddings[0])}")


if __name__ == "__main__":
    # 运行示例
    # asyncio.run(assistant_example())
    asyncio.run(embedding_example())
