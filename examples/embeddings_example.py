"""
Embeddings examples for langchain-llm-config package

This example demonstrates:
1. Basic text embeddings
2. Batch embedding generation
3. Semantic similarity calculation
4. Different embedding providers (Kunlun BGE-M3, Qwen3, BGE-Large-ZH)
5. Async embedding generation
"""

import asyncio
import os
from typing import List

from langchain_llm_config import create_embedding_provider


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    try:
        from numpy import dot
        from numpy.linalg import norm

        result: float = float(dot(vec1, vec2) / (norm(vec1) * norm(vec2)))
        return result
    except ImportError:
        # Fallback implementation without numpy
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        result_fallback: float = float(dot_product / (norm1 * norm2))
        return result_fallback


async def basic_embeddings_example():
    """Basic embeddings example"""
    print("🔗 Basic Embeddings Example")
    print("=" * 60)

    # Create embedding provider (uses default from config)
    embedding_provider = create_embedding_provider()

    # Single text embedding
    text = "Artificial Intelligence is transforming the world"
    print(f"\n💬 Text: {text}")

    embedding = embedding_provider.embed_texts([text])[0]
    print(f"\n📊 Embedding:")
    print(f"   Dimensions: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")


async def batch_embeddings_example():
    """Batch embeddings generation"""
    print("\n📦 Batch Embeddings Generation")
    print("=" * 60)

    embedding_provider = create_embedding_provider()

    texts = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Natural language processing enables text understanding",
        "Computer vision helps machines see",
        "The weather is sunny today",
    ]

    print(f"\n💬 Generating embeddings for {len(texts)} texts...")
    embeddings = embedding_provider.embed_texts(texts)

    print(f"\n📊 Results:")
    print(f"   Generated {len(embeddings)} embeddings")
    print(f"   Each embedding has {len(embeddings[0])} dimensions")


async def semantic_similarity_example():
    """Semantic similarity calculation"""
    print("\n🔍 Semantic Similarity Example")
    print("=" * 60)

    embedding_provider = create_embedding_provider()

    texts = [
        "I love programming in Python",
        "Python is my favorite programming language",
        "The weather is nice today",
        "Machine learning is fascinating",
    ]

    print("\n💬 Texts:")
    for i, text in enumerate(texts, 1):
        print(f"   {i}. {text}")

    print("\n📊 Generating embeddings...")
    embeddings = embedding_provider.embed_texts(texts)

    print("\n🔗 Similarity Matrix:")
    print("     ", end="")
    for i in range(len(texts)):
        print(f"  {i+1}  ", end="")
    print()

    for i, emb1 in enumerate(embeddings):
        print(f"  {i+1}  ", end="")
        for emb2 in embeddings:
            similarity = cosine_similarity(emb1, emb2)
            print(f"{similarity:.2f} ", end="")
        print()

    print("\n💡 Observations:")
    sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
    sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])
    print(f"   Texts 1 & 2 (similar topics): {sim_1_2:.3f}")
    print(f"   Texts 1 & 3 (different topics): {sim_1_3:.3f}")


async def async_embeddings_example():
    """Async embeddings generation"""
    print("\n⚡ Async Embeddings Generation")
    print("=" * 60)

    embedding_provider = create_embedding_provider()

    texts = [
        "Artificial Intelligence",
        "Machine Learning",
        "Deep Learning",
        "Neural Networks",
    ]

    print(f"\n💬 Generating embeddings asynchronously for {len(texts)} texts...")
    embeddings = await embedding_provider.embed_texts_async(texts)

    print(f"\n📊 Results:")
    print(f"   Generated {len(embeddings)} embeddings")
    print(f"   Dimensions: {len(embeddings[0])}")


async def different_providers_example():
    """Embeddings with different providers"""
    print("\n🔌 Different Embedding Providers")
    print("=" * 60)

    text = "Hello, world!"

    # Kunlun BGE-M3
    print("\n1️⃣  Kunlun BGE-M3 Embeddings:")
    if os.getenv("KUNLUN_BEARER_TOKEN"):
        try:
            provider_kunlun = create_embedding_provider(model="kunlun-bge-m3")
            embedding = provider_kunlun.embed_texts([text])[0]
            print(f"   ✅ Dimensions: {len(embedding)}")
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
    else:
        print("   ⚠️  Skipped: KUNLUN_BEARER_TOKEN not set")

    # Kunlun Qwen3 Embedding
    print("\n2️⃣  Kunlun Qwen3 Embeddings:")
    if os.getenv("KUNLUN_BEARER_TOKEN"):
        try:
            provider_qwen = create_embedding_provider(model="kunlun-qwen3-embedding")
            embedding = provider_qwen.embed_texts([text])[0]
            print(f"   ✅ Dimensions: {len(embedding)}")
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
    else:
        print("   ⚠️  Skipped: KUNLUN_BEARER_TOKEN not set")

    # Kunlun BGE-Large-ZH
    print("\n3️⃣  Kunlun BGE-Large-ZH Embeddings:")
    if os.getenv("KUNLUN_BEARER_TOKEN"):
        try:
            provider_bge = create_embedding_provider(model="kunlun-bge-large-zh")
            embedding = provider_bge.embed_texts([text])[0]
            print(f"   ✅ Dimensions: {len(embedding)}")
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
    else:
        print("   ⚠️  Skipped: KUNLUN_BEARER_TOKEN not set")


async def use_case_document_search():
    """Use case: Document search with embeddings"""
    print("\n📚 Use Case: Document Search")
    print("=" * 60)

    embedding_provider = create_embedding_provider()

    # Document corpus
    documents = [
        "Python is a high-level programming language known for its simplicity",
        "JavaScript is primarily used for web development and runs in browsers",
        "Machine learning algorithms can learn from data without explicit programming",
        "Docker containers provide lightweight virtualization for applications",
        "Git is a distributed version control system for tracking code changes",
    ]

    # User query
    query = "What is a programming language for beginners?"

    print(f"\n💬 Query: {query}")
    print(f"\n📄 Searching through {len(documents)} documents...")

    # Generate embeddings
    doc_embeddings = embedding_provider.embed_texts(documents)
    query_embedding = embedding_provider.embed_texts([query])[0]

    # Calculate similarities
    similarities = [
        (i, cosine_similarity(query_embedding, doc_emb))
        for i, doc_emb in enumerate(doc_embeddings)
    ]

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    print("\n🔍 Top 3 Results:")
    for rank, (doc_idx, similarity) in enumerate(similarities[:3], 1):
        print(f"\n   {rank}. Similarity: {similarity:.3f}")
        print(f"      {documents[doc_idx]}")


async def main():
    """Run all embeddings examples"""
    print("🚀 Langchain LLM Config - Embeddings Examples")
    print("=" * 60)

    await basic_embeddings_example()
    await batch_embeddings_example()
    await semantic_similarity_example()
    await async_embeddings_example()
    await different_providers_example()
    await use_case_document_search()

    print("\n✅ All embeddings examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
