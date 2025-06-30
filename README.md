# Langchain LLM Config

Yet another redundent Langchain abstraction: comprehensive Python package for managing and using multiple LLM providers (OpenAI, VLLM, Gemini, Infinity) with a unified interface for both chat assistants and embeddings.

## Features

- 🤖 **Multiple Chat Providers**: Support for OpenAI, VLLM, and Gemini
- 🔗 **Multiple Embedding Providers**: Support for OpenAI, VLLM, and Infinity
- ⚙️ **Unified Configuration**: Single YAML configuration file for all providers
- 🚀 **Easy Setup**: CLI tool for quick configuration initialization
- 🔄 **Easy Context Concatenation**: Simplified process for combining contexts into chat
- 🔒 **Environment Variables**: Secure API key management
- 📦 **Self-Contained**: No need to import specific paths

## Installation

### Using pip
```bash
pip install langchain-llm-config
```

### Using uv (recommended)
```bash
uv add langchain-llm-config
```

### Development installation
```bash
git clone https://github.com/liux2/Langchain-LLM-Config.git
cd langchain-llm-config
uv sync --dev
uv run pip install -e .
```

## Quick Start

### 1. Initialize Configuration

```bash
# Initialize config in current directory
llm-config init

# Or specify a custom location
llm-config init ~/.config/api.yaml
```

This creates an `api.yaml` file with all supported providers configured.

### 2. Configure Your Providers

Edit the generated `api.yaml` file with your API keys and settings:

```yaml
llm:
  openai:
    chat:
      api_base: "https://api.openai.com/v1"
      api_key: "${OPENAI_API_KEY}"
      model_name: "gpt-3.5-turbo"
      temperature: 0.7
      max_tokens: 8192
    embeddings:
      api_base: "https://api.openai.com/v1"
      api_key: "${OPENAI_API_KEY}"
      model_name: "text-embedding-ada-002"
  
  vllm:
    chat:
      api_base: "http://localhost:8000/v1"
      api_key: "${OPENAI_API_KEY}"
      model_name: "meta-llama/Llama-2-7b-chat-hf"
      temperature: 0.6
  
  default:
    chat_provider: "openai"
    embedding_provider: "openai"
```

### 3. Set Environment Variables

Create a `.env` file or set environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
```

### 4. Use in Your Code

```python
from langchain_llm_config import create_assistant, create_embedding_provider
from pydantic import BaseModel, Field
from typing import List

# Define your response model
class ArticleAnalysis(BaseModel):
    summary: str = Field(..., description="Article summary")
    keywords: List[str] = Field(..., description="Key topics")
    sentiment: str = Field(..., description="Overall sentiment")

# Create an assistant
assistant = create_assistant(
    response_model=ArticleAnalysis,
    system_prompt="You are a helpful article analyzer.",
    provider="openai"  # or "vllm", "gemini"
)

# Use the assistant
result = await assistant.ask("Analyze this article: ...")
print(result.summary)

# Create an embedding provider
embedding_provider = create_embedding_provider(provider="openai")

# Get embeddings
texts = ["Hello world", "How are you?"]
embeddings = await embedding_provider.embed_texts(texts)
```

## Supported Providers

### Chat Providers

| Provider | Models | Features |
|----------|--------|----------|
| **OpenAI** | GPT-3.5, GPT-4, etc. | Streaming, function calling, structured output |
| **VLLM** | Any HuggingFace model | Local deployment, high performance |
| **Gemini** | Gemini Pro, etc. | Google's latest models |

### Embedding Providers

| Provider | Models | Features |
|----------|--------|----------|
| **OpenAI** | text-embedding-ada-002, etc. | High quality, reliable |
| **VLLM** | BGE, sentence-transformers | Local deployment |
| **Infinity** | Various embedding models | Fast inference |

## CLI Commands

```bash
# Initialize a new configuration file
llm-config init [path]

# Validate existing configuration
llm-config validate [path]

# Show package information
llm-config info
```

## Advanced Usage

### Custom Configuration Path

```python
from langchain_llm_config import create_assistant

assistant = create_assistant(
    response_model=MyModel,
    config_path="/path/to/custom/api.yaml"
)
```

### Streaming Chat

```python
from langchain_llm_config import create_chat_streaming

streaming_chat = create_chat_streaming(
    provider="openai",
    system_prompt="You are a helpful assistant."
)

# Stream responses
async for chunk in streaming_chat.stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

### Direct Provider Usage

```python
from langchain_llm_config import VLLMAssistant, OpenAIEmbeddingProvider

# Use providers directly
vllm_assistant = VLLMAssistant(
    config={"api_base": "http://localhost:8000/v1", "model_name": "llama-2"},
    response_model=MyModel
)

openai_embeddings = OpenAIEmbeddingProvider(
    config={"api_key": "your-key", "model_name": "text-embedding-ada-002"}
)
```

## Configuration Reference

### Environment Variables

The package supports environment variable substitution in configuration:

```yaml
api_key: "${OPENAI_API_KEY}"  # Will be replaced with actual value
```

### Configuration Structure

```yaml
llm:
  provider_name:
    chat:
      api_base: "https://api.example.com/v1"
      api_key: "${API_KEY}"
      model_name: "model-name"
      temperature: 0.7
      max_tokens: 8192
      # ... other parameters
    embeddings:
      api_base: "https://api.example.com/v1"
      api_key: "${API_KEY}"
      model_name: "embedding-model"
      # ... other parameters
  default:
    chat_provider: "provider_name"
    embedding_provider: "provider_name"
```

## Development

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black .
uv run isort .
```

### Type Checking

```bash
uv run mypy .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- 📖 [Documentation](https://github.com/liux2/Langchain-LLM-Config#readme)
- 🐛 [Issue Tracker](https://github.com/liux2/Langchain-LLM-Config/issues)
- 💬 [Discussions](https://github.com/liux2/Langchain-LLM-Config/discussions)
