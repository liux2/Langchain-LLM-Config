"""
Command Line Interface for Langchain LLM Config
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .config import init_config, get_default_config_path, load_config


def init_command(args: argparse.Namespace) -> int:
    """Initialize a new configuration file"""
    try:
        config_path = init_config(args.config_path)
        print(f"✅ Configuration file created at: {config_path}")
        print("\n📝 Next steps:")
        print("1. Edit the configuration file with your API keys and settings")
        print("2. Set up your environment variables (e.g., OPENAI_API_KEY)")
        print("3. Start using the package in your Python code")
        return 0
    except Exception as e:
        print(f"❌ Error creating configuration file: {e}")
        return 1


def validate_command(args: argparse.Namespace) -> int:
    """Validate an existing configuration file"""
    try:
        config_path = args.config_path or get_default_config_path()
        config = load_config(str(config_path))
        print(f"✅ Configuration file is valid: {config_path}")
        print(f"📊 Default chat provider: {config['default']['chat_provider']}")
        print(f"📊 Default embedding provider: {config['default']['embedding_provider']}")
        return 0
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return 1


def info_command(args: argparse.Namespace) -> int:
    """Show information about the package and supported providers"""
    print("🤖 Langchain LLM Config")
    print("=" * 50)
    print("\n📦 Supported Chat Providers:")
    print("  • OpenAI - GPT models via OpenAI API")
    print("  • VLLM - Local and remote VLLM servers")
    print("  • Gemini - Google Gemini models")
    
    print("\n🔗 Supported Embedding Providers:")
    print("  • OpenAI - text-embedding models")
    print("  • VLLM - Local embedding models")
    print("  • Infinity - Fast embedding inference")
    
    print("\n🚀 Quick Start:")
    print("  1. llm-config init                    # Initialize config file")
    print("  2. Edit api.yaml with your settings   # Configure providers")
    print("  3. pip install langchain-llm-config   # Install package")
    print("  4. Use in your code:")
    print("     from langchain_llm_config import create_assistant")
    
    return 0


def main() -> int:
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog="llm-config",
        description="Langchain LLM Config - Manage LLM provider configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  llm-config init                    # Initialize config in current directory
  llm-config init ~/.config/api.yaml # Initialize config in specific location
  llm-config validate                # Validate current config
  llm-config info                    # Show package information
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize a new configuration file")
    init_parser.add_argument(
        "config_path",
        nargs="?",
        help="Path where to create the configuration file (default: ./api.yaml)"
    )
    init_parser.set_defaults(func=init_command)
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration file")
    validate_parser.add_argument(
        "config_path",
        nargs="?",
        help="Path to configuration file to validate (default: ./api.yaml)"
    )
    validate_parser.set_defaults(func=validate_command)
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show package information")
    info_parser.set_defaults(func=info_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main()) 