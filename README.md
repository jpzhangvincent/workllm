# WorkLLM - LLM-Powered Productivity Toolkit

WorkLLM is a command-line productivity toolkit powered by Large Language Models (LLMs) designed to help engineers and developers boost their productivity through AI-assisted workflows.

## Features

### Core Functionality
- **Code Review**: Analyze and review code files with LLM-powered suggestions
- **Text Summarization**: Generate concise summaries from text, clipboard content, or web pages
- **Debugging Assistant**: Analyze command output and provide debugging insights
- **RAG Integration**: Document ingestion and querying with Retrieval Augmented Generation

### Supported LLM Providers
- Ollama (default)
- OpenAI

## Installation

1. Ensure Python 3.12+ is installed
2. Install WorkLLM:
```bash
pip install workllm
```

## Usage

### Basic Commands

```bash
# Code review
workllm code-review --file path/to/file.py

# Text summarization
workllm summarize --text "paste"  # Summarize clipboard content
workllm summarize --text "https://example.com"  # Summarize web page

# Debugging
workllm debug --shell-command "python script.py"

# RAG Operations
workllm rag ingest documents/*.pdf  # Ingest documents
workllm rag query "search term"  # Query documents
```

### Configuration

Set up environment variables for LLM providers:

```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key"

# For Ollama
export OLLAMA_MODEL="llama3.2:3b"
```

## Development

### Project Structure

```
workllm/
├── cli.py          # Command-line interface
├── llm_clients.py  # LLM client implementations
├── productivity.py # Productivity utilities
├── rag.py          # RAG functionality
└── utils.py        # Common utilities
```

### Running Tests

```bash
pytest
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License
