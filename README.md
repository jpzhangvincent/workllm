# WorkLLM - LLM-Powered Productivity Toolkit

WorkLLM is a command-line productivity toolkit powered by Large Language Models (LLMs) designed to help engineers and developers boost their productivity through AI-assisted workflows.

## Features

### Core Functionality
- **Code Documentation**: Generate comprehensive documentation in Google, Sphinx, or NumPy style
- **Code Review**: Analyze and review code files or GitHub PRs with LLM-powered suggestions
- **Test Generation**: Create unit and integration tests with auto-fix capabilities
- **Text Summarization**: Generate concise summaries from text, clipboard content, or web pages
- **Debugging Assistant**: Analyze command output and provide debugging insights
- **RAG Integration**: 
  - Document ingestion and querying with Retrieval Augmented Generation
  - Interactive chat interface with context-aware responses
  - Automatic document processing (PDF, text, markdown and codebase)

### System Architecture
- **Modular Design**: Clean separation of concerns with specialized modules
- **Centralized Prompts**: All LLM system prompts managed in `prompts.py` for easy maintenance
- **Flexible LLM Integration**: Support for multiple LLM providers through abstracted client interfaces

### Supported LLM Providers
- Ollama (default)
- OpenAI (or OpenAI API Compatible)
- AWS Bedrock

## Installation

1. Install [uv](https://github.com/astral-sh/uv):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install WorkLLM:
```bash
uv pip install workllm
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for OpenAI models
- `GITHUB_TOKEN`: Required for GitHub integration
- `LLM_CLIENT`: Default LLM client (options: "ollama", "openai", "bedrock")

## Usage

### Code Documentation
```bash
# Generate documentation
workllm codedoc --file path/to/file.py --style google

# Focus on specific elements
workllm codedoc --file path/to/file.py --focus function
```

### Code Review

```bash
# Code review
workllm code-review --file path/to/file.py
# or
workllm code-review --pr owner/repo#number
```

### Test Generation

```bash
workllm addtests --file path/to/file.py [--unit/--no-unit] [--integration/--no-integration]
```

### Text Summarization

```bash
workllm summarize --text "text to summarize"
# or from clipboard
workllm summarize --text paste
# or from URL
workllm summarize --text https://example.com
```

### Debug Analysis

```bash
workllm debug --shell-command "command to analyze"
```

### RAG Commands

```bash
# Ingest documents
workllm rag ingest path/to/docs --collection_name documents_collection

# List collections
workllm rag list-collections

# Question Answering from documents
workllm rag query "your question" --collection_name documents_collection

# Interactive chat
workllm rag chat

# Delete collection
workllm rag delete-collection collection_name
```


### Generate Tests

```bash
# Generate both unit and integration tests
workllm addtests --file src/module.py

# Generate only unit tests with auto-fix
workllm addtests --file src/module.py --no-integration --auto-fix
```


## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License
