# WorkLLM - LLM-Powered Productivity Toolkit

WorkLLM is a command-line productivity toolkit powered by Large Language Models (LLMs) designed to help engineers and developers boost their productivity through AI-assisted workflows.

## Features

### Core Functionality
- **Code Documentation**: Generate comprehensive documentation in Google, Sphinx, or NumPy style
- **Code Review**: Analyze and review code files or GitHub PRs with LLM-powered suggestions
- **Test Generation**: Create unit and integration tests with auto-fix capabilities
- **Code Quality**: Run code linting with ruff and auto-fix issues using LLM suggestions
- **PR Summary**: Generate pull request descriptions from code changes
- **Diagram Creation**: Create a Mermaid architecture diagram from codebase or a file
- **Text Summarization**: Generate concise summaries from text, clipboard content, or web pages
- **Debugging Assistant**: Analyze command output and provide debugging insights
- **Chat with docs and codebase**: 
  - Document ingestion and question answering 
  - Interactive chat interface with Retrieval Augmented Generation (RAG)
  - Automatic document processing (PDF, text, markdown and codebase)
  - Hybrid search with semantic + keyword matching

### Supported LLM Providers
- Ollama (default - llama3.2:3b)
- OpenAI 
- Deepseek 
- OpenRouter 
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
- `LLM_CLIENT` and `LLM_CLIENT_MODEL`: Default LLM client (options: "ollama", "openai", "bedrock", "deepseek", "openrouter") and model choice

```bash
export LLM_CLIENT="ollama"
export LLM_CLIENT_MODEL="deepseek-r1:14b"
```

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

### Code Style Checker

```bash
# Code linting and formatting checker with `ruff`
workllm code-check path/to/file.py

# Code linting and formatting checker, and then use LLM to suggest fixes
workllm code-check --llm-fix path/to/file.py
```

### PR Description Generation

```bash 
# Generate PR description from changes against target branch
workllm pr-description --target-branch main

# Save PR description to a file
workllm pr-description --target-branch main --output pr.md
```

### Architecture Diagram

```bash
# Generate Mermaid diagram for a codebase
workllm generate-diagram --source src/

# Include README context and use LLM for enhanced analysis
workllm generate-diagram --source src/ --readme README.md --use-llm
```

### Test Generation

```bash
# Generate both unit and integration tests
workllm addtests --file src/module.py

# Generate only unit tests with auto-fix
workllm addtests --file src/module.py --no-integration --auto-fix
```

### Text Summarization

```bash
# arXiv papers: 
workllm summarize --url https://arxiv.org/pdf/2501.17811v1

# local PDF file: 
workllm summarize --file path/to/document.pdf

# direct text 
workllm summarize --text "Your text here"

#clipboard content: 
workllm summarize --paste
```

### Debug Analysis

```bash
workllm debug --shell-command "command to analyze"
```

### RAG Document Management & Chat

```bash
# Ingest local documents (PDF, text, markdown)
workllm rag ingest path/to/docs --collection_name documents_collection

# Ingest from a documentation website (e.g Langgraph)
workllm rag ingest --docs-url https://langchain-ai.github.io/langgraph --collection langgraph

# List all available collections
workllm rag list-collections

# Question answering with hybrid search (semantic + keyword)
workllm rag query "your question" --collection_name documents_collection

# Interactive chat with document context
workllm rag chat --collection documents_collection

# Delete a collection
workllm rag delete-collection collection_name
```

# TODO
- Agentic RAG
- More build-in Workflows/Agents


## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License
