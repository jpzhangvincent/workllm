import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from workllm.llm_clients import LLMClient, get_default_client
from workllm.productivity import generate_code_docs, review_code, summarize_text, generate_mermaid_diagram

@pytest.fixture
def mock_client():
    return MagicMock(spec=LLMClient)

def test_generate_code_docs(mock_client):
    """Test generate_code_docs function with mocked client"""
    # Arrange
    code = "def hello_world():\n\tprint('Hello World!')\n"
    mock_client.generate.return_value = "# Hello World!\n"

    # Act
    result = generate_code_docs(
        client=mock_client,
        code=code,
        style="google",
        focus=None,
        stream=False
    )

    # Assert
    assert result == "# Hello World!\n"
    mock_client.generate.assert_called_once()

def test_review_code(mock_client):
    """Test review_code function with mocked client"""
    # Arrange
    code = "def hello_world():\n\tprint('Hello World!')\n"
    mock_client.generate.return_value = "Code looks good!"

    # Act
    result = review_code(mock_client, code, stream=False)

    # Assert
    assert result == "Code looks good!"
    mock_client.generate.assert_called_once()

def test_summarize_text(mock_client):
    """Test summarize_text function with mocked client"""
    # Arrange
    text = "This is a sample text to be summarized.\nIt has multiple lines."
    mock_client.generate.return_value = "Sample text summary"

    # Act
    result = summarize_text(mock_client, text, stream=False)

    # Assert
    assert result == "Sample text summary"
    mock_client.generate.assert_called_once()

def test_generate_mermaid_diagram():
    with tempfile.TemporaryDirectory() as tempdir:
        # Create sample Python files
        file1 = os.path.join(tempdir, "file1.py")
        file2 = os.path.join(tempdir, "file2.py")
        with open(file1, 'w') as f:
            f.write("import os\nimport sys\ndef func1():\n    pass\n")
        with open(file2, 'w') as f:
            f.write("from file1 import func1\ndef func2():\n    func1()\n")

        # Generate Mermaid diagram
        diagram = generate_mermaid_diagram(tempdir)
        expected_diagram = (
            "graph TD\n"
            "    file2 --> file1\n"
            "    file2 --> func1\n"
            "    file1 --> os\n"
            "    file1 --> sys\n"
        )
        assert diagram.strip() == expected_diagram.strip()
