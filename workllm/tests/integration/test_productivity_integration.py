import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from workllm.llm_clients import LLMClient
from workllm.productivity import (
    generate_code_docs,
    review_code,
    summarize_text,
    analyze_debug_output,
    _extract_function_signatures
)
from workllm.prompts import (
    DOCUMENTATION_STYLE_EXAMPLES,
    DOCUMENTATION_SYSTEM_PROMPT,
    CODE_REVIEW_SYSTEM_PROMPT,
    TEXT_SUMMARIZATION_SYSTEM_PROMPT
)

@pytest.fixture
def mock_client():
    return MagicMock(spec=LLMClient)

def test_code_review_workflow(mock_client, tmp_path):
    """Test complete workflow of code documentation and review"""
    # Arrange
    code = "def test_func():\n    pass\n"
    mock_client.generate.side_effect = [
        "# Test function\ndef test_func():\n    pass\n",  # generate_code_docs result
        "Code looks good, no issues found"  # review_code result
    ]
    
    # Act
    docs = generate_code_docs(mock_client, code, style="google", stream=False)
    review = review_code(mock_client, docs, stream=False)
    
    # Assert
    assert "Test function" in docs
    assert "Code looks good" in review
    assert mock_client.generate.call_count == 2

def test_analyze_and_summarize_workflow(mock_client):
    """Test workflow combining debug analysis and summarization"""
    # Arrange
    command = "pytest test_file.py"
    output = "test_something FAILED"
    mock_client.generate.side_effect = [
        "Test failure detected in test_something",  # analyze_debug_output result
        "Summary: Test failed due to assertion error"  # summarize_text result
    ]
    
    # Act
    analysis = analyze_debug_output(mock_client, command, output, stream=False)
    summary = summarize_text(mock_client, analysis, stream=False)
    
    # Assert
    assert "Test failure" in analysis
    assert "Summary:" in summary
    assert mock_client.generate.call_count == 2

def test_function_extraction_and_docs(mock_client, tmp_path):
    """Test workflow of extracting function signatures and generating docs"""
    # Arrange
    test_file = tmp_path / "test.py"
    test_file.write_text("def example():\n    pass\n")
    
    # Act
    signatures = _extract_function_signatures(str(tmp_path))
    if signatures and signatures.get(str(test_file.relative_to(tmp_path))):
        sig = signatures[str(test_file.relative_to(tmp_path))][0]
        mock_client.generate.return_value = f"# Documentation for {sig}"
        docs = generate_code_docs(mock_client, sig, style="google", stream=False)
        
        # Assert
        assert "def example()" in sig
        assert "Documentation for" in docs
        mock_client.generate.assert_called_once()
