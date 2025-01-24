from typing import Protocol
from .llm_clients import LLMClient

def review_code(client: LLMClient, code: str, stream: bool = True) -> str:
    """Perform code review using an LLM client.
    
    Args:
        client: The LLM client to use for code review
        code: The code to be reviewed
        stream: Whether to stream the response or return it all at once
        
    Returns:
        The code review feedback from the LLM
    """
    system_prompt = """You are a senior software engineer reviewing code. 
    Provide concise, actionable feedback focusing on:
    - Security vulnerabilities
    - Performance optimizations
    - Code smells and anti-patterns
    - Modern Python best practices
    - Type hinting improvements"""
    
    return client.generate(
        prompt=code,
        system=system_prompt,
        stream=stream
    )

def summarize_text(client: LLMClient, text: str, stream: bool = True) -> str:
    """Generate a summary of the provided text using an LLM client.
    
    Args:
        client: The LLM client to use for summarization
        text: The text to be summarized
        stream: Whether to stream the response or return it all at once
        
    Returns:
        The generated summary of the text
    """
    return client.generate(
        prompt=f"Provide a comprehensive summary of this text:\n{text}",
        system="You are a research assistant skilled at distilling key information.",
        stream=stream
    )

def analyze_debug_output(client: LLMClient, command: str, output: str, stream: bool = True) -> str:
    """Analyze command output for debugging purposes using an LLM client.
    
    Args:
        client: The LLM client to use for analysis
        command: The command that was executed
        output: The output from the command execution
        stream: Whether to stream the response or return it all at once
        
    Returns:
        The analysis and debugging suggestions from the LLM
    """
    system_prompt = """You are a senior software engineer debugging command output.
    Analyze the following command and its output to:
    1. Identify potential errors or issues
    2. Suggest possible fixes or workarounds
    3. Explain what might be causing the problem
    4. Provide relevant documentation links if available
    
    Be concise but thorough in your analysis."""
    
    prompt = f"""Command executed:
    {command}

    Command output:
    {output}

    Please analyze the output and provide debugging assistance:"""
    
    return client.generate(
        prompt=prompt,
        system=system_prompt,
        stream=stream
    )
