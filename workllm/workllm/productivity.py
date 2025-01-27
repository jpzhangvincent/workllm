
from .llm_clients import LLMClient
from .prompts import (
    CODE_REVIEW_SYSTEM_PROMPT,
    DEBUG_ANALYSIS_SYSTEM_PROMPT,
    DOCUMENTATION_STYLE_EXAMPLES,
    DOCUMENTATION_SYSTEM_PROMPT,
    INTEGRATION_TEST_SYSTEM_PROMPT,
    TEST_FIX_SYSTEM_PROMPT,
    TEXT_SUMMARIZATION_SYSTEM_PROMPT,
    UNIT_TEST_SYSTEM_PROMPT,
)


def generate_code_docs(
    client: LLMClient,
    code: str,
    style: str = "google",
    focus: str | None = None,
    stream: bool = True,
    file_path: str | None = None,
    overwrite: bool = True
) -> str:
    """Generate documentation for code using an LLM client.

    Args:
        client: The LLM client to use for documentation generation
        code: The code to be documented
        style: Documentation style ('google' or 'sphinx')
        focus: Optional specific part to focus on ('class', 'function', or None for all)
        stream: Whether to stream the response or return it all at once

    Returns:
        The generated documentation for the code
    """
    focus_instruction = ""
    if focus:
        focus_instruction = f"Focus on documenting {focus}s in the code. "

    system_prompt = DOCUMENTATION_SYSTEM_PROMPT.format(
        style=style,
        focus_instruction=focus_instruction,
        style_example=DOCUMENTATION_STYLE_EXAMPLES[style]
    )

    documented_code = client.generate(
        prompt=f"Please document this code:\n{code}",
        system=system_prompt,
        stream=stream
    )

    if file_path and overwrite:
        try:
            with open(file_path, 'w') as f:
                f.write(documented_code)
        except Exception as e:
            print(f"Warning: Failed to overwrite file {file_path}: {str(e)}")

    return documented_code


def review_code(client: LLMClient, code: str, stream: bool = True) -> str:
    """Perform code review using an LLM client.

    Args:
        client: The LLM client to use for code review
        code: The code to be reviewed
        stream: Whether to stream the response or return it all at once

    Returns:
        The code review feedback from the LLM
    """
    system_prompt = CODE_REVIEW_SYSTEM_PROMPT

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
        system=TEXT_SUMMARIZATION_SYSTEM_PROMPT,
        stream=stream
    )

def _extract_function_signatures(directory: str) -> dict[str, list[str]]:
    """Extract function signatures from Python files in a directory.

    Args:
        directory: Path to the directory containing Python files

    Returns:
        Dictionary mapping file paths to lists of function signatures
    """
    import ast
    from pathlib import Path

    signatures = {}

    def get_signature(node: ast.FunctionDef) -> str:
        args = []
        for arg in node.args.args:
            annotation = ast.unparse(arg.annotation) if arg.annotation else 'Any'
            args.append(f"{arg.arg}: {annotation}")
        returns = ast.unparse(node.returns) if node.returns else 'None'
        return f"def {node.name}({', '.join(args)}) -> {returns}"

    for file in Path(directory).rglob("*.py"):
        try:
            with open(file) as f:
                tree = ast.parse(f.read())

            file_sigs = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    file_sigs.append(get_signature(node))

            if file_sigs:
                signatures[str(file.relative_to(directory))] = file_sigs
        except Exception as e:
            print(f"Warning: Failed to parse {file}: {str(e)}")

    return signatures

def add_tests(
    client: LLMClient,
    file_path: str,
    unit_tests: bool = True,
    integration_tests: bool = True,
    auto_fix: bool = False,
    max_iterations: int = 3,
    stream: bool = True
) -> tuple[str, str]:
    """Generate unit and integration tests for a given code file.

    Args:
        client: The LLM client to use for test generation
        file_path: Path to the source code file to generate tests for
        unit_tests: Whether to generate unit tests (default: True)
        integration_tests: Whether to generate integration tests (default: True)
        auto_fix: Whether to automatically fix failing tests using LLM (default: False)
        stream: Whether to stream the response or return it all at once

    Returns:
        A tuple containing (unit test code, integration test code)
    """
    import os
    import subprocess
    from pathlib import Path

    # Read the source file and get context
    with open(file_path) as f:
        source_code = f.read()

    # Get directory structure and function signatures for context
    source_dir = os.path.dirname(file_path)
    try:
        tree_output = subprocess.check_output(['tree', source_dir]).decode()
    except Exception:
        tree_output = "Failed to get directory structure"

    try:
        signatures = _extract_function_signatures(source_dir)
        signatures_text = "\n".join(
            f"File: {file}\nFunctions:\n" + "\n".join(f"  {sig}" for sig in sigs)
            for file, sigs in signatures.items()
        )
    except Exception:
        signatures_text = "Failed to extract function signatures"

    # Generate unit tests
    unit_test_code = ""
    if unit_tests:
        system_prompt = UNIT_TEST_SYSTEM_PROMPT

        prompt = f"""<source_code>
                {source_code}
                </source_code>

                <available_functions>
                {signatures_text}
                </available_functions>

                <project_structure>
                {tree_output}
                </project_structure>

                Generate unit tests for each function in the source code. Tests should verify individual function behavior in isolation."""

        unit_test_code = client.generate(
            prompt=prompt,
            system=system_prompt,
            stream=stream
        )

    # Generate integration tests
    integration_test_code = ""
    if integration_tests:
        system_prompt = INTEGRATION_TEST_SYSTEM_PROMPT

        prompt = f"""<source_code>
                {source_code}
                </source_code>

                <available_functions>
                {signatures_text}
                </available_functions>

                <project_structure>
                {tree_output}
                </project_structure>

                Generate integration tests focusing on interactions between functions. Tests should verify how functions work together in common workflows."""

        integration_test_code = client.generate(
            prompt=prompt,
            system=system_prompt,
            stream=stream
        )

    # Save tests in appropriate directories
    source_file = Path(file_path)
    test_dir = source_file.parent.parent / 'tests'

    # Create unit test directory
    unit_test_dir = test_dir / 'unit'
    unit_test_dir.mkdir(exist_ok=True, parents=True)
    unit_test_code = unit_test_code.replace(r".*```python", "").split("```")[0]

    # Create integration test directory
    integration_test_dir = test_dir / 'integration'
    integration_test_dir.mkdir(exist_ok=True, parents=True)
    integration_test_code = integration_test_code.replace(r".*```python", "").split("```")[0]

    # Create test files with proper paths
    if unit_tests:
        unit_test_path = unit_test_dir / f'test_{source_file.stem}.py'
        with open(unit_test_path, 'w') as f:
            # Add imports and path setup
            f.write(unit_test_code)

    if integration_tests:
        integration_test_path = integration_test_dir / f'test_{source_file.stem}.py'
        with open(integration_test_path, 'w') as f:
            f.write(integration_test_code)

    # Run tests and auto-fix if enabled
    try:
        iteration = 0
        while iteration < max_iterations:
            result = subprocess.run(['pytest', '-v', str(test_dir)],
                                 capture_output=True, text=True)

            if result.returncode == 0 or not auto_fix:
                break

            iteration += 1

            # If tests failed and auto_fix is enabled, use LLM to fix them
            system_prompt = TEST_FIX_SYSTEM_PROMPT

            prompt = f"""Test failure output:
            {result.stdout}
            {result.stderr}

            Current test code:
            Unit tests: {unit_test_code if unit_tests else 'N/A'}

            Integration tests: {integration_test_code if integration_tests else 'N/A'}

            Please fix the failing tests."""

            fixed_tests = client.generate(
                prompt=prompt,
                system=system_prompt,
                stream=stream
            )

            # Update test files with fixed code
            if unit_tests:
                with open(unit_test_path, 'w') as f:
                    f.write(fixed_tests)
            if integration_tests:
                with open(integration_test_path, 'w') as f:
                    f.write(fixed_tests)

    except Exception as e:
        print(f"Warning: Failed to run tests: {str(e)}")

    return unit_test_code, integration_test_code


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
    system_prompt = DEBUG_ANALYSIS_SYSTEM_PROMPT

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
