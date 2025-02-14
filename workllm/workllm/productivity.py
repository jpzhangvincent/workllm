import os
import ast
import json
from pathlib import Path

from .llm_clients import LLMClient
from .prompts import (
    CODE_REVIEW_SYSTEM_PROMPT,
    DEBUG_ANALYSIS_SYSTEM_PROMPT,
    DOCUMENTATION_STYLE_EXAMPLES,
    DOCUMENTATION_SYSTEM_PROMPT,
    INTEGRATION_TEST_SYSTEM_PROMPT,
    PR_SUMMARY_PROMPT,
    TEST_FIX_SYSTEM_PROMPT,
    TEXT_SUMMARIZATION_SYSTEM_PROMPT,
    UNIT_TEST_SYSTEM_PROMPT,
)

def extract_function_dependencies(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=str(file_path))

    dependencies = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            dependencies.append(node.func.id)

    return dependencies

class CodeRelationshipExtractor(ast.NodeVisitor):
    def __init__(self):
        self.relationships = {
            'imports': [],
            'classes': {},
            'functions': {},
            'calls': []
        }
        self.current_class = None

    def visit_Import(self, node):
        for alias in node.names:
            self.relationships['imports'].append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        self.relationships['imports'].append(f"{node.module}.{node.names[0].name}")
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.current_class = node.name
        self.relationships['classes'][node.name] = {
            'methods': [],
            'inherits': [base.id for base in node.bases if isinstance(base, ast.Name)]
        }
        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
        func_info = {
            'calls': [],
            'params': [arg.arg for arg in node.args.args]
        }
        
        if self.current_class:
            self.relationships['classes'][self.current_class]['methods'].append(node.name)
        else:
            self.relationships['functions'][node.name] = func_info
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.relationships['calls'].append(node.func.id)
        self.generic_visit(node)

def extract_code_relationships(file_path):
    """Extract comprehensive code relationships including imports, classes, and functions."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Skip binary files and non-Python files
            if '\x00' in content or not content.strip():
                return {
                    'imports': [],
                    'classes': {},
                    'functions': {},
                    'calls': []
                }
            tree = ast.parse(content, filename=str(file_path))
            extractor = CodeRelationshipExtractor()
            extractor.visit(tree)
            return extractor.relationships
    except (SyntaxError, UnicodeDecodeError, ValueError):
        # Return empty relationships for files that can't be parsed
        return {
            'imports': [],
            'classes': {},
            'functions': {},
            'calls': []
        }

def generate_mermaid_diagram(source, use_llm=False, client=None, readme_content=None):
    """
    Generate a comprehensive Mermaid diagram of the codebase structure and relationships.

    Args:
        source (str): Path to the directory containing the Python codebase
        use_llm (bool): Whether to use LLM for enhanced analysis
        client (LLMClient): The LLM client to use for reasoning
        readme_content (str): Content from README.md for additional context

    Returns:
        str: Mermaid diagram as a string
    """
    relationships = {}
    
    # Extract relationships from all Python files
    for file in Path(source).rglob("*.py"):
        file_relationships = extract_code_relationships(file)
        relationships[str(file.relative_to(source))] = file_relationships

    # Enhanced analysis with LLM if enabled
    if use_llm and client:
        prompt = (
            "Analyze the following code relationships and generate a structured dependency graph. "
            "Consider the following project context from README.md:\n"
            f"{readme_content}\n\n"
            "Code relationships:\n"
            f"{relationships}\n\n"
            "Generate a comprehensive Mermaid diagram with key details focusing on:\n"
            "1. Key architectural components\n"
            "2. Core functional relationships\n"
            "3. Important data flows\n"
            "4. Focus on custom modules instead of third-party common libraries.\n"
            "5. Make sure the dependency order rendered properly.\n"
            "Only output the Mermaid code for an architecture diagram with entity relationships directly:"
        )        
        diagram = client.generate(
            prompt=prompt,
            system="You are a software architect analyzing code relationships.",
            stream=False
        )
    else:
        # Generate Mermaid diagram programmatically
        diagram = ["graph TD"]
        
        # Add modules and files
        for file, rels in relationships.items():
            module_name = file.replace('.py', '')
            diagram.append(f"    subgraph {module_name}")
            
            # Add classes
            for class_name, class_info in rels['classes'].items():
                diagram.append(f"        class {class_name}")
                if class_info['inherits']:
                    for parent in class_info['inherits']:
                        diagram.append(f"        {parent} <|-- {class_name}")
                for method in class_info['methods']:
                    diagram.append(f"        {class_name} : {method}()")
            
            # Add functions
            for func_name, func_info in rels['functions'].items():
                diagram.append(f"        function {func_name}({', '.join(func_info['params'])})")
                for call in func_info['calls']:
                    diagram.append(f"        {func_name} --> {call}")
            
            diagram.append("    end")
            
            # Add file-level dependencies
            for imp in rels['imports']:
                diagram.append(f"    {module_name} --> {imp}")
            for call in rels['calls']:
                diagram.append(f"    {module_name} --> {call}")
            diagram = "\n".join(diagram)
    return diagram

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

def summarize_text(client: LLMClient, source: str | None = None, text: str | None = None, stream: bool = True) -> str:
    """Generate a summary of the provided text or document using an LLM client.

    Args:
        client: The LLM client to use for summarization
        source: Optional source path or URL to a document (supports PDF files and arXiv URLs)
        text: Optional raw text to be summarized directly
        stream: Whether to stream the response or return it all at once

    Returns:
        The generated summary of the text

    Raises:
        ValueError: If neither source nor text is provided, or if docling conversion fails
    """
    if not source and not text:
        raise ValueError("Either source or text must be provided")

    if source:
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(source)
            text = result.document.export_to_markdown()
        except Exception as e:
            raise ValueError(f"Failed to convert document: {str(e)}")

    if not text:
        raise ValueError("Failed to extract text from source")

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

def generate_pr_description(client: LLMClient, diff: str, stream: bool = True) -> str:
    """Generate a comprehensive PR description using an LLM client.

    Args:
        client: The LLM client to use for generating the PR description
        diff: The code difference to analyze
        stream: Whether to stream the response or return it all at once

    Returns:
        The generated PR description
    """
    prompt = PR_SUMMARY_PROMPT.format(diff=diff)
    return client.generate(
        prompt=prompt,
        system="",
        stream=stream
    )
