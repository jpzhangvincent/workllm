import os
import subprocess

import click
from dotenv import load_dotenv
from github import Github
from pathlib import Path

from .llm_clients import BedrockClient, DeepSeekClient, OllamaClient, OpenAIClient, OpenRouterClient
from .productivity import add_tests, analyze_debug_output, generate_code_docs, review_code, summarize_text, generate_pr_description, generate_mermaid_diagram
from .rag import rag_group
from .utils import get_clipboard_content, safe_subprocess_run, get_sitemap_urls, get_docling_content

load_dotenv()

if os.getenv("LLM_CLIENT") and os.getenv("LLM_CLIENT_MODEL"):
    DEFAULT_MODEL = f"{os.getenv('LLM_CLIENT')}:{os.getenv('LLM_CLIENT_MODEL')}"
else:
    DEFAULT_MODEL = "ollama:llama3.2:3b"
click.echo(click.style(f"Selected LLM model: {DEFAULT_MODEL}", fg="green"))

def _get_github_client():
    """Initialize GitHub client using GITHUB_TOKEN from environment"""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise click.ClickException("GITHUB_TOKEN environment variable is required for GitHub integration")
    return Github(token)

def _get_pr_content(repo_name: str, pr_number: int):
    """Retrieve PR content from GitHub"""
    github = _get_github_client()
    try:
        repo = github.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        files = pr.get_files()
        content = ""
        for file in files:
            content += f"File: {file.filename}\n"
            if file.patch:
                content += file.patch + "\n\n"
        return content
    except Exception as e:
        raise click.ClickException(f"Failed to fetch PR content: {str(e)}") from e

@click.group()
def cli():
    """WorkLLM - Productivity toolkit powered by LLMs"""
    pass

@cli.command()
@click.option("--file", type=click.Path(exists=True), help="File to review")
@click.option("--pr", help="GitHub PR in format owner/repo#number")
@click.option("--model", default=DEFAULT_MODEL, help="LLM model to use")
@click.option("--stream/--no-stream", default=True, help="Enable streaming output")
def code_review(file, pr, model, stream):
    """Perform code review on a file or GitHub PR"""
    client = _get_client(model)

    if pr:
        try:
            repo_name, pr_number = pr.split("#")
            content = _get_pr_content(repo_name, int(pr_number))
        except ValueError:
            raise click.ClickException("PR format must be owner/repo#number") from None
    elif file:
        with open(file) as f:
            content = f.read()
    else:
        raise click.ClickException("Either --file or --pr must be specified")

    result = review_code(client, content, stream=stream)
    click.echo(result)

@cli.command()
@click.option("--target", help="a local path or website url to parse the content")
@click.option("--is_doc_url", is_flag=False, help="Whether to crawl all the documentation urls on the target")
@click.option("--save_path", default="", help="Save parsed content to a directory")
@click.option("--save_format", default="markdown", type=click.Choice(["markdown", "html", "json"], case_sensitive=False), help="Format to save the parsed content (markdown or html)")
def save_parsed_content(target, is_doc_url, save_path, save_format):
    """Parse a file or website and save the content"""
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    if is_doc_url:
        sitemap_urls = get_sitemap_urls(target)
        conv_results_iter = converter.convert_all(sitemap_urls)
        docs = []
        for result in conv_results_iter:
            if result.document:
                document = get_docling_content(result.document, save_format)
                docs.append(document)
        output = "\n".join(docs)       
    else:
        result = converter.convert(target)
        output = get_docling_content(result.document, save_format)
    
    if output and save_path:
        # Ensure the parent directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(output)
    elif output:
        click.echo(output)
    else:
        click.echo("No parsed content to save!")
    

@cli.command()
@click.option("--text", help="Text to summarize directly")
@click.option("--file", type=click.Path(exists=True), help="Local PDF file to summarize")
@click.option("--url", help="URL to document (supports arXiv PDFs)")
@click.option("--paste", is_flag=True, help="Use clipboard content")
@click.option("--model", default=DEFAULT_MODEL, help="LLM model to use")
@click.option("--stream/--no-stream", default=True, help="Enable/disable streaming output")
def summarize(text, file, url, paste, model, stream):
    """Generate summary from text, PDF file, URL (including arXiv), or clipboard"""
    client = _get_client(model)

    try:
        if paste:
            text = get_clipboard_content()
        if file or url:
            source = file if file else url
            result = summarize_text(client, source=source, stream=stream)
        elif text:
            result = summarize_text(client, text=text, stream=stream)
        else:
            raise click.ClickException("One of --text, --file, --url, or --paste must be specified")

        click.echo(result)
    except ValueError as e:
        raise click.ClickException(str(e))

@cli.command()
@click.option("--shell-command", required=True, help="Command to debug")
@click.option("--model", default=DEFAULT_MODEL, help="LLM model to use")
@click.option("--execute/--no-execute", default=True, help="Execute the command or use as-is")
def debug(shell_command, model, execute):
    """Analyze command output for debugging"""
    client = _get_client(model)
    if execute:
        output = safe_subprocess_run(shell_command)
    else:
        output = shell_command
    result = analyze_debug_output(client, shell_command, output)
    click.echo(result)

def _get_client(model_str: str):
    provider, _, model = model_str.partition(":")
    if provider == "ollama":
        return OllamaClient(model)
    elif provider == "openai":
        return OpenAIClient(model)
    elif provider == "openrouter":
        return OpenRouterClient(model)
    elif provider == "deepseek":
        return DeepSeekClient(model)
    elif provider == "bedrock":
        return BedrockClient(model)
    raise ValueError(f"Unknown provider {provider}")

@cli.command()
@click.option("--file", type=click.Path(exists=True), help="File to document")
@click.option("--style", type=click.Choice(['google', 'sphinx']), default='google', help="Documentation style")
@click.option("--focus", type=click.Choice(['class', 'function', None]), default=None, help="Focus on specific code elements")
@click.option("--model", default=DEFAULT_MODEL, help="LLM model to use")
@click.option("--stream/--no-stream", default=True, help="Enable streaming output")
@click.option("--overwrite/--no-overwrite", default=True, help="Overwrite the input file with documented code")
def codedoc(file, style, focus, model, stream, overwrite):
    """Generate documentation for Python code"""
    client = _get_client(model)

    if not file:
        raise click.ClickException("--file must be specified")

    with open(file) as f:
        content = f.read()

    result = generate_code_docs(
        client,
        content,
        style=style,
        focus=focus,
        stream=stream,
        file_path=file if overwrite else None,
        overwrite=overwrite
    )
    click.echo(result)

@cli.command()
@click.option("--file", type=click.Path(exists=True), required=True, help="Source file to generate tests for")
@click.option("--unit/--no-unit", default=True, help="Generate unit tests")
@click.option("--integration/--no-integration", default=True, help="Generate integration tests")
@click.option("--auto-fix/--no-auto-fix", default=False, help="Automatically fix failing tests using LLM")
@click.option("--max-iterations", default=3, type=int, help="Maximum auto-fix iterations")
@click.option("--model", default=DEFAULT_MODEL, help="LLM model to use")
@click.option("--stream/--no-stream", default=True, help="Enable streaming output")
def addtests(file, unit, integration, auto_fix, max_iterations, model, stream):
    """Generate unit and integration tests for Python code"""
    client = _get_client(model)

    unit_tests, integration_tests = add_tests(
        client,
        file_path=file,
        unit_tests=unit,
        integration_tests=integration,
        auto_fix=auto_fix,
        max_iterations=max_iterations,
        stream=stream
    )

    if unit:
        click.echo("\nGenerated unit tests:")
        click.echo(unit_tests)
    if integration:
        click.echo("\nGenerated integration tests:")
        click.echo(integration_tests)

@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--llm-fix", is_flag=True, help="Use LLM to fix remaining linting errors")
@click.option("--model", default=DEFAULT_MODEL, help="LLM model to use")
def code_check(path, llm_fix, model):
    """Run ruff linting and optionally fix remaining errors with LLM"""
    client = _get_client(model)

    # Run ruff check with fixes
    try:
        result = subprocess.run(
            ["ruff", "check", "--fix", "--unsafe-fixes", path],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            click.echo("Ruff found issues:")
            click.echo(result.stdout)

            if llm_fix:
                from .utils import fix_code
                fixed_code = fix_code(client, result.stdout)
                click.echo("\nLLM suggested fixes:")
                click.echo(fixed_code)
        else:
            click.echo("No linting issues found!")

    except subprocess.SubprocessError as e:
        raise click.ClickException(f"Failed to run ruff: {str(e)}") from e

@cli.command()
@click.option("--target-branch", default="master", help="Target branch to compare against")
@click.option("--output", type=click.Path(), help="Output markdown file path")
@click.option("--model", default=DEFAULT_MODEL, help="LLM model to use")
def pr_description(target_branch, output, model):
    """Generate concise PR description based on changes against target branch"""
    from pathlib import Path
    import subprocess

    try:
        # Get the current branch name
        current_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()

        # Get the diff between the current branch and the target branch
        diff = subprocess.check_output(
            ["git", "diff", target_branch], text=True
        )

        client = _get_client(model)
        description = generate_pr_description(client, diff)

        if output:
            Path(output).write_text(description)
            click.echo(f"PR description saved to {output}")
        else:
            click.echo(description)

    except subprocess.CalledProcessError as e:
        raise click.ClickException(f"Git error: {str(e)}")

@cli.command()
@click.option("--source", type=click.Path(exists=True, file_okay=False), required=True, help="Directory to generate Mermaid diagram for")
@click.option("--readme", type=click.Path(exists=True, dir_okay=False), help="Path to README.md for additional context")
@click.option("--use-llm", is_flag=True, help="Use LLM reasoning to extract key functions")
@click.option("--model", default=DEFAULT_MODEL, help="LLM model to use")
def generate_diagram(source, readme, use_llm, model):
    """Generate a Mermaid diagram based on the dependencies in a codebase
    
    Args:
        source: File or Directory containing the codebase
        readme: Path to README.md for additional context
        use_llm: Whether to use LLM for enhanced analysis
        model: LLM model to use
    """
    client = _get_client(model) if use_llm else None
    
    # Read README content if provided
    readme_content = None
    if readme:
        try:
            with open(readme) as f:
                readme_content = f.read()
        except Exception as e:
            raise click.ClickException(f"Failed to read README file: {str(e)}")
    
    diagram = generate_mermaid_diagram(
        source, 
        use_llm=use_llm, 
        client=client,
        readme_content=readme_content
    )
    click.echo(diagram)

cli.add_command(rag_group, name="rag")

if __name__ == "__main__":
    cli()
