import os
import subprocess

import click
from dotenv import load_dotenv
from github import Github

from .llm_clients import BedrockClient, DeepSeekClient, OllamaClient, OpenAIClient, OpenRouterClient
from .productivity import add_tests, analyze_debug_output, generate_code_docs, review_code, summarize_text
from .rag import rag_group
from .utils import get_clipboard_content as _get_clipboard_content
from .utils import safe_subprocess_run

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
@click.option("--model", default="ollama:llama3.2:3b", help="LLM model to use")
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
@click.option("--text", help="Text to summarize, 'paste' for clipboard, or URL starting with http(s)://")
@click.option("--model", default="ollama:llama3.2:3b", help="LLM model to use")
def summarize(text, model):
    """Generate summary of text content from direct input, clipboard, or URL"""
    client = _get_client(model)
    if text == "paste":
        text = _get_clipboard_content()
    elif text and text.startswith(('http://', 'https://')):
        import requests
        from unstructured.partition.html import partition_html
        try:
            response = requests.get(text)
            response.raise_for_status()
            elements = partition_html(text=response.text)
            text = "\n".join([str(el) for el in elements])
        except requests.RequestException as e:
            raise click.ClickException(f"Failed to fetch URL content: {str(e)}") from e
        except ImportError:
            raise click.ClickException("unstructured library required for URL processing") from None
    result = summarize_text(client, text)
    click.echo(result)

@cli.command()
@click.option("--shell-command", required=True, help="Command to debug")
@click.option("--model", default="ollama:llama3.2:3b", help="LLM model to use")
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
@click.option("--model", default="ollama:llama3.2:3b", help="LLM model to use")
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
@click.option("--model", default="ollama:llama3.2:3b", help="LLM model to use")
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
@click.option("--model", default="ollama:llama3.2:3b", help="LLM model to use")
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

cli.add_command(rag_group, name="rag")

if __name__ == "__main__":
    cli()
