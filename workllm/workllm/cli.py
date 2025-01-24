import click
import os
from github import Github
from .llm_clients import OllamaClient, OpenAIClient
from .productivity import review_code, summarize_text, analyze_debug_output
from .utils import get_clipboard_content as _get_clipboard_content, safe_subprocess_run
from .rag import rag_group

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
        raise click.ClickException(f"Failed to fetch PR content: {str(e)}")

@click.group()
def cli():
    """WorkLLM - Productivity toolkit powered by LLMs"""

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
            raise click.ClickException("PR format must be owner/repo#number")
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
        from unstructured.partition.html import partition_html
        import requests
        try:
            response = requests.get(text)
            response.raise_for_status()
            elements = partition_html(text=response.text)
            text = "\n".join([str(el) for el in elements])
        except requests.RequestException as e:
            raise click.ClickException(f"Failed to fetch URL content: {str(e)}")
        except ImportError:
            raise click.ClickException("unstructured library required for URL processing")
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
    raise ValueError(f"Unknown provider {provider}")

cli.add_command(rag_group, name="rag")

if __name__ == "__main__":
    cli()
