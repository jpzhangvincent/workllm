import os
from pathlib import Path

import click
from unstructured.partition.pdf import partition_pdf

from .utils import ingest_doc, retrieve_doc

DEFAULT_COLLECTION = "workllm"
PERSIST_DIR = os.path.expanduser("~/.workllm/vectorstore")

@click.group()
def rag_group():
    """RAG document management commands"""

@rag_group.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option("--collection", default=DEFAULT_COLLECTION, help="Collection name")
def ingest(paths, collection):
    """Ingest documents into vector store"""
    for path in paths:
        path = Path(path)
        if path.is_dir():
            files = list(path.rglob("*"))
        else:
            files = [path]

        for file in files:
            if file.is_file() and file.suffix in (".txt", ".md", ".pdf"):
                if file.suffix.lower() == '.pdf':
                    elements = partition_pdf(str(file))
                    content = "\n".join([str(el) for el in elements])
                else:
                    content = file.read_text()

                ingest_doc(
                    text=content,
                    persist_directory=PERSIST_DIR,
                    collection_name=collection,
                    chunking_config=None,
                    embedding_config=None,
                    source=file.name
                )
                click.echo(f"Ingested {file} into '{collection}' collection")

@rag_group.command()
@click.argument("query")
@click.option("--collection", default=DEFAULT_COLLECTION, help="Collection name")
@click.option("--n-results", default=5, help="Number of results to return")
def retrieve(query, collection, n_results):
    """Retrieve documents from vector store"""
    results = retrieve_doc(
        query=query,
        k=n_results,
        persist_directory=PERSIST_DIR,
        collection_name=collection
    )

    for result in results:
        click.echo(f"\n=== From {result['metadata'].get('source', 'unknown')} ===\n")
        click.echo(result['text'])
        click.echo(f"\nScore: {result['score']:.4f}\n")

@rag_group.command()
@click.argument("query")
@click.option("--collection", default=DEFAULT_COLLECTION, help="Collection name")
def query(query: str, collection: str = DEFAULT_COLLECTION) -> str:
    """Perform a RAG query using LangChain and LLM client.

    Args:
        query: The query to answer
        collection: Name of the collection to query

    Returns:
        The generated response from the LLM incorporating the retrieved context
    """
    from .llm_clients import get_default_client
    client = get_default_client()

    # Get relevant context
    results = retrieve_doc(
        query=query,
        k=3,
        persist_directory=PERSIST_DIR,
        collection_name=collection
    )
    context = "\n\n".join(r['text'] for r in results)

    # Create prompt with context
    prompt = f"""
    Answer the question based only on the context provided.

    Context: {context}

    Question: {query}
    """

    # Get response from LLM
    return client.generate(prompt)

@rag_group.command()
@click.option("--collection", default=DEFAULT_COLLECTION, help="Collection name")
def chat(collection: str = DEFAULT_COLLECTION):
    """Start an interactive chat session with RAG context"""
    from .llm_clients import get_default_client
    client = get_default_client()

    messages = []
    system_prompt = """You are a helpful assistant that answers questions based on the provided context.
    - Always respond based on the context when available
    - Be concise and factual
    - If you don't know the answer, say so
    - Type '/exit' to end the chat"""

    click.echo(click.style("Starting RAG chat session. Type '/exit' to quit.\n", fg="green"))
    click.echo(click.style(system_prompt + "\n", fg="blue"))

    while True:
        query = click.prompt(click.style("You", fg="yellow", bold=True))

        if query.lower() == "/exit":
            break

        # Get relevant context
        results = retrieve_doc(
            query=query,
            k=3,
            persist_directory=PERSIST_DIR,
            collection_name=collection
        )
        context = "\n\n".join(r['text'] for r in results)

        # Build prompt with chat history
        prompt = f"""
        System: {system_prompt}

        Context: {context}

        Chat History:
        {format_chat_history(messages[-5:])}

        Question: {query}
        """

        # Get response
        click.echo(click.style("\nAssistant: ", fg="cyan", bold=True))
        response = client.generate(prompt)
        messages.append(("user", query))
        messages.append(("assistant", response))
        click.echo(click.style("\n"+"-" * 80, fg="magenta"))

@rag_group.command()
def list_collections():
    """List all collections in the vector store"""
    from chromadb import PersistentClient

    client = PersistentClient(path=PERSIST_DIR)
    collections = client.list_collections()

    if not collections:
        click.echo("No collections found")
        return

    click.echo(f"Collections in {PERSIST_DIR}:")
    for collection in collections:
        click.echo(f"\n=== {collection.name} ===")
        click.echo(f"Metadata: {collection.metadata}")
        click.echo(f"Count: {collection.count()} documents")

@rag_group.command()
@click.argument("collection_name")
def delete_collection(collection_name: str):
    """Delete a collection from the vector store"""
    from .utils import delete_collection

    if click.confirm(f"Are you sure you want to delete collection '{collection_name}'?"):
        delete_collection(collection_name, PERSIST_DIR)
        click.echo(f"Deleted collection '{collection_name}'")
    else:
        click.echo("Collection deletion cancelled")

def format_chat_history(messages):
    """Format chat history for prompt"""
    return "\n".join(
        f"{role.capitalize()}: {message}"
        for role, message in messages
    )
