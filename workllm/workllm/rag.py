import click
from pathlib import Path
from typing import List
import chromadb
from chromadb.utils import embedding_functions
from .llm_clients import LLMClient
import hashlib
from unstructured.partition.pdf import partition_pdf
import os

DEFAULT_COLLECTION = "workllm"
PERSIST_DIR = os.path.expanduser("~/.workllm/vectorstore")

def get_vector_client():
    """Get a ChromaDB persistent client instance.
    
    Returns:
        A ChromaDB persistent client configured to use the default persistence directory
    """
    return chromadb.PersistentClient(path=PERSIST_DIR)

def get_collection(collection: str = DEFAULT_COLLECTION):
    """Get or create a ChromaDB collection with default embedding function.
    
    Args:
        collection: Name of the collection to get/create (defaults to DEFAULT_COLLECTION)
        
    Returns:
        A ChromaDB collection instance configured with the default embedding function
    """
    client = get_vector_client()
    embedding_func = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name=collection,
        embedding_function=embedding_func
    )

def hash_content(content: str) -> str:
    """Generate a SHA-256 hash of the content.
    
    Args:
        content: The string content to hash
        
    Returns:
        A hexadecimal string representation of the SHA-256 hash
    """
    return hashlib.sha256(content.encode()).hexdigest()

@click.group()
def rag_group():
    """RAG document management commands"""

@rag_group.command()
def list_collections():
    """List all collections and their document counts"""
    client = get_vector_client()
    collections = client.list_collections()
    
    if not collections:
        click.echo("No collections found")
        return
        
    for collection_name in collections:
        collection = client.get_collection(collection_name)
        count = collection.count()
        click.echo(f"{collection_name} - {count} documents")

@rag_group.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option("--collection", default=DEFAULT_COLLECTION, help="Collection name")
def ingest(paths, collection):
    """Ingest documents into vector store"""
    coll = get_collection(collection)
    
    documents = []
    metadatas = []
    ids = []
    
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
                
                doc_id = hash_content(content)
                documents.append(content)
                metadatas.append({"source": str(file)})
                ids.append(doc_id)
    
    coll.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    click.echo(f"Ingested {len(ids)} documents into '{collection}' collection")

@rag_group.command()
@click.argument("retrieve")
@click.option("--collection", default=DEFAULT_COLLECTION, help="Collection name")
@click.option("--n-results", default=3, help="Number of results to return")
def retrieve(query, collection, n_results):
    """Retrieve documents from vector store"""
    coll = get_collection(collection)
    results = coll.query(
        query_texts=[query],
        n_results=n_results
    )
    
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        click.echo(f"\n=== From {meta['source']} ===\n{doc}\n")

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
    
    # Get vector store collection
    coll = get_collection(collection)
    
    # Query the collection
    results = coll.query(
        query_texts=[query],
        n_results=3
    )
    
    # Format context from retrieved documents
    context = "\n\n".join(results["documents"][0])
    
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
    coll = get_collection(collection)
    
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
        results = coll.query(
            query_texts=[query],
            n_results=3
        )
        context = "\n\n".join(results["documents"][0])
        
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

def format_chat_history(messages):
    """Format chat history for prompt"""
    return "\n".join(
        f"{role.capitalize()}: {message}"
        for role, message in messages
    )
