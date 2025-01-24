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
@click.argument("query")
@click.option("--collection", default=DEFAULT_COLLECTION, help="Collection name")
@click.option("--n-results", default=3, help="Number of results to return")
def query(query, collection, n_results):
    """Query vector store"""
    coll = get_collection(collection)
    results = coll.query(
        query_texts=[query],
        n_results=n_results
    )
    
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        click.echo(f"\n=== From {meta['source']} ===\n{doc}\n")

def rag_query(client: LLMClient, query: str, context: List[str]) -> str:
    context_str = "\n\n".join(context)
    return client.generate(
        prompt=f"Query: {query}\n\nContext:\n{context_str}",
        system="Answer the query using the provided context. Be concise and cite sources."
    )
