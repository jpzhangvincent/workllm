import datetime
import os
import subprocess
from typing import Any, Literal, List
import xml.etree.ElementTree as ET
from urllib.parse import urljoin

from chromadb import PersistentClient
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import PythonCodeTextSplitter, RecursiveCharacterTextSplitter
import requests
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

# Cache for vectorstores and retrievers
_vectorstore_cache = {}
_retriever_cache = {}

def get_clipboard_content() -> str:
    try:
        return subprocess.check_output(['pbpaste'], text=True)
    except subprocess.CalledProcessError:
        return ""
    except FileNotFoundError:
        raise RuntimeError("Clipboard access requires macOS with pbcopy/pbpaste") from None

def safe_subprocess_run(command: str) -> str:
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        check=False
    )
    return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

def fix_code(client: Any, lint_output: str) -> str:
    """Use LLM to fix code based on linting output

    Args:
        client: LLM client instance
        lint_output: Output from linting tool

    Returns:
        str: Suggested fixes from LLM
    """
    prompt = f"""You are a code quality expert. Below is the output from a linting tool:

    {lint_output}

    Please provide specific fixes for these issues. Format your response as:
    1. For each issue, explain the problem and how to fix it
    2. Provide the complete fixed code at the end

    Return only the fixes, no additional commentary."""

    try:
        return client.generate(prompt)
    except Exception as e:
        raise RuntimeError(f"Failed to generate fixes: {str(e)}") from e


def get_sitemap_urls(base_url: str, sitemap_filename: str = "sitemap.xml") -> List[str]:
    """Fetches and parses a sitemap XML file to extract URLs.

    Args:
        base_url: The base URL of the website
        sitemap_filename: The filename of the sitemap (default: sitemap.xml)

    Returns:
        List of URLs found in the sitemap. If sitemap is not found, returns a list
        containing only the base URL.

    Raises:
        ValueError: If there's an error fetching (except 404) or parsing the sitemap
    """
    try:
        sitemap_url = urljoin(base_url, sitemap_filename)

        # Fetch sitemap URL
        response = requests.get(sitemap_url, timeout=10)

        # # Return just the base URL if sitemap not found
        if response.status_code == 404:
            return [base_url.rstrip("/")]

        response.raise_for_status()

        # Parse XML content
        root = ET.fromstring(response.content)

        # Handle different XML namespaces that sitemaps might use
        namespaces = (
            {"ns": root.tag.split("}")[0].strip("{")} if "}" in root.tag else ""
        )

        # Extract URLs using namespace if present
        if namespaces:
            urls = [elem.text for elem in root.findall(".//ns:loc", namespaces)]
        else:
            urls = [elem.text for elem in root.findall(".//loc")]

        return urls

    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch sitemap: {str(e)}")
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse sitemap XML: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error processing sitemap: {str(e)}")
    

def get_docling_content(doc, save_format: str):
        if save_format == 'markdown':
            output = doc.export_to_markdown()
        elif save_format == 'html':
            output = doc.export_to_html()
        elif save_format == 'json':
            output = doc.export_to_json()
        else:
            print("Unsupported format. Please choose markdown, html or json.")
        return output

class ChunkingConfig:
    """Configuration for text chunking strategies"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
        code_splitter: bool = False,
        chunking_strategy: Literal["recursive", "hybrid"] = "recursive",
        tokenizer: PreTrainedTokenizerBase | str = "sentence-transformers/all-MiniLM-L6-v2",
        max_tokens: int = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        self.code_splitter = code_splitter
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens


class EmbeddingConfig:
    """Configuration for embedding models"""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs: dict[str, Any] | None = None
    ):
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {'device': 'cpu', 'normalize_embeddings': False}


def ingest_doc(
    text: str,
    chunking_config: ChunkingConfig | None = None,
    embedding_config: EmbeddingConfig | None = None,
    persist_directory: str = "data/chroma",
    collection_name: str = "workllm",
    source: str = "",
    batch_size: int = 32
) -> None:
    """Ingest text data with chunking and embedding

    Args:
        text: The text to ingest
        chunking_config: Configuration for text chunking
        embedding_config: Configuration for embedding model
        persist_directory: Directory to store Chroma vector database
        collection_name: Name of the collection to store documents
        source: Source identifier for the documents
        batch_size: Number of documents to process at once for embeddings
    """
    # Initialize configurations with defaults if not provided
    chunking_config = chunking_config or ChunkingConfig()
    embedding_config = embedding_config or EmbeddingConfig()

    # Create text splitter based on configuration
    if chunking_config.chunking_strategy == "recursive":
        if chunking_config.code_splitter:
            splitter = PythonCodeTextSplitter(
                chunk_size=chunking_config.chunk_size,
                chunk_overlap=chunking_config.chunk_overlap
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunking_config.chunk_size,
                chunk_overlap=chunking_config.chunk_overlap,
                separators=chunking_config.separators
            )
        # Split text into chunks and convert to Documents
        chunks = splitter.split_text(text)
    else:
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
        chunker = HybridChunker(
            tokenizer=chunking_config.tokenizer,
            max_tokens=chunking_config.MAX_TOKENS,
            merge_peers=True,
        )
        chunk_iter = chunker.chunk(dl_doc=text)
        chunks = list(chunk_iter)

    documents = [
        Document(
            page_content=chunk,
            metadata={"source": source, "chunk_index": i}
        )
        for i, chunk in enumerate(chunks)
    ]

    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_config.model_name,
        model_kwargs={'device': 'cpu'}  # Simplified model kwargs
    )

    # Initialize Chroma client and create directory
    os.makedirs(persist_directory, exist_ok=True)

    # Create or get vectorstore
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    # Process documents in batches with progress bar
    for i in tqdm(range(0, len(documents), batch_size), desc="Processing documents"):
        batch = documents[i:i + batch_size]
        try:
            vectorstore.add_documents(batch)
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            continue

    # Update metadata
    client = PersistentClient(path=persist_directory)
    collection = client.get_collection(name=collection_name)

    metadata = collection.metadata or {}
    sources = metadata.get("sources", "")
    if source and source not in sources:
        sources += source + ", "

    collection.modify(
        metadata={
            **metadata,
            "sources": sources,
            "document_count": len(chunks),
            "chunk_size": chunking_config.chunk_size,
            "chunk_overlap": chunking_config.chunk_overlap,
            "last_updated": datetime.datetime.now().isoformat(),
            "embedding_model": embedding_config.model_name
        }
    )

def delete_collection(
    collection_name: str,
    persist_directory: str = "data/chroma"
) -> None:
    """Delete a collection from the vector store

    Args:
        collection_name: Name of collection to delete
        persist_directory: Directory where Chroma vector database is stored
    """
    client = PersistentClient(path=persist_directory)
    try:
        client.delete_collection(name=collection_name)
    except ValueError as e:
        if "Collection not found" in str(e):
            return
        raise


def retrieve_doc(
    query: str,
    k: int = 5,
    persist_directory: str = "data/chroma",
    embedding_config: EmbeddingConfig | None = None,
    search_mode: str = "hybrid",  # "semantic" or "hybrid"
    text_weight: float = 0.3,
    semantic_weight: float = 0.7,
    rerank: bool = False,
    collection_name: str = "workllm"
) -> list[dict[str, Any]]:
    """Retrieve documents using semantic or hybrid search with optional reranking

    Args:
        query: The search query
        k: Number of results to return
        persist_directory: Directory where Chroma vector database is stored
        embedding_config: Configuration for embedding model
        search_mode: Search mode to use - "semantic" or "hybrid"
        text_weight: Weight for text search score (only used in hybrid mode)
        semantic_weight: Weight for semantic search score (only used in hybrid mode)
        rerank: Whether to apply reranking to results
        collection_name: Name of the collection to search

    Returns:
        List of documents with scores and metadata

    Raises:
        ValueError: If embedding dimensions don't match collection dimensions or invalid search mode
    """
    if search_mode not in ["semantic", "hybrid"]:
        raise ValueError("search_mode must be either 'semantic' or 'hybrid'")

    try:
        # Get or create vectorstore from cache
        cache_key = f"{persist_directory}_{collection_name}"
        if cache_key not in _vectorstore_cache:
            # Get embedding config from collection metadata if not provided
            if embedding_config is None:
                client = PersistentClient(path=persist_directory)
                collection = client.get_collection(name=collection_name)
                metadata = collection.metadata or {}
                embedding_config = EmbeddingConfig(
                    model_name=metadata.get("embedding_model", "sentence-transformers/all-mpnet-base-v2"),
                    model_kwargs=metadata.get("embedding_model_kwargs", {'device': 'cpu', 'normalize_embeddings': False})
                )

            # Initialize embedding model
            embeddings = HuggingFaceEmbeddings(
                model_name=embedding_config.model_name,
                model_kwargs={'device': 'cpu'}  # Simplified model kwargs
            )

            # Create vectorstore
            _vectorstore_cache[cache_key] = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_directory
            )

        vectorstore = _vectorstore_cache[cache_key]

        if search_mode == "semantic":
            # Use only vector similarity search
            results = vectorstore.similarity_search_with_score(query, k=k)
            formatted_results = [
                {
                    'id': str(hash(doc.page_content)),
                    'text': doc.page_content,
                    'score': score,
                    'metadata': doc.metadata
                }
                for doc, score in results
            ]
        else:
            # hybrid search
            # Get or create retrievers from cache
            if cache_key not in _retriever_cache:
                vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

                # Get all documents from collection
                client = PersistentClient(path=persist_directory)
                collection = client.get_collection(name=collection_name)
                all_docs = collection.get()
                documents = [
                    Document(
                        page_content=doc,
                        metadata=meta if meta else {}
                    )
                    for doc, meta in zip(all_docs['documents'], all_docs['metadatas'], strict=False)
                ]

                # Create BM25 retriever
                bm25_retriever = BM25Retriever.from_documents(documents)
                bm25_retriever.k = k

                # Create and cache ensemble retriever
                _retriever_cache[cache_key] = EnsembleRetriever(
                    retrievers=[vectorstore_retriever, bm25_retriever],
                    weights=[semantic_weight, text_weight]
                )

            # Get results using cached retriever
            results = _retriever_cache[cache_key].invoke(query)

            # Convert to standard format
            formatted_results = [
                {
                    'id': str(hash(doc.page_content)),
                    'text': doc.page_content,
                    'score': 1.0,  # Base similarity score
                    'metadata': doc.metadata
                }
                for doc in results
            ]

        if rerank:
            try:
                from sentence_transformers import CrossEncoder
                reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

                # Prepare pairs for reranking
                pairs = [[query, doc['text']] for doc in formatted_results]

                # Get reranking scores
                rerank_scores = reranker.predict(pairs)

                # Update scores and resort
                for i, score in enumerate(rerank_scores):
                    formatted_results[i]['score'] = float(score)

                formatted_results.sort(key=lambda x: x['score'], reverse=True)

            except ImportError:
                print("Warning: sentence-transformers not installed. Skipping reranking.")
                print("Install with: pip install sentence-transformers")
            except Exception as e:
                print(f"Warning: Reranking failed: {str(e)}")

        return formatted_results[:k]

    except Exception as e:
        print(f"Error during retrieval: {str(e)}")
        return []
