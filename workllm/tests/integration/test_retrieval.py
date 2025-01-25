import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workllm.utils import retrieve_doc, ingest_doc

def test_retrieval():
    # Test data
    test_docs = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing helps computers understand human language.",
        "Deep learning models require significant computational resources."
    ]
    
    # Ingest test documents separately
    for i, doc in enumerate(test_docs):
        ingest_doc(doc, collection_name="test_collection", source=f"doc_{i}")
    
    # Test queries
    query = "What is machine learning?"
    
    print("\nTesting semantic search...")
    semantic_results = retrieve_doc(
        query,
        k=2,
        collection_name="test_collection",
        search_mode="semantic"
    )
    print("\nSemantic search results:")
    for r in semantic_results:
        print(f"Score: {r['score']:.4f} - {r['text']}")
    
    print("\nTesting hybrid search...")
    hybrid_results = retrieve_doc(
        query,
        k=2,
        collection_name="test_collection",
        search_mode="hybrid",
        text_weight=0.3,
        semantic_weight=0.7
    )
    print("\nHybrid search results:")
    for r in hybrid_results:
        print(f"Score: {r['score']:.4f} - {r['text']}")
    
    print("\nTesting hybrid search with reranking...")
    reranked_results = retrieve_doc(
        query,
        k=2,
        collection_name="test_collection",
        search_mode="hybrid",
        text_weight=0.3,
        semantic_weight=0.7,
        rerank=True
    )
    print("\nReranked results:")
    for r in reranked_results:
        print(f"Score: {r['score']:.4f} - {r['text']}")

if __name__ == "__main__":
    test_retrieval()
