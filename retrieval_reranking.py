from typing import List
from sentence_transformers import CrossEncoder

def retrieve_top_chunks(db, query: str, top_k: int = 5) -> List[str]:
    """Retrieve the top_k most relevant chunks from the FAISS index."""
    results = db.similarity_search(query, k=top_k)
    return [doc.page_content for doc in results]

def rerank_chunks(query: str, chunks: List[str], reranker_model_name: str = "BAAI/bge-reranker-large") -> List[str]:
    """Rerank the retrieved chunks using a cross-encoder."""
    model = CrossEncoder(reranker_model_name)
    scores = model.predict([(query, chunk) for chunk in chunks])
    # Sort chunks by score in descending order
    reranked_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)]
    return reranked_chunks

# Example usage:
if __name__ == "__main__":
    from embedding_storage import embed_and_store, load_embeddings_and_index
    text_chunks = ["This is the first chunk about cats.", "This is the second chunk about dogs.", "This is a chunk about both."]
    faiss_index_path = embed_and_store(text_chunks)
    db = load_embeddings_and_index(faiss_index_path)

    query = "What are dogs?"
    top_chunks = retrieve_top_chunks(db, query)
    print(f"Top chunks before reranking: {top_chunks}")

    reranked_chunks = rerank_chunks(query, top_chunks)
    print(f"Top chunks after reranking: {reranked_chunks}")
