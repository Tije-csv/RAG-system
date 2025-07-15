import os
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def embed_and_store(text_chunks: List[str], embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", faiss_index_path: str = "faiss_index"):
    """Embed text chunks and store them in FAISS."""
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = FAISS.from_texts(text_chunks, embeddings)
    db.save_local(faiss_index_path)
    return faiss_index_path

def load_embeddings_and_index(faiss_index_path: str = "faiss_index", embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load embeddings and FAISS index from disk."""
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = FAISS.load_local(faiss_index_path, embeddings)
    return db

# Example usage:
if __name__ == "__main__":
    text_chunks = ["This is the first chunk.", "This is the second chunk."]
    faiss_index_path = embed_and_store(text_chunks)
    print(f"FAISS index saved to: {faiss_index_path}")

    loaded_db = load_embeddings_and_index(faiss_index_path)
    print("FAISS index loaded successfully.")
