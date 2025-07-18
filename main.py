from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from data_loader import load_data_from_url, load_data_from_pdf, transcribe_audio
from chunking import chunk_text
from embedding_storage import embed_and_store, load_embeddings_and_index
from retrieval_reranking import retrieve_top_chunks, rerank_chunks
from generation import generate_answer

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class QueryRequest(BaseModel):
    query: str
    data_sources: List[str] = []  # List of URLs, PDF paths, or audio paths

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Endpoint to answer queries based on provided data sources."""
    try:
        # 1. Data Loading
        all_text = ""
        for source in request.data_sources:
            if source.startswith("http://") or source.startswith("https://"):
                all_text += load_data_from_url(source)
            elif source.endswith(".pdf"):
                all_text += load_data_from_pdf(source)
            elif source.endswith(".mp3") or source.endswith(".wav"):
                all_text += transcribe_audio(source)
            else:
                raise ValueError(f"Unsupported data source: {source}")

        # 2. Chunking
        text_chunks = chunk_text(all_text)

        # 3. Embedding and Storage
        faiss_index_path = "faiss_index"  # You can parameterize this
        embed_and_store(text_chunks, faiss_index_path=faiss_index_path)
        db = load_embeddings_and_index(faiss_index_path=faiss_index_path)

        # 4. Retrieval and Reranking
        top_chunks = retrieve_top_chunks(db, request.query)
        reranked_chunks = rerank_chunks(request.query, top_chunks)
        context = "\n".join(reranked_chunks)

        # 5. Generation
        answer = generate_answer(context, request.query)

        # 6. Output
        return {"answer": answer, "sources": request.data_sources}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Example Usage (in a separate terminal):
    # uvicorn main:app --reload
    # Then, send a POST request to http://localhost:8000/query with a JSON body like:
    # {"query": "What is FastAPI?", "data_sources": ["https://fastapi.tiangolo.com/"]}
    uvicorn.run(app, host="0.0.0.0", port=8000)
