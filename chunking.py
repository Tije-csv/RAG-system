from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
    """
    Chunk text into smaller parts using RecursiveCharacterTextSplitter.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The overlap between adjacent chunks (in characters).

    Returns:
        List[str]: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Example usage:
if __name__ == "__main__":
    text = "This is a long text that needs to be chunked into smaller parts. " * 20
    chunks = chunk_text(text)
    print(f"Number of chunks: {len(chunks)}")
    print(f"First chunk: {chunks[0]}")
