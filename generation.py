import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

def generate_answer(context: str, query: str, api_key: str = os.getenv("GOOGLE_API_KEY"), model_name: str = '2.0-flash'):
    """Generate an answer using Gemini with the given context and query."""

    if api_key is None:
        raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    # Create a strict prompt that enforces context-bound answers
    system_prompt = """You are an AI assistant that can ONLY answer questions based on the provided context. 
    If the question cannot be answered using the given context, respond with: 
    "I cannot answer this question as it's not covered in the provided context."
    DO NOT use any external knowledge or make assumptions beyond what's in the context."""
    
    prompt = f"""{system_prompt}

Context:
{context}

Question: {query}

Answer (based ONLY on the above context):"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating content: {e}")
        return "I'm sorry, I couldn't generate an answer."

# Example usage:
if __name__ == "__main__":
    context = "Cats are cute and dogs are loyal."
    query = "What are cats?"
    answer = generate_answer(context, query)
    print(f"Answer: {answer}")
