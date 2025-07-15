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
    prompt = f"Context: {context}\nQuery: {query}\nAnswer:"

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
