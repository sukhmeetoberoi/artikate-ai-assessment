import os
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class Generator:
    def __init__(self):
        print("Initializing Groq Generator (llama3-8b-8192)...")
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate a real answer based on provided context chunks using Groq."""
        if not context_chunks:
            return "I don't have enough information in the provided documents to answer this question."

        context_text = ""
        for i, chunk in enumerate(context_chunks):
            context_text += f"\n--- Chunk {i+1} (Source: {chunk['filename']}, Page: {chunk['page']}) ---\n{chunk['text']}\n"

        system_prompt = (
            "You are a legal assistant specializing in contract analysis. "
            "Answer the user's question ONLY using the provided context chunks. "
            "If the context is insufficient, say 'I don't have enough information in the provided documents to answer this question'. "
            "Always cite the source document and page number for your information. "
            "Be precise and professional."
        )

        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

        try:
            print(f"Generating answer using Groq for: {query}...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in Groq generation: {e}")
            return "An error occurred while generating the answer using Groq."

if __name__ == "__main__":
    # Test generation
    gen = Generator()
    test_context = [{"text": "The notice period is 30 days.", "filename": "test.pdf", "page": 1}]
    # Note: Requires GROQ_API_KEY
    print(gen.generate_answer("What is the notice period?", test_context))
