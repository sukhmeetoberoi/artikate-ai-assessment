import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class Generator:
    def __init__(self):
        self.mock_mode = os.getenv("MOCK_MODE", "False").lower() == "true"
        if not self.mock_mode:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate an answer (Simulated if MOCK_MODE=True)."""
        if self.mock_mode:
            return f"[MOCK ANSWER] Based on the provided context (found in {context_chunks[0]['filename']}), the answer to '{query}' is derived from the retrieved documents."

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
            print(f"Generating answer for: {query}...")
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
            print(f"Error in generation: {e}")
            return "An error occurred while generating the answer."

if __name__ == "__main__":
    # Test generation
    gen = Generator()
    test_context = [{"text": "The notice period is 30 days.", "filename": "test.pdf", "page": 1}]
    print(gen.generate_answer("What is the notice period?", test_context))
