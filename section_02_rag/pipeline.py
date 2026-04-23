import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from section_02_rag.retriever import Retriever
from section_02_rag.generator import Generator
from section_02_rag.hallucination_guard import HallucinationGuard

load_dotenv()

class RAGPipeline:
    def __init__(self):
        print("Initializing RAG Pipeline...")
        self.retriever = Retriever()
        self.generator = Generator()
        self.guard = HallucinationGuard()
        
        # Ensure index is loaded
        if not self.retriever.load_index():
            print("Warning: Vector index not found. Please run ingest.py and then retriever.py to build it.")

    def query(self, question: str) -> Dict[str, Any]:
        """Process a query through the RAG pipeline."""
        print(f"\n--- Processing Query: {question} ---")
        try:
            # 1. Retrieve
            retrieval_results = self.retriever.retrieve(question, top_k=5)
            
            # 2. Guard - Calculate Confidence
            confidence = self.guard.calculate_confidence(retrieval_results)
            print(f"Calculated Confidence: {confidence:.2f}")

            # 3. Refusal check
            if self.guard.should_refuse(confidence):
                print("Confidence too low. Refusing to answer.")
                return {
                    'answer': self.guard.get_refusal_message(),
                    'sources': [],
                    'confidence': confidence,
                }

            # 4. Generate
            context_chunks = [res[0] for res in retrieval_results]
            answer = self.generator.generate_answer(question, context_chunks)

            # 5. Format sources
            sources = []
            for chunk, score in retrieval_results:
                sources.append({
                    'document': chunk['filename'],
                    'page': chunk['page'],
                    'chunk': chunk['text']
                })

            return {
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
            }

        except Exception as e:
            print(f"Error in RAG Pipeline: {e}")
            return {
                'answer': "An internal error occurred while processing your request.",
                'sources': [],
                'confidence': 0.0,
            }

def main():
    pipeline = RAGPipeline()
    
    # Demo Queries
    queries = [
        "What is the notice period in the NDA with TechCorp Solutions?",
        "What is the limitation of liability in the Acme Corporation agreement?",
        "What is the policy for space travel in the employment contract?" # Should trigger low confidence/refusal
    ]

    for q in queries:
        result = pipeline.query(q)
        print(f"ANSWER: {result['answer']}")
        print(f"CONFIDENCE: {result['confidence']:.2f}")
        if result['sources']:
            print(f"SOURCE: {result['sources'][0]['document']} (Page {result['sources'][0]['page']})")
        print("-" * 50)

if __name__ == "__main__":
    main()
