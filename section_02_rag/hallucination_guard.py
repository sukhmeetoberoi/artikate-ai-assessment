from typing import List, Tuple, Dict

class HallucinationGuard:
    @staticmethod
    def calculate_confidence(retrieval_results: List[Tuple[Dict, float]]) -> float:
        """
        Calculate confidence score based on actual cosine similarity scores.
        Adjusted for local model 'all-MiniLM-L6-v2'.
        """
        if not retrieval_results:
            return 0.0
        
        # Raw cosine similarity from the top retrieved chunk
        max_similarity = retrieval_results[0][1]
        print(f"[DEBUG] Raw Similarity Score: {max_similarity:.4f}")
        
        # Scaling logic for local model (all-MiniLM-L6-v2):
        # - Strong matches typically fall between 0.55 and 0.80
        # - Weak/Irrelevant matches typically fall below 0.40
        
        if max_similarity >= 0.60:
            # High confidence
            confidence = 0.8 + (max_similarity - 0.60) * 0.5
        elif max_similarity < 0.45:
            # Low confidence (will trigger refusal)
            confidence = max_similarity * 0.8 
        else:
            # Linear mapping between 0.45 (refusal) and 0.60 (high confidence)
            # Should cross the 0.5 threshold around 0.50 similarity
            confidence = (max_similarity - 0.45) * (0.8 - 0.4) / (0.60 - 0.45) + 0.4
            
        return min(max(confidence, 0.0), 1.0)

    @staticmethod
    def should_refuse(confidence: float) -> bool:
        """Determine if the system should refuse to answer based on confidence."""
        return confidence < 0.5

    @staticmethod
    def get_refusal_message() -> str:
        """Standard refusal message for low-confidence queries."""
        return "I don't have enough information in the provided documents to answer this question."
