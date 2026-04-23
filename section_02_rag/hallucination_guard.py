from typing import List, Tuple, Dict

class HallucinationGuard:
    @staticmethod
    def calculate_confidence(retrieval_results: List[Tuple[Dict, float]]) -> float:
        """
        Calculate confidence score based on similarity scores.
        If the highest similarity score is below 0.75, confidence is low.
        """
        if not retrieval_results:
            return 0.0
        
        # Get the highest similarity score (already cosine similarity due to normalize_L2 + IndexFlatIP)
        max_score = retrieval_results[0][1]
        
        # Scale score: if max_score >= 0.75, we keep it as is or boost it.
        # If max_score < 0.75, we significantly penalize it.
        if max_score < 0.75:
            confidence = max_score * 0.5  # Will be below 0.5
        else:
            confidence = max_score
            
        return min(max(confidence, 0.0), 1.0)

    @staticmethod
    def should_refuse(confidence: float) -> bool:
        """Determine if the system should refuse to answer."""
        return confidence < 0.5

    @staticmethod
    def get_refusal_message() -> str:
        return "I don't have enough information in the provided documents to answer this question."
