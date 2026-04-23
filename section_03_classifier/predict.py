import os
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "section_03_classifier/model"

class TicketClassifier:
    def __init__(self):
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"Model directory {MODEL_DIR} not found. Train the model first.")
            
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        self.model.eval()
        
        with open(os.path.join(MODEL_DIR, "labels.json"), "r") as f:
            self.id2label = json.load(f)

    def predict(self, text: str):
        """Predict the category of a support ticket."""
        start_time = time.perf_counter()
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            
        label = self.id2label[str(pred_idx.item())]
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        return {
            "label": label,
            "confidence": confidence.item(),
            "latency_ms": latency_ms
        }

if __name__ == "__main__":
    try:
        classifier = TicketClassifier()
        sample_text = "I was charged twice this month for my subscription"
        result = classifier.predict(sample_text)
        print(f"\nText: {sample_text}")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Inference Time: {result['latency_ms']:.2f} ms")
    except Exception as e:
        print(f"Error: {e}")
