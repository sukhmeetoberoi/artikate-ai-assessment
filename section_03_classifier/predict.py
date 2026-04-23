import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "section_03_classifier/model"

class TicketClassifier:
    def __init__(self):
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError("Model files not found. Please run train.py first.")
            
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        self.model.eval()
        
        with open(os.path.join(MODEL_DIR, "labels.json"), "r") as f:
            self.id2label = json.load(f)

    def predict(self, text: str):
        """Predict the category of a support ticket."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            
        label = self.id2label[str(pred_idx.item())]
        return {
            "label": label,
            "confidence": confidence.item()
        }

if __name__ == "__main__":
    try:
        classifier = TicketClassifier()
        sample_text = "I need help resetting my password, I'm locked out."
        result = classifier.predict(sample_text)
        print(f"Ticket: '{sample_text}'")
        print(f"Prediction: {result['label']} (Confidence: {result['confidence']:.4f})")
    except Exception as e:
        print(f"Prediction error: {e}")
