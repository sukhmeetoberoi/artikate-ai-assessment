import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import List, Dict

MODEL_DIR = "section_03_classifier/model"
TEST_FILE = "section_03_classifier/data/test.json"
CATEGORIES = ["billing", "technical_issue", "feature_request", "complaint", "other"]

class Evaluator:
    def __init__(self):
        print(f"Loading model from {MODEL_DIR}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        self.model.eval()
        
        with open(os.path.join(MODEL_DIR, "labels.json"), "r") as f:
            self.id2label = json.load(f)

    def evaluate(self):
        """Evaluate the model on the test set and report metrics."""
        try:
            with open(TEST_FILE, "r") as f:
                test_data = json.load(f)

            texts = [item["text"] for item in test_data]
            y_true = [item["label"] for item in test_data]
            y_pred = []

            print(f"Running inference on {len(texts)} test examples...")
            
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    pred_idx = torch.argmax(outputs.logits, dim=1).item()
                    y_pred.append(self.id2label[str(pred_idx)])

            # Metrics
            acc = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred, target_names=CATEGORIES)
            cm = confusion_matrix(y_true, y_pred, labels=CATEGORIES)

            print("\n" + "="*30)
            print("EVALUATION REPORT")
            print("="*30)
            print(f"Overall Accuracy: {acc:.4f}")
            print("\nClassification Report:")
            print(report)
            
            print("\nConfusion Matrix:")
            header = "          " + "".join([f"{cat:>15}" for cat in CATEGORIES])
            print(header)
            for i, row in enumerate(cm):
                row_str = f"{CATEGORIES[i]:<10}" + "".join([f"{val:>15}" for val in row])
                print(row_str)

            # Identify 2 most confused pairs
            confusions = []
            for i in range(len(CATEGORIES)):
                for j in range(len(CATEGORIES)):
                    if i != j:
                        confusions.append((CATEGORIES[i], CATEGORIES[j], cm[i][j]))
            
            # Sort by count descending
            confusions.sort(key=lambda x: x[2], reverse=True)
            
            print("\nTop 2 Most Confused Class Pairs:")
            for pair in confusions[:2]:
                print(f" - {pair[0]} misclassified as {pair[1]}: {pair[2]} times")

        except Exception as e:
            print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate()
