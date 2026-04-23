import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch.nn.functional as F

MODEL_DIR = "section_03_classifier/model"
DATA_DIR = "section_03_classifier/data"

def evaluate():
    print("--- Starting Model Evaluation ---")
    
    if not os.path.exists(MODEL_DIR):
        print("Model not found. Please run train.py first.")
        return

    # Load test data
    with open(os.path.join(DATA_DIR, "test.json"), "r") as f:
        test_data = json.load(f)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    # Load labels
    with open(os.path.join(MODEL_DIR, "labels.json"), "r") as f:
        id2label = json.load(f)
    
    label2id = {v: int(k) for k, v in id2label.items()}
    true_labels = [label2id[d["label"]] for d in test_data]
    texts = [d["text"] for d in test_data]

    predictions = []
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(pred)

    # Calculate metrics
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")
    cm = confusion_matrix(true_labels, predictions)

    print("\nEvaluation Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (Weighted): {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nLabel Map:")
    for k, v in id2label.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    evaluate()
