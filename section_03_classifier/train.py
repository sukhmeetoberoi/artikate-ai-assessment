import os
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

# Constants
MODEL_NAME = "distilbert-base-uncased"
DATA_FILE = "section_03_classifier/data/train.json"
MODEL_DIR = "section_03_classifier/model"
CATEGORIES = ["billing", "technical_issue", "feature_request", "complaint", "other"]
LABEL2ID = {label: i for i, label in enumerate(CATEGORIES)}
ID2LABEL = {i: label for i, label in enumerate(CATEGORIES)}

def load_data():
    """Load and preprocess data from JSON."""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file {DATA_FILE} not found. Run generate_data.py first.")
    
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    
    texts = [item["text"] for item in data]
    labels = [LABEL2ID[item["label"]] for item in data]
    return texts, labels

def compute_metrics(eval_pred):
    """Compute accuracy for validation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

def train():
    """Fine-tune DistilBERT for ticket classification."""
    try:
        print("Loading data...")
        texts, labels = load_data()
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        print(f"Dataset sizes: Train={len(train_texts)}, Val={len(val_texts)}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding=True)

        train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
        val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=len(CATEGORIES),
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )

        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch", # Changed from evaluation_strategy
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
        )

        print("Starting training...")
        trainer.train()

        print(f"Saving model to {MODEL_DIR}...")
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        
        # Save labels for reference during inference
        with open(os.path.join(MODEL_DIR, "labels.json"), "w") as f:
            json.dump(ID2LABEL, f)
            
        print("Training complete.")

    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    train()
