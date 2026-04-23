import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.preprocessing import LabelEncoder

MODEL_NAME = "distilbert-base-uncased"
DATA_DIR = "section_03_classifier/data"
MODEL_DIR = "section_03_classifier/model"

def train():
    print("--- Starting Classifier Training ---")
    
    # Load data
    try:
        with open(os.path.join(DATA_DIR, "train.json"), "r") as f:
            train_data = json.load(f)
        with open(os.path.join(DATA_DIR, "test.json"), "r") as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print("Data files not found. Run generate_data.py first.")
        return

    # Encode labels
    le = LabelEncoder()
    labels = [d["label"] for d in train_data]
    le.fit(labels)
    
    num_labels = len(le.classes_)
    id2label = {i: label for i, label in enumerate(le.classes_)}
    label2id = {label: i for i, label in enumerate(le.classes_)}

    # Convert to HuggingFace Dataset
    def format_dataset(data):
        return Dataset.from_dict({
            "text": [d["text"] for d in data],
            "label": [label2id[d["label"]] for d in data]
        })

    train_ds = format_dataset(train_data)
    test_ds = format_dataset(test_data)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_train = train_ds.map(preprocess_function, batched=True)
    tokenized_test = test_ds.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir='./logs',
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Save model and tokenizer
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    
    # Save label encoder classes
    with open(os.path.join(MODEL_DIR, "labels.json"), "w") as f:
        json.dump(id2label, f)

    print(f"--- Training Complete. Model saved to {MODEL_DIR} ---")

if __name__ == "__main__":
    train()
