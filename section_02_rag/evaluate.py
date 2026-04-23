import json
import os
from typing import List, Dict
from section_02_rag.retriever import Retriever

EVAL_FILE = "section_02_rag/eval_questions.json"

def run_evaluation():
    print("--- Starting Evaluation Harness ---")
    
    if not os.path.exists(EVAL_FILE):
        print(f"Evaluation file {EVAL_FILE} not found.")
        return

    with open(EVAL_FILE, "r") as f:
        questions = json.load(f)

    retriever = Retriever()
    if not retriever.load_index():
        print("Failed to load index. Please run ingest.py and retriever.py first.")
        return

    results = []
    correct_count = 0

    for item in questions:
        question = item["question"]
        expected_keywords = item["expected_chunk_keywords"]
        source_doc = item["source_document"]

        print(f"\nEvaluating: {question}")
        
        # Retrieve top-3 chunks
        retrieval_results = retriever.retrieve(question, top_k=3)
        
        found_keywords = False
        match_info = "Fail"
        
        # Check if any of the top-3 chunks contain the keywords and come from the right doc
        for chunk_meta, score in retrieval_results:
            text = chunk_meta["text"].lower()
            if chunk_meta["filename"] == source_doc:
                # Check if all keywords are present (simple keyword check)
                if all(kw.lower() in text for kw in expected_keywords):
                    found_keywords = True
                    match_info = f"Pass (Found in {source_doc})"
                    break
        
        if found_keywords:
            correct_count += 1
        
        results.append({
            "question": question,
            "status": match_info
        })
        print(f"Result: {match_info}")

    precision_at_3 = correct_count / len(questions) if questions else 0
    
    print("\n" + "="*30)
    print("EVALUATION REPORT")
    print("="*30)
    for res in results:
        print(f"{res['status']}: {res['question']}")
    print("-" * 30)
    print(f"FINAL PRECISION@3: {precision_at_3:.2f}")
    print("="*30)

if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        print(f"Evaluation failed: {e}")
