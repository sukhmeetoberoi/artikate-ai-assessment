import os
import fitz  # PyMuPDF
import json
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATA_DIR = "section_02_rag/data"
VECTOR_STORE_DIR = "section_02_rag/vector_store"

def create_sample_pdfs():
    """Create sample legal PDF documents for the assessment."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    docs = [
        {
            "filename": "sample_doc_1.pdf",
            "title": "Non-Disclosure Agreement (NDA)",
            "content": [
                "This Non-Disclosure Agreement is entered into between TechCorp Solutions ('Vendor') and the Disclosing Party.",
                "1. TERMINATION: Either party may terminate this agreement with 30 days written notice to the other party.",
                "2. CONFIDENTIALITY PERIOD: The recipient shall maintain the confidentiality of all disclosed information for a period of 3 years following the date of disclosure.",
                "3. GOVERNING LAW: This agreement shall be governed by and construed in accordance with the laws of the State of Delaware.",
                "4. PURPOSE: The parties are entering into discussions regarding a potential business relationship."
            ]
        },
        {
            "filename": "sample_doc_2.pdf",
            "title": "Service Agreement",
            "content": [
                "This Service Agreement is between Acme Corporation ('Client') and the Service Provider.",
                "1. PAYMENT TERMS: All invoices shall be paid within Net 30 days from the date of invoice receipt.",
                "2. LIMITATION OF LIABILITY: The total liability of the Service Provider under this agreement shall not exceed ₹50 lakhs.",
                "3. DURATION: This contract shall remain in effect for a duration of 12 months from the commencement date.",
                "4. SERVICES: The Service Provider will provide cloud infrastructure management and monitoring services."
            ]
        },
        {
            "filename": "sample_doc_3.pdf",
            "title": "Employment Policy",
            "content": [
                "Artikate AI Solutions - Standard Employment Policy",
                "1. LEAVE POLICY: All full-time employees are entitled to 21 days annual leave per calendar year.",
                "2. RESIGNATION: The notice period for resignation by an employee is 60 days.",
                "3. PROBATION: New employees will be on a probation period for the first 6 months of their employment.",
                "4. WORKING HOURS: Standard working hours are from 9:00 AM to 6:00 PM, Monday through Friday."
            ]
        }
    ]

    for doc in docs:
        filepath = os.path.join(DATA_DIR, doc["filename"])
        print(f"Creating {filepath}...")
        c = canvas.Canvas(filepath, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, doc["title"])
        c.setFont("Helvetica", 12)
        
        y = 700
        for line in doc["content"]:
            c.drawString(100, y, line)
            y -= 30
        
        c.save()
    print("All sample PDFs created successfully.")

def load_and_chunk_pdfs() -> List[Dict]:
    """Load PDFs and split them into metadata-enriched chunks."""
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=64
    )

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DATA_DIR, filename)
            print(f"Processing {filename}...")
            try:
                doc = fitz.open(filepath)
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    chunks = text_splitter.split_text(text)
                    
                    for i, chunk_text in enumerate(chunks):
                        all_chunks.append({
                            "text": chunk_text,
                            "metadata": {
                                "filename": filename,
                                "page": page_num + 1,
                                "chunk_index": i
                            }
                        })
                doc.close()
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return all_chunks

def main():
    try:
        print("--- Starting Ingestion Process ---")
        create_sample_pdfs()
        chunks = load_and_chunk_pdfs()
        print(f"Total chunks created: {len(chunks)}")
        
        # In a real scenario, we'd pass these chunks to the retriever to build the index.
        # For now, we save them to a temporary JSON so retriever.py can use them to build FAISS index.
        with open(os.path.join(VECTOR_STORE_DIR, "chunks.json"), "w") as f:
            json.dump(chunks, f)
        
        print(f"Saved chunks to {VECTOR_STORE_DIR}/chunks.json")
        print("--- Ingestion Complete ---")
    except Exception as e:
        print(f"Ingestion failed: {e}")

if __name__ == "__main__":
    main()
