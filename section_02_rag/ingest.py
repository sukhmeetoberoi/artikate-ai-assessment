import os
import fitz  # PyMuPDF
import json
import faiss
import numpy as np
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATA_DIR = "section_02_rag/data"
VECTOR_STORE_DIR = "section_02_rag/vector_store"
INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.index")
METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "metadata.json")

def create_sample_pdfs():
    """Create sample legal PDF documents with wrapped text to prevent truncation."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    docs = [
        {
            "filename": "sample_doc_1.pdf",
            "title": "Non-Disclosure Agreement (NDA)",
            "content": [
                "This Non-Disclosure Agreement is between TechCorp Solutions",
                "and the Disclosing Party.",
                "1. NOTICE PERIOD: Either party may terminate with 30 days notice.",
                "2. CONFIDENTIALITY PERIOD: The recipient shall maintain secrecy",
                "for a period of 3 years following disclosure.",
                "3. GOVERNING LAW: This agreement is governed by the laws",
                "of the State of Delaware.",
                "4. PURPOSE: Covers all technical and business discussions."
            ]
        },
        {
            "filename": "sample_doc_2.pdf",
            "title": "Service Agreement",
            "content": [
                "This Service Agreement is between Acme Corporation and Provider.",
                "1. PAYMENT TERMS: Invoices paid within Net 30 days.",
                "2. LIMITATION OF LIABILITY: Total liability shall not exceed",
                "the amount of INR 50 lakhs.",
                "3. CONTRACT DURATION: This contract is valid for 12 months.",
                "4. SERVICES: Provider will deliver cloud management services."
            ]
        },
        {
            "filename": "sample_doc_3.pdf",
            "title": "Employment Policy",
            "content": [
                "Artikate AI Solutions - Standard Employment Policy",
                "1. ANNUAL LEAVE: Employees get 21 days annual leave.",
                "2. NOTICE PERIOD: Resignation notice period is 60 days.",
                "3. PROBATION PERIOD: Probation period is 6 months.",
                "4. WORKING HOURS: Hours are 9am to 6pm, Monday to Friday."
            ]
        }
    ]

    for doc in docs:
        filepath = os.path.join(DATA_DIR, doc["filename"])
        print(f"Creating {filepath}...")
        c = canvas.Canvas(filepath, pagesize=letter)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, 750, doc["title"])
        c.setFont("Helvetica", 10)
        y = 720
        for line in doc["content"]:
            c.drawString(50, y, line)
            y -= 20
        c.save()
    print("Sample PDFs recreated with properly wrapped text.")

def ingest_and_build_index():
    """Load PDFs, chunk them, and build the FAISS index using local embeddings."""
    print("Initializing Embedding Model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=64
    )

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DATA_DIR, filename)
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

    print(f"Embedding {len(all_chunks)} chunks...")
    embeddings = []
    metadata_list = []

    for chunk in all_chunks:
        emb = model.encode(chunk["text"])
        embeddings.append(emb)
        metadata_list.append(chunk["metadata"] | {"text": chunk["text"]})

    embeddings_np = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings_np)
    
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)

    if not os.path.exists(VECTOR_STORE_DIR):
        os.makedirs(VECTOR_STORE_DIR)

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata_list, f)
    
    print(f"Successfully built and saved index to {INDEX_PATH}")

def main():
    try:
        create_sample_pdfs()
        ingest_and_build_index()
    except Exception as e:
        print(f"Ingestion failed: {e}")

if __name__ == "__main__":
    main()
