# Section 04: System Design

## Scaling the RAG Pipeline to 10 Million Documents

### 1. Vector Database Selection
- Move from FAISS-local to a distributed vector database like **Pinecone**, **Milvus**, or **Weaviate**.
- Use **HNSW** indexing for faster approximate nearest neighbor search at scale.
- Implement **Metadata Filtering** at the database level to narrow down the search space (e.g., filter by department or year).

### 2. Ingestion Pipeline
- Use a distributed task queue like **Celery** or **AWS Lambda** for parallel PDF processing.
- Implement a **Change Data Capture (CDC)** mechanism to only ingest new or updated documents.
- Use **OCR** (e.g., Tesseract or Azure Form Recognizer) for scanned PDF documents.

### 3. Retrieval Optimization
- **Hybrid Search**: Combine vector search (semantic) with BM25 (keyword) search to handle legal jargon and specific identifiers better.
- **Re-ranking**: Use a cross-encoder model (e.g., BGE-Reranker) on the top 50 retrieved chunks to select the best 5 for the LLM.
- **Query Expansion**: Use an LLM to generate alternative versions of the user's query to improve recall.

### 4. Infrastructure
- Deploy the model services (retriever/generator) as **microservices** on Kubernetes.
- Use **Redis** for caching common queries and their embeddings to reduce latency and API costs.
- Implement **Horizontal Pod Autoscaling (HPA)** based on request volume.
