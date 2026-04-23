# RAG Architecture Decisions

## 1. Chunking Strategy
We chose a **RecursiveCharacterTextSplitter** with a chunk size of **512 tokens** and **64-token overlap**.
- **Rationale**: Legal clauses are often lengthy and context-dependent. Overlap ensures that a clause split across chunks remains semantically meaningful in at least one of them. Tiktoken encoding ensures chunks fit within LLM context windows precisely.

## 2. Embedding Model
Used **text-embedding-3-small**.
- **Rationale**: It offers the best balance between performance, latency, and cost for production applications compared to older models like `ada-002`.

## 3. Vector Store
Used **FAISS (IndexFlatIP)** with L2 normalization.
- **Rationale**: Since the assessment requires a local setup, FAISS is the industry standard for efficient vector search. `IndexFlatIP` combined with normalized vectors provides accurate Cosine Similarity.

## 4. Hallucination Guard
Implemented a **Confidence Threshold (0.75)**.
- **Rationale**: In legal contexts, an incorrect answer is worse than no answer. By thresholding the similarity score, we refuse to answer queries that don't have a strong semantic match in the database.

## 5. Generation
Used **gpt-4o-mini**.
- **Rationale**: Fast, cost-effective, and highly capable of following complex system instructions like "Only answer from context" and "Cite sources".
