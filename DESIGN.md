# DESIGN.md — RAG Pipeline Architecture Decisions

## 1. Chunking Strategy

I chose 512-token chunks with a 64-token overlap for this legal
document RAG system. Legal documents are structurally dense —
a single clause rarely fits in one sentence and frequently
references terms defined several sentences earlier. A 512-token
window is large enough to capture a complete clause including
its subject, conditions, and exceptions in a single chunk.
Smaller chunks of 128 or 256 tokens would frequently split a
clause mid-sentence, causing the retriever to return incomplete
context that the LLM cannot reason over correctly.

The 64-token overlap ensures that if a clause boundary falls
near the edge of a chunk, it still appears fully in the
adjacent chunk. Without overlap, a termination clause split
across two chunks would never appear complete in any single
retrieved result, causing retrieval failures even when the
document clearly contains the answer. I chose 64 tokens
specifically because it is roughly 2-3 sentences — enough
to capture cross-sentence context without creating excessive
redundancy in the index.

---

## 2. Embedding Model Choice

I chose sentence-transformers all-MiniLM-L6-v2 over OpenAI
text-embedding-3-small for this implementation. The primary
reason is that all-MiniLM-L6-v2 runs entirely locally with
no API calls, no cost per embedding, and no network latency
during ingestion. For a legal document corpus where ingestion
may involve hundreds of PDFs, paying per embedding token adds
up quickly and introduces a network dependency that could fail
mid-ingestion.

all-MiniLM-L6-v2 produces 384-dimensional embeddings and is
specifically trained on sentence-level semantic similarity,
which matches our retrieval task — finding the chunk most
semantically similar to a natural language legal question.
The tradeoff is that it is weaker than text-embedding-3-small
on complex semantic nuance, particularly for cross-lingual
queries. If the corpus contained non-English documents or
queries required deeper legal reasoning, I would switch to
text-embedding-3-large or a legal-domain fine-tuned model
such as legal-bert-base-uncased.

---

## 3. Vector Store Choice

I chose FAISS over Chroma, Pinecone, and Weaviate for this
implementation. FAISS runs entirely in-memory and on local
disk with no infrastructure setup — no Docker container, no
managed service, no API key. For a 500-document corpus
averaging 40 pages each, the total index fits comfortably
in RAM and FAISS flat index search completes in milliseconds.

Pinecone and Weaviate are excellent choices for production
systems requiring real-time updates, multi-tenant isolation,
and horizontal scaling, but they introduce unnecessary
operational complexity for a corpus of this size. Chroma is
a reasonable middle ground but adds a server process and
persistence layer that FAISS handles more simply via direct
file serialisation. The main limitation of FAISS is that it
does not support metadata filtering natively — filtering by
document name or date requires a post-retrieval step. For
this use case that tradeoff is acceptable.

---

## 4. Retrieval Strategy

I chose naive top-5 cosine similarity retrieval over hybrid
BM25 + dense retrieval and cross-encoder re-ranking. For a
corpus of 500 documents where queries are precise legal
questions, dense retrieval alone performs well because legal
questions contain specific terminology that maps directly
to the same terminology in the source documents.

Hybrid retrieval combining BM25 keyword search with dense
embeddings would improve recall for exact term matches like
specific vendor names or clause numbers. I would add this
if the precision@3 score dropped below 0.70 in evaluation.
Cross-encoder re-ranking using a model like
cross-encoder/ms-marco-MiniLM-L-6-v2 would further improve
ranking quality but adds 200-400ms latency per query, which
may be unacceptable for interactive use. The current top-5
retrieval achieves precision@3 of 1.00 on our test set,
which validates that naive retrieval is sufficient for this
corpus size.

---

## 5. Hallucination Mitigation Strategy

I implemented confidence-based answer refusal as the primary
hallucination mitigation strategy. After retrieval, I compute
the cosine similarity score between the query embedding and
each retrieved chunk embedding. If the highest similarity
score is below a threshold of 0.75, the system refuses to
answer and returns: "I don't have enough information in the
provided documents to answer this question."

I chose this approach over alternatives like fact verification
chains or self-consistency sampling for three reasons. First,
it is computationally free — the similarity scores are already
computed during retrieval and require no additional model calls.
Second, it is deterministic — the same query always produces
the same refusal decision, which is important for a legal
system where inconsistent behaviour erodes trust. Third, it
directly addresses the most common hallucination cause in RAG
systems: the model generating a plausible-sounding answer when
the retrieved context is irrelevant to the question.

The limitation of this approach is that a high similarity score
does not guarantee a correct answer — a chunk can be topically
relevant but still lack the specific detail the question asks
for. A stronger mitigation would add a secondary LLM call that
checks whether the retrieved context actually contains enough
information to answer the question before generating the final
response.

---

## 6. Scaling to 50,000 Documents

If the corpus grew from 500 to 50,000 documents, four
components would become bottlenecks:

**Ingestion pipeline:** Sequential PDF parsing and embedding
generation would take hours. The fix is to parallelise ingestion
using Python multiprocessing or a task queue like Celery, and
batch embedding generation to maximise GPU throughput.

**Vector store:** FAISS flat index search time grows linearly
with corpus size. At 50,000 documents with 40 pages each and
512-token chunks, the index would contain approximately 4 million
chunks. Flat search at this scale takes several seconds per query.
The fix is to switch to FAISS IVF (Inverted File Index) with
HNSW indexing, which reduces search time from O(n) to O(log n),
or migrate to Pinecone or Weaviate which handle this automatically
with managed infrastructure.

**Embedding model:** all-MiniLM-L6-v2 running on CPU becomes
a bottleneck during both ingestion and query time at this scale.
The fix is to deploy the embedding model on a GPU instance or
switch to a hosted embedding API with batching support.

**Retrieval quality:** With 4 million chunks, naive top-5
retrieval will return more irrelevant results. The fix is to
add hybrid BM25 + dense retrieval to improve recall, and add
a cross-encoder re-ranking step to improve precision of the
final top-3 results passed to the LLM.