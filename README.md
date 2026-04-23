Paste this into Cursor/Antigravity:
Rewrite the README.md file at the root of the project.
Make it professional and well formatted using proper markdown.
Use proper markdown syntax throughout:
- Use triple backticks for ALL code blocks and commands
- Use proper headers with # ## ###
- Use tables where needed
- Use bullet points for lists
- Make sure every terminal command is inside a code block

Write exactly this content with proper markdown formatting:
Then paste this content after it:
# Artikate AI Assessment
### AI / ML / LLM Engineer — Technical Submission

---

## 🚀 Quick Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd artikate-ai-assessment
```

### 2. Create virtual environment
```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API key
```bash
cp .env.example .env
```
Open `.env` and add:
```
GROQ_API_KEY=your_groq_api_key_here
```
> Get a free Groq API key at: https://console.groq.com
> No credit card required

---

## 📁 Project Structure

```
artikate-ai-assessment/
├── README.md                        
├── DESIGN.md                        # RAG architecture decisions
├── ANSWERS.md                       # Written answers all sections
├── requirements.txt                 
├── .env.example                     
├── .gitignore                       
├── section_01_diagnosis/
│   └── answers.md                   # Diagnosis log + post-mortem
├── section_02_rag/
│   ├── pipeline.py                  # Main RAG interface
│   ├── ingest.py                    # PDF loading + chunking
│   ├── retriever.py                 # FAISS vector store
│   ├── generator.py                 # Groq LLM generation
│   ├── hallucination_guard.py       # Confidence + refusal
│   ├── evaluate.py                  # Evaluation harness
│   ├── eval_questions.json          # 10 Q&A pairs
│   ├── data/                        # 3 sample legal PDFs
│   └── vector_store/                # FAISS index (auto-generated)
├── section_03_classifier/
│   ├── train.py                     # DistilBERT fine-tuning
│   ├── predict.py                   # Single ticket inference
│   ├── evaluate.py                  # Metrics + confusion matrix
│   ├── generate_data.py             # Synthetic data via Groq
│   ├── test_latency.py              # pytest latency assertions
│   └── data/
│       ├── train.json               # 1000 synthetic examples
│       └── test.json                # 100 manually written examples
└── section_04_design/
    └── answers.md                   # Q-A and Q-C written answers
```

---

## 📊 Results Summary

| Section | Metric | Result |
|---|---|---|
| 02 RAG Pipeline | Precision@3 | ✅ 1.00 |
| 02 RAG Pipeline | Hallucination refusal | ✅ Working |
| 03 Classifier | Overall Accuracy | ✅ 0.73 |
| 03 Classifier | Latency test | ✅ 20/20 passed |
| 03 Classifier | Avg inference time | ✅ under 200ms |

---

## 🔍 Section 02 — RAG Pipeline

### How to run

```bash
# Step 1 — Create sample PDFs and build vector store
python -m section_02_rag.ingest

# Step 2 — Run demo queries
python -m section_02_rag.pipeline

# Step 3 — Run evaluation harness
python -m section_02_rag.evaluate
```

### Expected output — pipeline
```
Question: What is the notice period in the NDA with TechCorp?
Answer: The notice period is 30 days written notice.
Source: sample_doc_1.pdf | Page 1
Confidence: 0.85
```

### Expected output — evaluate
```
FINAL PRECISION@3: 1.00
```

### Models used
| Task | Model |
|---|---|
| Embeddings | all-MiniLM-L6-v2 (runs locally, no API) |
| Generation | llama3-8b-8192 via Groq API |

---

## 🤖 Section 03 — Ticket Classifier

### How to run

```bash
# Step 1 — Generate synthetic training data
python -m section_03_classifier.generate_data

# Step 2 — Fine-tune DistilBERT (15-30 minutes)
python -m section_03_classifier.train

# Step 3 — Evaluate classifier
python -m section_03_classifier.evaluate

# Step 4 — Run latency assertion test
pytest section_03_classifier/test_latency.py
```

### Expected output — evaluate
```
Overall Accuracy: 0.73

Per-class F1:
  billing             0.88
  technical_issue     0.56
  feature_request     0.77
  complaint           0.77
  other               0.62

Top confused pairs:
  technical_issue → feature_request : 5 times
  complaint → technical_issue       : 5 times
```

### Expected output — latency test
```
20 passed in 8.04s
```

### Models used
| Task | Model |
|---|---|
| Classifier | DistilBERT fine-tuned locally (no API needed) |
| Data generation | llama3-8b-8192 via Groq API |

---

## ✍️ Section 01 and Section 04

No code required for these sections.

- **Section 01** — Diagnosis log for 3 production failures
  and a non-technical post-mortem → see `ANSWERS.md`
- **Section 04** — Prompt injection mitigations (Question A)
  and on-premise LLM deployment (Question C) → see `ANSWERS.md`
- **DESIGN.md** — Full RAG architecture reasoning covering
  chunking strategy, embedding model, vector store choice,
  retrieval strategy, hallucination mitigation, and scaling plan

---

## 🔑 API Keys

| Key | Purpose | Get it free |
|---|---|---|
| `GROQ_API_KEY` | RAG generation + data generation | [console.groq.com](https://console.groq.com) |

> No paid APIs required. Embeddings and classification run locally.

---

## ⚙️ Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Embeddings | all-MiniLM-L6-v2 | Runs locally, no API cost |
| Vector store | FAISS | Zero infrastructure setup |
| Chunk size | 512 tokens, 64 overlap | Preserves full legal clauses |
| Hallucination guard | Confidence threshold 0.75 | Deterministic, no extra API call |
| Classifier | DistilBERT fine-tuned | Meets 500ms CPU latency constraint |

> Full reasoning in `DESIGN.md`

---

## 🛠️ Troubleshooting

| Error | Fix |
|---|---|
| `GROQ_API_KEY not found` | Check `.env` file has real key |
| `No module named fitz` | `pip install PyMuPDF` |
| `No module named faiss` | `pip install faiss-cpu` |
| `No module named sentence_transformers` | `pip install sentence-transformers` |
| `Vector store not found` | Run `python -m section_02_rag.ingest` first |
| `Model not found` | Run `python -m section_03_classifier.train` first |
| `Groq rate limit error` | Wait 60 seconds, free tier has per-minute limits |

---

## 📝 Notes

- `section_02_rag/vector_store/` is in `.gitignore` — run `ingest.py` to regenerate
- `section_03_classifier/model/` is in `.gitignore` — run `train.py` to regenerate  
- No API keys or `.env` files are committed to this repository