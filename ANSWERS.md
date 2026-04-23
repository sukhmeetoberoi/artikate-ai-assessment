# Artikate AI Assessment: Final Answers

This document provides a summary of all written sections for the assessment.

## Section 01: Diagnosis
Detailed analysis of model drift and diagnostic steps can be found in:
[section_01_diagnosis/answers.md](./section_01_diagnosis/answers.md)

## Section 02: RAG Design
Architecture decisions and chunking strategies are documented in:
[DESIGN.md](./DESIGN.md)

## Section 03: Classifier Evaluation
The classifier is a DistilBERT model fine-tuned on synthetic support tickets.
- **Accuracy**: ~95% (on synthetic test set)
- **Latency**: < 200ms (on CPU)
Detailed evaluation can be seen by running `python section_03_classifier/evaluate.py`.

## Section 04: System Design
Scaling strategies for 10M+ documents and hybrid search implementation:
[section_04_design/answers.md](./section_04_design/answers.md)
