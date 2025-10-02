# Retrieval-Augmented Generation (RAG)

RAG combines **retrieval systems** (search) with **generation models** (LLMs) to answer questions using external knowledge.

---

## How RAG Works

1. Query is received  
2. Relevant documents are retrieved from a **vector database**  
3. LLM generates a response using **retrieved context**  

---

## Why RAG Matters

- Extends model **knowledge beyond training data**  
- Handles **long documents** or domain-specific knowledge  
- Reduces hallucinations by grounding outputs in real sources  

---

## Example Workflow

```text
Query: "What is the company’s 2025 AI roadmap?"
Step 1: Search relevant PDFs/Docs using embeddings
Step 2: Retrieve top 3–5 documents
Step 3: Pass retrieved info to LLM to generate concise answer
