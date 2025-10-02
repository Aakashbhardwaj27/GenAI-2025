# Context Windows in LLMs

A context window is the **maximum number of tokens** an LLM can “see” at once.

---

## Why Context Matters

- Determines how much **information the model can use** for generating answers  
- Example: GPT-4 may have 8k–128k token context windows  

---

## Strategies to Manage Long Contexts

- **Chunking** → break long documents into smaller pieces  
- **Sliding window** → overlap chunks to preserve context  
- **RAG (Retrieval-Augmented Generation)** → fetch relevant info dynamically  

---

## Key Takeaways

- Always design prompts and data inputs **within the model’s context window**  
- Use chunking or RAG for long documents to improve accuracy
