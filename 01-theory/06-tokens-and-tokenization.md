# Tokens and Tokenization

Tokens are the **basic units of text** that LLMs process.  
Tokenization is how text is **broken down into these units**.

---

## What is a Token?

- Can be a word, subword, or character  
- Example:  
  - Text: "Hello world"  
  - Tokens: ["Hello", " world"] (subword tokenization)  

---

## Why Tokens Matter

- Determines **model input size**  
- Directly affects **cost** when using APIs  
- Helps model understand structure and meaning of text  

---

## Tokenization Methods

1. **Word-level** → splits by words  
2. **Subword-level** → splits frequent patterns, balances vocabulary size  
3. **Byte-Pair Encoding (BPE)** → commonly used in modern LLMs  

---

## Key Takeaways

- Always be aware of tokens for **prompting and cost calculation**  
- Tokenization affects model efficiency and performance
