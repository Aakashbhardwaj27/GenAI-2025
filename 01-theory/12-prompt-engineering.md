# Prompt Engineering

Prompt engineering is the process of **crafting effective inputs for Large Language Models (LLMs)** to get the desired output.  
It is a critical skill for working with Generative AI.

---

## Why Prompt Engineering Matters

- LLMs respond **exactly to the input you give**.  
- Clear prompts = more accurate and useful results.  
- Poor prompts can cause irrelevant or hallucinated outputs.  

---

## Types of Prompts

| Type | Description | Example |
|------|------------|---------|
| Zero-shot | Ask the model to perform a task **without examples** | "Summarize this article in 3 sentences." |
| Few-shot | Provide 1–5 **examples in the prompt** to guide output | "Translate English to French: Hello → Bonjour" |
| Chain-of-Thought | Guide the model to reason **step by step** | "Step 1: Identify the question. Step 2: List relevant facts. Step 3: Answer the question." |

---

## Best Practices

1. **Be clear and specific**  
   - Instead of “Explain AI,” say: “Explain AI in simple terms for a beginner.”  

2. **Provide context**  
   - Include background information if needed for accurate answers.  

3. **Use examples when helpful**  
   - Few-shot prompts improve model reliability.  

4. **Limit scope**  
   - Avoid overly broad instructions to reduce ambiguity.  

5. **Iterate and refine**  
   - Test multiple prompt versions to find what works best.  

---

## Practical Examples

**Zero-shot Prompt**
```text
Summarize this text in 2 sentences:
"The company launched a new AI platform for developers..."
```

**Few-shot Prompt**
```text
Q: Translate English to French
A: Hello → Bonjour

Q: Translate English to French
A: Good morning → Bonjour

Q: Translate English to French
A: How are you?
```

**Chain-of-Thought Prompt**
```text
Question: If a train travels 60 km in 1 hour, how far will it go in 4 hours?

Step 1: Calculate distance for 1 hour.
Step 2: Multiply by 4.
Step 3: Provide the answer.
```

---

## Key Takeaways

- Prompt engineering is **as important as coding** in GenAI projects.  
- Clear, contextual, and example-driven prompts improve output quality.  
- Experimentation is key: **test, refine, and iterate**.  

> Pro Tip: Always think like a teacher — explain clearly what you want the model to do.
