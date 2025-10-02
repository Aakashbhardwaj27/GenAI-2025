# Embeddings and Vector Search

Embeddings are **numerical representations of text, images, or other data** that capture their meaning in a high-dimensional space.

---

## What Are Embeddings?

- Convert data into **vectors** (arrays of numbers)  
- Similar items have **closer vectors**  
- Example: "king" and "queen" vectors are close in semantic space  

---

## How They Work

1. Text → Embedding model → Vector  
2. Store vectors in a **vector database**  
3. Use **cosine similarity** or other distance metrics to find relevant results  

---

## Vector Databases

| Database | Use Case |
|----------|----------|
| pgvector | PostgreSQL integration for embeddings |
| Pinecone | Scalable cloud-native vector search |
| Weaviate | Knowledge graph + vector search |

---

## Practical Example

```python
from openai import OpenAI
client = OpenAI()

embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input="Hello world"
)
vector = embedding.data[0].embedding
