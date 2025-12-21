## RAG Guide (Stage 6)

**Retrieval-Augmented Generation for Long Context**

```
Stage 1-5 → Stage 6: RAG (Codebase Indexing + Retrieval)
```

---

## What is RAG?

RAG adds **retrieval** capabilities to overcome context length limits:
1. **Index codebase**: Chunk and embed code files
2. **Semantic search**: Find relevant code for queries
3. **Augmented generation**: Use retrieved context to answer questions

**Use cases:**
- "Where is the authentication logic?"
- "How does the payment system work?"
- "Find all API endpoints"
- "Explain the database schema"

---

## Quick Start

### Step 1: Build Codebase Index

```bash
python3 scripts/build_codebase_index.py \
  --checkpoint models/multimodal_model_best.pth \
  --codebase /path/to/your/codebase \
  --model-size large \
  --chunk-size 512 \
  --extensions .py,.js,.ts
```

**Output:**
- `data/rag/embeddings.npy` - Code embeddings
- `data/rag/metadata.json` - Chunk metadata

### Step 2: Search Codebase

```bash
# Interactive mode
python3 scripts/rag_search.py --top-k 5

# Single query
python3 scripts/rag_search.py --query "authentication logic"
```

---

## Architecture

```
Query: "Where is authentication?"
    ↓
Embed Query → [768-dim vector]
    ↓
Vector Search (cosine similarity)
    ↓
Top-K Retrieved Chunks
    ↓
Format as Context
    ↓
LLM + Context → Answer
```

---

## Components

### 1. Code Chunker
- Splits files into chunks (512 tokens each)
- Overlapping chunks for context preservation
- Handles .py, .js, .ts, .java, etc.

### 2. Vector Store
- Stores embeddings + metadata
- Cosine similarity search
- In-memory (can upgrade to FAISS for scale)

### 3. Code Embedder
- Uses trained model to embed code
- Mean pooling over sequence
- 768-dim embeddings

---

## Example Workflow

```python
# 1. Index codebase
chunks = chunker.chunk_codebase("./my_project")
# → 1,234 chunks

# 2. Embed chunks
embeddings = embedder.embed_batch(chunks)
# → [1234, 768] embeddings

# 3. Build vector store
store.add_batch(embeddings, chunks)
store.save("data/rag")

# 4. Search
query_emb = embedder.embed_text("auth logic")
results = store.search(query_emb, k=5)

# 5. Use in generation
context = format_context(results)
prompt = f"{context}\n\nQ: How does auth work?\nA:"
answer = model.generate(prompt)
```

---

## Production Improvements

1. **Better embeddings**: Use CodeBERT or dedicated code embedding models
2. **FAISS indexing**: Scale to millions of chunks
3. **Hybrid search**: Combine semantic + keyword search
4. **Re-ranking**: Re-rank results with cross-encoder
5. **Metadata filtering**: Filter by file type, date, author

---

**Next**: Stage 7: Agentic Workflows 🚀
