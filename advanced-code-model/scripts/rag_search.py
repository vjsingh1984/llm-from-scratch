#!/usr/bin/env python3
"""
RAG-powered code search and Q&A (Stage 6).

Search codebase and answer questions with retrieved context.
"""

import torch
from pathlib import Path
import sys
from tokenizers import Tokenizer

sys.path.append(str(Path(__file__).parent.parent))
from src.model import create_model_from_config
from src.model.config import get_config
from src.rag.vector_store import VectorStore
from build_codebase_index import CodeEmbedder


def format_retrieved_context(results: list, max_chunks: int = 3) -> str:
    """Format retrieved code chunks for context."""
    context = "# Retrieved Code Context:\n\n"

    for i, (metadata, score) in enumerate(results[:max_chunks]):
        context += f"## Chunk {i+1} (similarity: {score:.3f})\n"
        context += f"File: {metadata['file_path']}\n"
        context += f"Lines: {metadata['start_line']}-{metadata['end_line']}\n\n"
        context += "```python\n"
        context += metadata['content']
        context += "\n```\n\n"

    return context


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG-powered code search")
    parser.add_argument("--checkpoint", type=str, default="models/multimodal_model_best.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--index-dir", type=str, default="data/rag",
                       help="Path to vector store directory")
    parser.add_argument("--model-size", type=str, default="large",
                       choices=["tiny", "medium", "large", "xlarge"])
    parser.add_argument("--device", type=str, default="mps",
                       choices=["cpu", "mps", "cuda"])
    parser.add_argument("--query", type=str, default=None,
                       help="Search query (if not provided, enters interactive mode)")
    parser.add_argument("--top-k", type=int, default=3,
                       help="Number of results to retrieve")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 60)
    print("RAG-Powered Code Search (Stage 6)")
    print("=" * 60)

    # Paths
    project_root = Path(__file__).parent.parent
    tokenizer_path = project_root / "data" / "tokenizer" / "tokenizer.json"
    index_dir = project_root / args.index_dir

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print("✓ Tokenizer loaded")

    # Load model
    print("\nLoading model...")
    config = get_config(args.model_size)
    config.use_rmsnorm = True
    config.use_rope = True

    model = create_model_from_config(config, architecture="dense", device=device)

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint['model_state_dict']

        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            state_dict = {
                key.replace('_orig_mod.', ''): value
                for key, value in state_dict.items()
            }

        model.load_state_dict(state_dict)
        model.eval()
        print("✓ Model loaded")
    except Exception as e:
        print(f"⚠️  Could not load checkpoint: {e}")
        print("Using untrained model for demo purposes")

    # Load vector store
    print("\nLoading vector store...")
    vector_store = VectorStore()
    try:
        vector_store.load(Path(index_dir))
    except Exception as e:
        print(f"⚠️  Could not load vector store: {e}")
        print("Please run build_codebase_index.py first")
        return

    # Create embedder
    embedder = CodeEmbedder(model, tokenizer, device=args.device)

    print("\n" + "=" * 60)
    print("Ready for search!")
    print("=" * 60)
    print(f"Indexed chunks: {len(vector_store)}")
    print(f"Top-k results: {args.top_k}\n")

    # Search function
    def search_codebase(query: str):
        print(f"\nQuery: {query}")
        print("-" * 60)

        # Embed query
        query_embedding = embedder.embed_text(query)

        # Search
        results = vector_store.search(query_embedding, k=args.top_k)

        if not results:
            print("No results found.")
            return

        # Display results
        for i, (metadata, score) in enumerate(results):
            print(f"\n### Result {i+1} (Similarity: {score:.4f})")
            print(f"File: {metadata['file_path']}")
            print(f"Lines: {metadata['start_line']}-{metadata['end_line']}")
            print(f"\nCode:")
            print("-" * 40)
            print(metadata['content'][:500])  # First 500 chars
            if len(metadata['content']) > 500:
                print("... (truncated)")
            print("-" * 40)

    # Single query mode or interactive
    if args.query:
        search_codebase(args.query)
    else:
        print("Interactive mode. Type 'quit' to exit.\n")
        while True:
            try:
                query = input("Search query: ")
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if query.strip():
                    search_codebase(query)
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break


if __name__ == "__main__":
    main()
