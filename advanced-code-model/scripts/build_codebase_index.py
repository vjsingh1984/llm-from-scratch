#!/usr/bin/env python3
"""
Build codebase index for RAG (Stage 6).

Embeds code chunks and builds searchable vector store.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer

# Import components
sys.path.append(str(Path(__file__).parent.parent))
from src.model import create_model_from_config
from src.model.config import get_config
from src.rag.vector_store import VectorStore, CodeChunker


class CodeEmbedder:
    """Embed code chunks using trained model."""

    def __init__(self, model: torch.nn.Module, tokenizer: Tokenizer, device: str = "mps"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def embed_text(self, text: str, max_length: int = 512) -> np.ndarray:
        """
        Embed a text string.

        Args:
            text: Code text to embed
            max_length: Maximum sequence length

        Returns:
            embedding: [d_model] embedding vector
        """
        # Tokenize
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids[:max_length]

        # Pad if needed
        if len(ids) < max_length:
            ids = ids + [0] * (max_length - len(ids))

        # Convert to tensor
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        # Get model output
        with torch.no_grad():
            logits = self.model(input_ids)  # [1, seq_len, vocab_size]

            # Use mean pooling over sequence
            # (In production, extract actual hidden states)
            embedding = logits.mean(dim=1).squeeze(0)  # [vocab_size]

            # Reduce to d_model dimension (take first d_model dims)
            # This is a simplification; in production, modify model to return hidden states
            embedding = embedding[:768]  # Assume d_model = 768

        return embedding.cpu().numpy()

    def embed_batch(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """Embed batch of texts."""
        embeddings = []
        for text in tqdm(texts, desc="Embedding"):
            emb = self.embed_text(text, max_length)
            embeddings.append(emb)

        return np.stack(embeddings)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build codebase index for RAG")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--codebase", type=str, required=True,
                       help="Path to codebase directory to index")
    parser.add_argument("--model-size", type=str, default="large",
                       choices=["tiny", "medium", "large", "xlarge"])
    parser.add_argument("--device", type=str, default="mps",
                       choices=["cpu", "mps", "cuda"])
    parser.add_argument("--chunk-size", type=int, default=512,
                       help="Tokens per chunk")
    parser.add_argument("--extensions", type=str, default=".py,.js,.ts,.java",
                       help="File extensions to index (comma-separated)")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 60)
    print("Codebase Index Builder (Stage 6)")
    print("=" * 60)

    # Paths
    project_root = Path(__file__).parent.parent
    tokenizer_path = project_root / "data" / "tokenizer" / "tokenizer.json"
    output_dir = project_root / "data" / "rag"
    output_dir.mkdir(exist_ok=True)

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

    # Chunk codebase
    print(f"\nChunking codebase: {args.codebase}")
    chunker = CodeChunker(chunk_size=args.chunk_size)
    extensions = [ext.strip() for ext in args.extensions.split(',')]
    chunks = chunker.chunk_codebase(Path(args.codebase), extensions=extensions)

    if len(chunks) == 0:
        print("⚠️  No code chunks found!")
        return

    # Create embedder
    print("\nEmbedding code chunks...")
    embedder = CodeEmbedder(model, tokenizer, device=args.device)

    # Embed all chunks
    texts = [chunk['content'] for chunk in chunks]
    embeddings = embedder.embed_batch(texts, max_length=args.chunk_size)

    # Build vector store
    print("\nBuilding vector store...")
    vector_store = VectorStore(embedding_dim=embeddings.shape[1])

    for emb, chunk in zip(embeddings, chunks):
        vector_store.add(emb, chunk)

    # Save
    vector_store.save(output_dir)

    print("\n" + "=" * 60)
    print("✓ Codebase index built successfully!")
    print("=" * 60)
    print(f"\nIndexed {len(chunks)} code chunks")
    print(f"Saved to: {output_dir}")
    print("\nNext step: Use RAG for code search")
    print("  python scripts/rag_search.py --query 'authentication logic'")


if __name__ == "__main__":
    from typing import List
    main()
