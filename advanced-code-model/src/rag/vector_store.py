"""
Vector Store for RAG (Stage 6).

Stores code embeddings and enables semantic search.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json


class VectorStore:
    """Simple vector store for code embeddings."""

    def __init__(self, embedding_dim: int = 768):
        """
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.embeddings = []  # List of numpy arrays
        self.metadata = []  # List of metadata dicts
        self.index = None  # Optional FAISS index for large-scale

    def add(self, embedding: np.ndarray, metadata: Dict):
        """
        Add embedding with metadata.

        Args:
            embedding: [embedding_dim] embedding vector
            metadata: Dictionary with code info (file_path, function_name, etc.)
        """
        assert embedding.shape[0] == self.embedding_dim
        self.embeddings.append(embedding)
        self.metadata.append(metadata)

    def add_batch(self, embeddings: np.ndarray, metadata_list: List[Dict]):
        """
        Add batch of embeddings.

        Args:
            embeddings: [n, embedding_dim]
            metadata_list: List of n metadata dicts
        """
        for emb, meta in zip(embeddings, metadata_list):
            self.add(emb, meta)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for most similar embeddings.

        Args:
            query_embedding: [embedding_dim] query vector
            k: Number of results to return

        Returns:
            List of (metadata, similarity_score) tuples
        """
        if len(self.embeddings) == 0:
            return []

        # Stack embeddings
        db_embeddings = np.stack(self.embeddings)  # [n, embedding_dim]

        # Compute cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        db_norms = db_embeddings / (np.linalg.norm(db_embeddings, axis=1, keepdims=True) + 1e-8)

        similarities = np.dot(db_norms, query_norm)

        # Get top-k
        top_k_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_k_indices:
            results.append((self.metadata[idx], float(similarities[idx])))

        return results

    def save(self, path: Path):
        """Save vector store to disk."""
        path.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        if self.embeddings:
            embeddings_array = np.stack(self.embeddings)
            np.save(path / "embeddings.npy", embeddings_array)

        # Save metadata
        with open(path / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"✓ Saved vector store: {path}")
        print(f"  Entries: {len(self.embeddings)}")

    def load(self, path: Path):
        """Load vector store from disk."""
        # Load embeddings
        embeddings_path = path / "embeddings.npy"
        if embeddings_path.exists():
            embeddings_array = np.load(embeddings_path)
            self.embeddings = [emb for emb in embeddings_array]

        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

        print(f"✓ Loaded vector store: {path}")
        print(f"  Entries: {len(self.embeddings)}")

    def __len__(self):
        return len(self.embeddings)


class CodeChunker:
    """Chunk code files for embedding."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Args:
            chunk_size: Maximum tokens per chunk
            overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_file(self, file_path: Path, content: str) -> List[Dict]:
        """
        Chunk a code file.

        Args:
            file_path: Path to file
            content: File content

        Returns:
            List of chunk dictionaries
        """
        # Simple line-based chunking
        lines = content.split('\n')
        chunks = []

        # Function to estimate tokens (rough approximation)
        def estimate_tokens(text):
            return len(text.split())

        current_chunk = []
        current_size = 0

        for i, line in enumerate(lines):
            line_size = estimate_tokens(line)

            if current_size + line_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'file_path': str(file_path),
                    'start_line': i - len(current_chunk),
                    'end_line': i,
                    'content': chunk_text
                })

                # Start new chunk with overlap
                if self.overlap > 0:
                    overlap_lines = current_chunk[-self.overlap:]
                    current_chunk = overlap_lines + [line]
                    current_size = sum(estimate_tokens(l) for l in current_chunk)
                else:
                    current_chunk = [line]
                    current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append({
                'file_path': str(file_path),
                'start_line': len(lines) - len(current_chunk),
                'end_line': len(lines),
                'content': chunk_text
            })

        return chunks

    def chunk_codebase(self, codebase_dir: Path,
                      extensions: List[str] = ['.py', '.js', '.ts', '.java']) -> List[Dict]:
        """
        Chunk entire codebase.

        Args:
            codebase_dir: Root directory
            extensions: File extensions to include

        Returns:
            List of all chunks
        """
        all_chunks = []

        for ext in extensions:
            for file_path in codebase_dir.rglob(f'*{ext}'):
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        chunks = self.chunk_file(file_path, content)
                        all_chunks.extend(chunks)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

        print(f"✓ Chunked codebase: {codebase_dir}")
        print(f"  Total chunks: {len(all_chunks)}")

        return all_chunks
