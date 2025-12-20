"""Embedding model wrapper for memory retrieval."""

import logging
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity(query_embedding: np.ndarray, corpus_embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and corpus embeddings.
    
    Args:
        query_embedding: Query embedding vector (1D or 2D with shape [1, dim]).
        corpus_embeddings: Corpus embeddings matrix (2D with shape [n, dim]).
        
    Returns:
        Array of similarity scores.
    """
    # Ensure query is 1D
    if query_embedding.ndim == 2:
        query_embedding = query_embedding.squeeze(0)
    
    # Normalize vectors
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    corpus_norms = corpus_embeddings / (np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute dot products
    similarities = np.dot(corpus_norms, query_norm)
    
    return similarities


class EmbeddingModel:
    """Wrapper for sentence embedding models.
    
    Uses sentence-transformers library for encoding text into embeddings.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cpu",
    ):
        """Initialize embedding model.
        
        Args:
            model_name: Name or path of the sentence transformer model.
            device: Device to run the model on ('cpu' or 'cuda').
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension: Optional[int] = None

    def _ensure_model(self) -> None:
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Embedding model loaded, dimension: {self._dimension}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        self._ensure_model()
        return self._dimension

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts to encode.
            batch_size: Batch size for encoding.
            show_progress_bar: Whether to show progress bar.
            
        Returns:
            Numpy array of embeddings with shape [n, dim].
        """
        self._ensure_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into embedding.
        
        Args:
            text: Text to encode.
            
        Returns:
            Numpy array of embedding with shape [dim].
        """
        embeddings = self.encode([text])
        return embeddings[0]

