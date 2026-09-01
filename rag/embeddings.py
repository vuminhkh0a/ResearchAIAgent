"""Optimized embeddings with ONNX Runtime for 3-5x faster inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

from config.settings import settings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

_embeddings: Embeddings | None = None


def get_embeddings() -> Embeddings:
    """Load embeddings with ONNX optimization when available."""
    global _embeddings
    if _embeddings is None:
        if settings.embedding_use_onnx:
            _embeddings = _create_onnx_embeddings()
        else:
            _embeddings = _create_standard_embeddings()
    return _embeddings


def _create_onnx_embeddings() -> Embeddings:
    """Create ONNX-optimized embeddings for CPU inference."""
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
        import numpy as np
        
        model_name = settings.embedding_model
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load ONNX model
        ort_model = ORTModelForFeatureExtraction.from_pretrained(
            model_name,
            export=True,
            provider="CPUExecutionProvider",
            session_options=None,
        )
        
        class ONNXEmbeddings:
            def __init__(self, ort_model, tokenizer):
                self._ort_model = ort_model
                self._tokenizer = tokenizer
            
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                """Batch embed documents for efficiency."""
                batch_size = settings.embedding_batch_size
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    inputs = self._tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="np",
                    )
                    outputs = self._ort_model(**inputs)
                    # Mean pooling
                    embeddings = outputs.last_hidden_state.mean(axis=1)
                    # Normalize
                    norms = (embeddings ** 2).sum(axis=1, keepdims=True) ** 0.5
                    embeddings = embeddings / (norms + 1e-12)
                    all_embeddings.extend(embeddings.tolist())
                
                return all_embeddings
            
            def embed_query(self, text: str) -> list[float]:
                return self.embed_documents([text])[0]
        
        return ONNXEmbeddings(ort_model=ort_model, tokenizer=tokenizer)
    except ImportError:
        return _create_standard_embeddings()
    except Exception as e:
        print(f"ONNX embeddings failed ({e}), falling back to standard")
        return _create_standard_embeddings()


def _create_standard_embeddings() -> Embeddings:
    """Standard HuggingFace embeddings fallback."""
    from langchain_huggingface import HuggingFaceEmbeddings
    
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": settings.embedding_batch_size,
        },
        model_kwargs={"device": "cpu"},
    )