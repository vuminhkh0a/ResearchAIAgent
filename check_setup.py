#!/usr/bin/env python
"""Check system setup and recommend best LLM backend."""

from __future__ import annotations

import sys
import platform


def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"  PyTorch: {torch.__version__}")
            return True
        else:
            print("[NO] CUDA not available")
            return False
    except Exception as e:
        print(f"[NO] PyTorch/CUDA check failed: {e}")
        return False


def check_vllm():
    """Check if vLLM works."""
    try:
        import vllm
        print(f"[OK] vLLM installed: {vllm.__version__}")
        # Test import of C extension
        try:
            import vllm._C
            print("[OK] vLLM C extensions loaded")
            return True
        except ImportError as e:
            print(f"[NO] vLLM C extensions missing: {e}")
            return False
    except Exception as e:
        print(f"[NO] vLLM not installed: {e}")
        return False


def check_ollama():
    """Check if Ollama is available."""
    import httpx
    try:
        resp = httpx.get("http://127.0.0.1:11434/api/tags", timeout=3.0)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            names = [m.get("name") or m.get("model") or "" for m in models]
            print(f"[OK] Ollama running with models: {', '.join(names)}")
            return True
        else:
            print("[NO] Ollama not responding")
            return False
    except Exception as e:
        print(f"[NO] Ollama not available: {e}")
        return False


def check_embeddings():
    """Check embeddings setup."""
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        print("[OK] ONNX Runtime available")
    except ImportError:
        print("[WARN] ONNX Runtime not installed (embeddings will use PyTorch)")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("[OK] sentence-transformers installed")
    except ImportError:
        print("[NO] sentence-transformers not installed")


def main():
    print("=" * 60)
    print("System Setup Check")
    print("=" * 60)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print()
    
    cuda_ok = check_cuda()
    print()
    vllm_ok = check_vllm()
    print()
    ollama_ok = check_ollama()
    print()
    check_embeddings()
    print()
    
    print("=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    if platform.system() == "Windows":
        if vllm_ok and cuda_ok:
            print("→ Try vLLM (set LLM_BACKEND=vllm in .env)")
            print("  Note: Windows vLLM support is experimental")
        elif ollama_ok:
            print("→ Use Ollama (set LLM_BACKEND=ollama in .env) ✓ RECOMMENDED")
            print("  Most stable on Windows, good performance")
        else:
            print("→ Install Ollama: https://ollama.com/download/windows")
            print("  Then: ollama pull qwen3:8b")
    else:
        if cuda_ok and vllm_ok:
            print("→ Use vLLM (set LLM_BACKEND=vllm in .env) ✓ RECOMMENDED")
            print("  Best performance on Linux with NVIDIA GPU")
        elif ollama_ok:
            print("→ Use Ollama (set LLM_BACKEND=ollama in .env)")
        else:
            print("→ Install Ollama or vLLM")


if __name__ == "__main__":
    main()