import gc
import os
import threading
import warnings

warnings.filterwarnings("ignore", message=".*position_ids.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Module-level singletons (None until first use) ────────────────────────────
_nlp   = None          # spaCy pipeline
_sbert = None          # Sentence-Transformer

# ── One lock per model — prevents duplicate loading under concurrent requests ──
_nlp_lock   = threading.Lock()
_sbert_lock = threading.Lock()


# ── spaCy ─────────────────────────────────────────────────────────────────────
def get_nlp():
    """
    Return the spaCy pipeline, loading it on first call.

    Optimisations applied:
      - disable=["parser", "senter"]: we only use tok2vec, tagger, ner,
        attribute_ruler, lemmatizer. Disabling parser saves ~15MB and
        cuts load time by ~30%.
      - Double-checked locking: the outer `if` avoids acquiring the lock
        on every call (expensive); the inner `if` ensures only one thread
        actually loads the model.
    """
    global _nlp
    if _nlp is None:
        with _nlp_lock:
            if _nlp is None:          # re-check after acquiring lock
                import spacy
                print("[model_loader] Loading spaCy en_core_web_md …")
                _nlp = spacy.load(
                    "en_core_web_md",
                    disable=["parser", "senter"],  # not used — saves RAM + time
                )
                print("[model_loader] spaCy ready.")
                gc.collect()
    return _nlp


# ── SBERT ─────────────────────────────────────────────────────────────────────
def get_sbert():
    """
    Return the SentenceTransformer, loading it on first call.

    Optimisations applied:
      - device="cpu": forces CPU inference — no CUDA overhead.
      - torch.set_num_threads(1): prevents PyTorch spawning 8+ OS threads
        on a single-core free-tier container, which causes OOM via
        thread-stack overhead.
      - normalize_embeddings handled at call site, not stored.
    """
    global _sbert
    if _sbert is None:
        with _sbert_lock:
            if _sbert is None:
                import torch
                from sentence_transformers import SentenceTransformer

                # Single-threaded CPU — critical on free-tier containers
                torch.set_num_threads(1)

                print("[model_loader] Loading SBERT all-MiniLM-L6-v2 …")
                _sbert = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                    device="cpu",
                )
                print("[model_loader] SBERT ready.")
                gc.collect()
    return _sbert


# ── Warm-up (optional — call from FastAPI lifespan) ───────────────────────────
def warmup():
    """
    Pre-load both models during the startup event so the first real request
    is not slow. Only call this if RAM allows (i.e. paid tier).
    On free tier, leave lazy and let the first request trigger loading.
    """
    get_nlp()
    get_sbert()
    print("[model_loader] Warm-up complete.")


# ── Memory diagnostics (useful for debugging) ─────────────────────────────────
def memory_usage_mb() -> float:
    """Return current process RSS in MB (requires psutil)."""
    try:
        import psutil, os as _os
        process = psutil.Process(_os.getpid())
        return round(process.memory_info().rss / 1024 / 1024, 1)
    except ImportError:
        return -1.0
