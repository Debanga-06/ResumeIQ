#!/usr/bin/env bash
set -e   # exit immediately on any error

echo "════════════════════════════════════════"
echo " ResumeIQ — Render Build Script"
echo "════════════════════════════════════════"

# ── Step 1: CPU-only PyTorch ────────────────────────────────────────────────
echo ""
echo "==> [1/5] Installing CPU-only PyTorch..."
pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
echo "    PyTorch (CPU) installed."

# ── Step 2: Rest of dependencies ────────────────────────────────────────────
echo ""
echo "==> [2/5] Installing requirements.txt..."
pip install -r requirements.txt --quiet
echo "    Dependencies installed."

# ── Step 3: spaCy model ─────────────────────────────────────────────────────
echo ""
echo "==> [3/5] Downloading spaCy en_core_web_md..."
python -m spacy download en_core_web_md
echo "    spaCy model ready."

# ── Step 4: Pre-cache SBERT model ───────────────────────────────────────────
echo ""
echo "==> [4/5] Pre-caching SBERT all-MiniLM-L6-v2..."
SENTENCE_TRANSFORMERS_HOME=/tmp/sbert_cache python -c "
import os, torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.set_num_threads(1)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
print('    SBERT model cached at /tmp/sbert_cache')
del model
"

# ── Step 5: Verify imports ───────────────────────────────────────────────────
echo ""
echo "==> [5/5] Verifying critical imports..."
python -c "
import spacy, sklearn, rank_bm25, rapidfuzz, fastapi, uvicorn
from google import genai
print('    All imports OK.')
"

echo ""
echo "════════════════════════════════════════"
echo " Build complete."
echo "════════════════════════════════════════"
