#!/usr/bin/env bash
set -e
echo "==> Installing CPU-only PyTorch..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing remaining Python dependencies..."
pip install -r requirements.txt

echo "==> Downloading spaCy model (en_core_web_md)..."
python -m spacy download en_core_web_md

echo "==> Pre-caching SBERT model (all-MiniLM-L6-v2)..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('SBERT model cached.')"

echo "==> Build complete."
