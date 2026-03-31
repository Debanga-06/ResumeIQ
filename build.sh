#!/usr/bin/env bash
# build.sh — Render runs this once during deployment
set -e  # exit immediately if any command fails

echo "==> Installing Python dependencies..."
pip install -r requirements.txt

echo "==> Downloading spaCy model (en_core_web_md)..."
python -m spacy download en_core_web_md

echo "==> Pre-caching SBERT model (all-MiniLM-L6-v2)..."
# Download the model during build so first request isn't slow
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('SBERT model cached.')"

echo "==> Build complete."