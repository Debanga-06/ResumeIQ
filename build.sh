#!/usr/bin/env bash
set -e

echo "==> Upgrading pip build tools..."
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing CPU-only PyTorch..."
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing Python dependencies..."
python -m pip install -r requirements.txt

echo "==> Downloading spaCy model..."
python -m spacy download en_core_web_md

echo "==> Build complete."
