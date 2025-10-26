#!/bin/bash

# Fix langchain version conflicts
# This removes incompatible langchain packages and installs the correct version

set -e  # Exit on error

echo "=========================================="
echo "Fixing Langchain Conflicts"
echo "=========================================="
echo ""

echo "[1/3] Uninstalling conflicting langchain packages..."
pip uninstall -y langchain-openai langchain-text-splitters langchain-core langchain 2>/dev/null || true
echo "✓ Conflicting packages removed"
echo ""

echo "[2/3] Installing langchain 0.0.354 (required for PaddleX)..."
pip install "langchain==0.0.354"
echo "✓ Langchain installed"
echo ""

echo "[3/3] Verifying installation..."
python -c "import langchain; print(f'Langchain version: {langchain.__version__}')"
echo ""

echo "=========================================="
echo "✓ Langchain conflicts resolved!"
echo "=========================================="

