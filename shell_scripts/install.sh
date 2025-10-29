#!/bin/bash

# Installation script for caption-remover
# Base container: runpod/pytorch_2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04/ssh

set -e  # Exit on error

echo "=========================================="
echo "Caption Remover Installation Script"
echo "=========================================="
echo ""


python -m pip install --upgrade pip


# Update apt and install system dependencies
echo "[1/15] Updating apt and installing ffmpeg..."
apt-get update && apt-get install -y ffmpeg
echo "✓ System dependencies installed"
echo ""

# Remove conda's ffmpeg (lacks libx264) and use system ffmpeg instead
echo "[2/15] Removing conda ffmpeg to use system ffmpeg..."
conda remove -y --force ffmpeg 2>/dev/null || true
echo "✓ Conda ffmpeg removed, system ffmpeg will be used"
echo ""

# Install PaddlePaddle GPU (CUDA 12.6)
echo "[3/15] Installing PaddlePaddle GPU for CUDA 12.6..."
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
echo "✓ PaddlePaddle GPU installed"
echo ""

# Install OpenCV
echo "[4/15] Installing opencv-python..."
pip install opencv-python
echo "✓ OpenCV installed"
echo ""

# Install PaddleOCR with all dependencies
echo "[5/15] Installing PaddleOCR with all dependencies..."
python -m pip install "paddleocr[all]"
echo "✓ PaddleOCR installed"
echo ""

# Fix PaddleX langchain dependency (requires older langchain with legacy import paths)
echo "[6/15] Installing langchain for PaddleX compatibility..."
pip install "langchain==0.0.354"
echo "✓ Langchain installed"
echo ""

# Install numpy (version < 2)
echo "[7/15] Installing numpy<2..."
pip install "numpy<2"
echo "✓ NumPy installed"
echo ""

# Install decord
echo "[8/15] Installing decord..."
pip install decord
echo "✓ Decord installed"
echo ""

# Install diffusers
echo "[9/15] Installing diffusers==0.29.2..."
pip install diffusers==0.29.2
echo "✓ Diffusers installed"
echo ""

# Install transformers
echo "[10/15] Installing transformers==4.41.1..."
pip install transformers==4.41.1
echo "✓ Transformers installed"
echo ""

# Install matplotlib
echo "[11/15] Installing matplotlib..."
pip install matplotlib
echo "✓ Matplotlib installed"
echo ""

# Install DiffuEraser dependencies
echo "[12/15] Installing DiffuEraser dependencies..."
pip install av==14.0.1 accelerate==0.25.0 imageio==2.34.1 einops==0.8.0 datasets==2.19.1 peft==0.13.2 scipy==1.13.1
echo "✓ DiffuEraser dependencies installed"
echo ""

# Install PyTorch for CUDA 11.8
echo "[13/15] Installing PyTorch 2.3.1 for CUDA 11.8..."
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
echo "✓ PyTorch installed"
echo ""

# Fix cuDNN conflict: PaddlePaddle (cuDNN 9) and PyTorch (cuDNN 8) both install to same location
# We need both .so.8 and .so.9 files to coexist, so we manually extract and copy cuDNN 8 files
echo "[14/15] Installing cuDNN 9 for PaddlePaddle..."
pip install --force-reinstall nvidia-cudnn-cu12==9.5.1.17
echo "✓ cuDNN 9 installed"
echo ""

echo "[15/15] Adding cuDNN 8 files for PyTorch (manual extraction)..."
cd /tmp && \
pip download --no-deps nvidia-cudnn-cu11==8.7.0.84 && \
python3 -c "import zipfile; zipfile.ZipFile('nvidia_cudnn_cu11-8.7.0.84-py3-none-manylinux1_x86_64.whl').extractall()" && \
cp nvidia/cudnn/lib/libcudnn*.so.8 /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/ && \
rm -rf nvidia *.whl && \
cd - > /dev/null
echo "✓ cuDNN 8 files added - both cuDNN 8 and 9 now available"
echo ""

echo "=========================================="
echo "✓ Installation complete!"
echo "=========================================="

