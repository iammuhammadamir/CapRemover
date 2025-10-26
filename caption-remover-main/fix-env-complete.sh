#!/bin/bash

# Complete environment fix script
# Fixes the dependency issues and installs missing packages

set -e  # Exit on error

echo "=========================================="
echo "Fixing Caption Remover Environment"
echo "=========================================="
echo ""

# Fix the nvidia-cublas version conflict
echo "[1/8] Fixing nvidia-cublas version conflict..."
pip install --force-reinstall nvidia-cublas-cu12==12.6.4.1
echo "✓ nvidia-cublas fixed"
echo ""

# Install PaddleOCR with all dependencies (missing from previous install)
echo "[2/8] Installing PaddleOCR with all dependencies..."
pip install "paddleocr[all]"
echo "✓ PaddleOCR installed"
echo ""

# Reinstall numpy<2 (requirement from install.sh)
echo "[3/8] Ensuring numpy<2..."
pip install "numpy<2"
echo "✓ NumPy version fixed"
echo ""

# Ensure decord is installed
echo "[4/8] Verifying decord installation..."
pip install decord
echo "✓ Decord verified"
echo ""

# Fix langchain for PaddleX compatibility (must be done before other packages)
echo "[5/8] Installing langchain for PaddleX compatibility..."
pip install "langchain==0.0.354"
echo "✓ Langchain installed"
echo ""

# Install OpenCV with compatible version for numpy<2
echo "[6/8] Installing opencv-python compatible with numpy<2..."
pip install "opencv-python==4.10.0.84"
echo "✓ OpenCV verified"
echo ""

# Ensure other key dependencies are present
echo "[7/8] Verifying other dependencies..."
pip install diffusers==0.29.2 transformers==4.41.1 matplotlib
pip install av==14.0.1 accelerate==0.25.0 imageio==2.34.1 einops==0.8.0 datasets==2.19.1 peft==0.13.2 scipy==1.13.1
echo "✓ Dependencies verified"
echo ""

# Verify cuDNN 8 files exist for PyTorch
echo "[8/8] Verifying cuDNN 8 files for PyTorch..."
CUDNN_PATH="/opt/venv/lib/python3.10/site-packages/nvidia/cudnn/lib"
if [ ! -f "$CUDNN_PATH/libcudnn.so.8" ]; then
    echo "⚠ cuDNN 8 files missing, adding them..."
    cd /tmp && \
    pip download --no-deps nvidia-cudnn-cu11==8.7.0.84 && \
    python3 -c "import zipfile; zipfile.ZipFile('nvidia_cudnn_cu11-8.7.0.84-py3-none-manylinux1_x86_64.whl').extractall()" && \
    cp nvidia/cudnn/lib/libcudnn*.so.8 "$CUDNN_PATH/" && \
    rm -rf nvidia *.whl && \
    cd - > /dev/null
    echo "✓ cuDNN 8 files added"
else
    echo "✓ cuDNN 8 files already present"
fi
echo ""

echo "=========================================="
echo "Verifying Installation..."
echo "=========================================="
echo ""

echo "PyTorch version:"
python -c "import torch; print(f'  {torch.__version__}')"
echo ""

echo "PaddlePaddle version:"
python -c "import paddle; print(f'  {paddle.__version__}')"
echo ""

echo "NumPy version:"
python -c "import numpy; print(f'  {numpy.__version__}')"
echo ""

echo "Checking PaddleOCR import..."
python -c "from paddleocr import TextDetection; print('  ✓ PaddleOCR import successful')"
echo ""

echo "Checking Decord import..."
python -c "from decord import VideoReader, cpu; print('  ✓ Decord import successful')"
echo ""

echo "=========================================="
echo "✓ Environment Fixed!"
echo "=========================================="
echo ""
echo "You can now run: python main.py"
echo ""

