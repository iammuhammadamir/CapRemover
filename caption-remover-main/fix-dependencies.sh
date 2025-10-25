#!/bin/bash

# Fix script to restore working environment after xFormers broke dependencies
# This restores the exact state that install.sh created

set -e

echo "=========================================="
echo "Fixing Broken Dependencies"
echo "=========================================="
echo ""

# Uninstall xFormers (the culprit)
echo "[1/6] Uninstalling xFormers..."
pip uninstall xformers -y 2>/dev/null || true
echo "✓ xFormers removed"
echo ""

# Reinstall PyTorch 2.3.1 for CUDA 11.8
echo "[2/6] Reinstalling PyTorch 2.3.1 for CUDA 11.8..."
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
echo "✓ PyTorch 2.3.1 restored"
echo ""

# Reinstall PaddlePaddle GPU (CUDA 12.6)
echo "[3/6] Reinstalling PaddlePaddle GPU for CUDA 12.6..."
pip install --force-reinstall paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
echo "✓ PaddlePaddle GPU restored"
echo ""

# Fix cuDNN conflict: Install cuDNN 9 for PaddlePaddle
echo "[4/6] Reinstalling cuDNN 9 for PaddlePaddle..."
pip install --force-reinstall nvidia-cudnn-cu12==9.5.1.17
echo "✓ cuDNN 9 restored"
echo ""

# Add cuDNN 8 files for PyTorch
echo "[5/6] Adding cuDNN 8 files for PyTorch..."
cd /tmp
pip download --no-deps nvidia-cudnn-cu11==8.7.0.84
python3 -c "import zipfile; zipfile.ZipFile('nvidia_cudnn_cu11-8.7.0.84-py3-none-manylinux1_x86_64.whl').extractall()"
cp nvidia/cudnn/lib/libcudnn*.so.8 /opt/venv/lib/python3.10/site-packages/nvidia/cudnn/lib/
rm -rf nvidia *.whl
cd - > /dev/null
echo "✓ cuDNN 8 files restored"
echo ""

# Verify installation
echo "[6/6] Verifying installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import paddle; print(f'PaddlePaddle: {paddle.__version__}')"
echo "✓ Verification complete"
echo ""

echo "=========================================="
echo "✓ Dependencies Fixed!"
echo "=========================================="
echo ""
echo "You can now run: python main.py"
echo ""

