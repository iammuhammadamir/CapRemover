#!/bin/bash

echo "=========================================="
echo "Testing NVENC Availability"
echo "=========================================="

echo ""
echo "1. Checking FFmpeg NVENC encoders:"
echo "-----------------------------------"
if ffmpeg -encoders 2>/dev/null | grep -q nvenc; then
    echo "✓ NVENC encoders found:"
    ffmpeg -encoders 2>/dev/null | grep nvenc
else
    echo "✗ NVENC encoders NOT found"
    echo "  FFmpeg may not be compiled with --enable-nvenc"
fi

echo ""
echo "2. Checking NVIDIA GPU:"
echo "-----------------------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    echo "✓ NVIDIA GPU detected"
else
    echo "✗ nvidia-smi not found"
fi

echo ""
echo "3. Testing NVENC encoding (quick test):"
echo "-----------------------------------"
if [ -f "data/examples/long.mp4" ]; then
    echo "Testing with data/examples/long.mp4..."
    ffmpeg -y -hwaccel cuda -i data/examples/long.mp4 -t 1 -c:v h264_nvenc -preset p4 -cq 28 /tmp/nvenc_test.mp4 >/dev/null 2>&1
    
    if [ -f "/tmp/nvenc_test.mp4" ] && [ -s "/tmp/nvenc_test.mp4" ]; then
        echo "✓ NVENC test PASSED"
        echo "  GPU acceleration is working!"
        rm -f /tmp/nvenc_test.mp4
    else
        echo "✗ NVENC test FAILED"
        echo "  Pipeline will use CPU fallback"
    fi
else
    echo "⚠ Test video not found, skipping encode test"
fi

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="

