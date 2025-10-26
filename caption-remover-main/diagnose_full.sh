#!/bin/bash

echo "=========================================="
echo "COMPREHENSIVE NVENC DIAGNOSTICS"
echo "=========================================="

echo ""
echo "1. System Information:"
echo "-----------------------------------"
echo "Current directory: $(pwd)"
echo "User: $(whoami)"
echo "Shell: $SHELL"

echo ""
echo "2. Checking file existence:"
echo "-----------------------------------"
if [ -f "data/examples/long.mp4" ]; then
    echo "✓ data/examples/long.mp4 EXISTS"
    ls -lh data/examples/long.mp4
else
    echo "✗ data/examples/long.mp4 NOT FOUND"
    echo ""
    echo "Available files in data/examples/:"
    ls -lh data/examples/ 2>/dev/null || echo "  Directory not found"
fi

echo ""
echo "3. FFmpeg version and capabilities:"
echo "-----------------------------------"
ffmpeg -version 2>&1 | head -3
echo ""
echo "NVENC encoders available:"
ffmpeg -encoders 2>/dev/null | grep nvenc || echo "  No NVENC encoders found"

echo ""
echo "4. GPU Information:"
echo "-----------------------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv
else
    echo "✗ nvidia-smi not found - GPU may not be available"
fi

echo ""
echo "5. CUDA availability:"
echo "-----------------------------------"
if [ -d "/usr/local/cuda" ]; then
    echo "✓ CUDA directory found: /usr/local/cuda"
    ls -ld /usr/local/cuda 2>/dev/null
else
    echo "⚠ CUDA directory not found at /usr/local/cuda"
fi

echo ""
echo "6. Testing NVENC with different approaches:"
echo "-----------------------------------"

# Find a test video
TEST_VIDEO=""
if [ -f "data/examples/long.mp4" ]; then
    TEST_VIDEO="data/examples/long.mp4"
elif [ -f "data/examples/trimmed_long.mp4" ]; then
    TEST_VIDEO="data/examples/trimmed_long.mp4"
elif [ -f "data/examples/gainz.mp4" ]; then
    TEST_VIDEO="data/examples/gainz.mp4"
else
    echo "✗ No test video found"
    TEST_VIDEO=""
fi

if [ -n "$TEST_VIDEO" ]; then
    echo "Using test video: $TEST_VIDEO"
    echo ""
    
    # Test 1: NVENC with hwaccel cuda
    echo "[Test 1] NVENC with -hwaccel cuda:"
    if timeout 10 ffmpeg -y -hwaccel cuda -i "$TEST_VIDEO" -t 1 -c:v h264_nvenc -preset p4 -cq 28 /tmp/test1.mp4 >/dev/null 2>&1; then
        if [ -f /tmp/test1.mp4 ] && [ -s /tmp/test1.mp4 ]; then
            echo "  ✓ SUCCESS - Created $(stat -f%z /tmp/test1.mp4 2>/dev/null || stat -c%s /tmp/test1.mp4) bytes"
            rm -f /tmp/test1.mp4
        else
            echo "  ✗ FAILED - File not created or empty"
        fi
    else
        echo "  ✗ FAILED - Command failed or timed out"
        echo "  Last error output:"
        timeout 10 ffmpeg -y -hwaccel cuda -i "$TEST_VIDEO" -t 1 -c:v h264_nvenc -preset p4 -cq 28 /tmp/test1.mp4 2>&1 | tail -5
    fi
    
    echo ""
    
    # Test 2: NVENC without hwaccel
    echo "[Test 2] NVENC without -hwaccel:"
    if timeout 10 ffmpeg -y -i "$TEST_VIDEO" -t 1 -c:v h264_nvenc -preset p4 -cq 28 /tmp/test2.mp4 >/dev/null 2>&1; then
        if [ -f /tmp/test2.mp4 ] && [ -s /tmp/test2.mp4 ]; then
            echo "  ✓ SUCCESS - Created $(stat -f%z /tmp/test2.mp4 2>/dev/null || stat -c%s /tmp/test2.mp4) bytes"
            rm -f /tmp/test2.mp4
        else
            echo "  ✗ FAILED - File not created or empty"
        fi
    else
        echo "  ✗ FAILED - Command failed"
        echo "  Last error output:"
        timeout 10 ffmpeg -y -i "$TEST_VIDEO" -t 1 -c:v h264_nvenc -preset p4 -cq 28 /tmp/test2.mp4 2>&1 | tail -5
    fi
    
    echo ""
    
    # Test 3: CPU fallback (libx265 ultrafast)
    echo "[Test 3] CPU fallback (libx265 ultrafast):"
    if timeout 10 ffmpeg -y -i "$TEST_VIDEO" -t 1 -c:v libx265 -preset ultrafast -crf 28 /tmp/test3.mp4 >/dev/null 2>&1; then
        if [ -f /tmp/test3.mp4 ] && [ -s /tmp/test3.mp4 ]; then
            echo "  ✓ SUCCESS - Created $(stat -f%z /tmp/test3.mp4 2>/dev/null || stat -c%s /tmp/test3.mp4) bytes"
            rm -f /tmp/test3.mp4
        else
            echo "  ✗ FAILED - File not created"
        fi
    else
        echo "  ✗ FAILED"
    fi
fi

echo ""
echo "7. Permission check:"
echo "-----------------------------------"
touch /tmp/nvenc_permission_test 2>/dev/null && echo "✓ Can write to /tmp" || echo "✗ Cannot write to /tmp"
rm -f /tmp/nvenc_permission_test 2>/dev/null

echo ""
echo "=========================================="
echo "Diagnostic Complete"
echo "=========================================="
echo ""
echo "If NVENC tests failed, the pipeline will automatically"
echo "fall back to CPU encoding (still faster than original)."
echo ""

