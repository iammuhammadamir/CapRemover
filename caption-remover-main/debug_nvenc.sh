#!/bin/bash

echo "=========================================="
echo "NVENC Debug - Detailed Error Output"
echo "=========================================="

echo ""
echo "Testing NVENC with verbose output:"
echo "-----------------------------------"

if [ -f "data/examples/long.mp4" ]; then
    echo "Running: ffmpeg -hwaccel cuda -i long.mp4 -t 1 -c:v h264_nvenc -preset p4 -cq 28 test.mp4"
    echo ""
    
    ffmpeg -y -hwaccel cuda -i data/examples/long.mp4 -t 1 -c:v h264_nvenc -preset p4 -cq 28 /tmp/nvenc_test.mp4 2>&1 | tail -30
    
    echo ""
    echo "-----------------------------------"
    
    if [ -f "/tmp/nvenc_test.mp4" ]; then
        echo "✓ Test file created successfully!"
        rm -f /tmp/nvenc_test.mp4
    else
        echo "✗ Test file NOT created"
    fi
else
    echo "✗ Test video not found"
fi

echo ""
echo "Testing without hwaccel flag:"
echo "-----------------------------------"
if [ -f "data/examples/long.mp4" ]; then
    echo "Running: ffmpeg -i long.mp4 -t 1 -c:v h264_nvenc -preset p4 -cq 28 test.mp4"
    echo ""
    
    ffmpeg -y -i data/examples/long.mp4 -t 1 -c:v h264_nvenc -preset p4 -cq 28 /tmp/nvenc_test2.mp4 2>&1 | tail -20
    
    if [ -f "/tmp/nvenc_test2.mp4" ]; then
        echo "✓ Works WITHOUT hwaccel flag!"
        rm -f /tmp/nvenc_test2.mp4
    else
        echo "✗ Still fails without hwaccel"
    fi
fi

echo ""
echo "=========================================="

