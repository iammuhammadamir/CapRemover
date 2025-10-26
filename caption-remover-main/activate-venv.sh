#!/bin/bash
# Quick helper to activate venv
# Usage: source activate-venv.sh

if [ -f /opt/venv/bin/activate ]; then
    source /opt/venv/bin/activate
    echo "✓ Virtual environment activated: /opt/venv"
    echo "Python: $(python --version)"
else
    echo "❌ Venv not found. Run: ../setup-project.sh"
fi
