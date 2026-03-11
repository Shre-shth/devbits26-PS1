#!/bin/bash
# Get the project root directory relative to this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Initializing Smart Secretary environment..."

# Set LD_LIBRARY_PATH for NVIDIA libraries
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PROJECT_ROOT/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:$PROJECT_ROOT/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib

# Run from project root to ensure module imports work correctly
cd "$PROJECT_ROOT"
./.venv/bin/python3 -m "module B Smart Secretary.secretary" "$@"
