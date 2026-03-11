#!/bin/bash
# Get the project root directory relative to this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Initializing Smart Secretary environment..."

# Run from project root to ensure module imports work correctly
cd "$PROJECT_ROOT"
./.venv/bin/python3 -m "module B Smart Secretary.secretary" "$@"
