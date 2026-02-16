#!/bin/bash
echo "Starting Rox Quant Server..."

# Set path to venv python
PYTHON_EXEC="/Users/mac/Documents/trae_projects/word/.venv/bin/python"

# Kill any existing uvicorn processes to free up ports
pkill -f uvicorn || true

# Start the server using python -m uvicorn
"$PYTHON_EXEC" -m uvicorn app.main:app --host 127.0.0.1 --port 8081 --reload
