#!/bin/bash

# --- 1. Start the FastAPI Backend (Internal Only) ---
# Running FastAPI on the local loopback interface (127.0.0.1) 
# This ensures it ONLY listens for Streamlit's internal requests.
echo "Starting FastAPI server on internal port 8000..."
/usr/local/bin/uvicorn fastapi_part.main:app --host 127.0.0.1 --port 8000 &

# --- 2. Start the Streamlit Frontend in the Foreground ---
# We keep Streamlit on 8501 and rely on Render's port forwarding.
echo "Starting Streamlit frontend on port 8501..."
exec /usr/local/bin/streamlit run frontend/frontend.py --server.port 8501 --server.enableCORS true --server.enableXsrfProtection false

# NOTE: The 'exec' command ensures Streamlit takes over the main process, keeping the container alive.