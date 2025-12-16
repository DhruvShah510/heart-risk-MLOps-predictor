# --- STAGE 1: Dependency Installation ---
# Start from a lightweight Python base image (recommended for production)
FROM python:3.10-slim

# Set environment variables
# PYTHONUNBUFFERED ensures logs appear immediately in the console/Render dashboard
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy only the requirements file first. This layer rarely changes, speeding up subsequent builds.
COPY requirements.txt .

# Install all necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt


# --- STAGE 2: Application Setup ---

# Copy the saved model artifact to the root of the container
# This is crucial for main.py to load the model.
COPY rf_pipeline.joblib .

# Copy the application code folders (fastapi_part and frontend)
COPY fastapi_part $APP_HOME/fastapi_part
COPY frontend $APP_HOME/frontend

# Copy the startup script and make it executable
COPY start.sh .
RUN chmod +x start.sh

# Define the port the container will listen on (Streamlit's default is 8501)
EXPOSE 8501

# The main command to execute when the container starts
CMD ["./start.sh"]