# Use Python 3.11 for stability
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (FFmpeg is required for Whisper's audio transcription)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for caching
COPY Requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r Requirements.txt

# Copy the entire backend codebase
COPY . .

# Expose the correct port (Hugging Face Spaces defaults to 7860)
EXPOSE 7860

# Command to run the FastAPI app
# We bind to 0.0.0.0 so external sources (Vercel) can talk to this Brain!
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
