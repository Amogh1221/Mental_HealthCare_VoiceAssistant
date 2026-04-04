# 🚀 Dr. Aiden 2.0 Deployment Guide (Split-Architecture)

This guide explains how to deploy your high-speed psychiatric assistant across two specialized platforms: **Hugging Face Spaces** for the AI brain and **Vercel** for the high-speed interface.

---

## 🧠 Part 1: Deploy Backend (Hugging Face Spaces)

Hugging Face Spaces is the specialized home for AI models. Since we use **Whisper** locally, we will use a **Docker Space**.

### 1. Create the Space
1. Log in to [Hugging Face](https://huggingface.co/).
2. Click **New Space**.
3. Name: `Mental_HealthCare_VoiceAssistant`.
4. SDK: Select **Docker**.
5. Hardware: Use the default **CPU Basic (Free)**. It is powerful enough for our `Whisper-Tiny` model.
6. Visibility: Public.

### 2. Add Secrets
Before pushing code, you must add your API keys:
1. Go to **Settings** in your Space.
2. Scroll to **Variables and Secrets**.
3. Add these **New Secrets**:
   - `HUGGINGFACE_API_TOKEN`: (Your HF Token)
   - `PINECONE_API_KEY`: (Your Pinecone Key for RAG)
   - `PINECONE_ENVIRONMENT`: (e.g. us-east-1-aws)
   - `PINECONE_INDEX_NAME`: (e.g. mhcva-db)

### 3. Create the `Dockerfile` in your local project:
Create a file named `Dockerfile` (no extension) in your root folder:

```dockerfile
# Use Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for audio/ffmpeg)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy local requirements
COPY Requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r Requirements.txt

# Copy everything else
COPY . .

# Expose the standard FastAPI port (7860 for HF Spaces)
EXPOSE 7860

# Run with uvicorn on the correct port for Spaces
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 4. Push to Spaces
You can connect your GitHub repo to the Space or manually upload files. The Space will automatically build and start the server.

---

## 🎨 Part 2: Deploy Frontend (Vercel)

Vercel will host your `main.html` as a high-speed static site.

### 1. Update `main.html` for Cross-Origin (Required)
I've added a `BACKEND_URL` variable to your `main.html`. Once your Space is live, just change that URL to your Space URL!

### 2. Deploy to Vercel
1. Log in to [Vercel](https://vercel.com/).
2. Create a new folder on your computer named `frontend`.
3. Copy **`templates/main.html`** into that folder and rename it to **`index.html`**.
4. Drag and drop that folder onto the Vercel dashboard.
5. **BOOM!** Your frontend is live.

---

## 🛠️ Essential Final Step: Link the Two
Once your Hugging Face Space is "Running", you will see a small **"Embed"** or **"Public URL"** (e.g., `https://amogh-mhcva.hf.space`). 
Just update that single line in your `index.html` and you are good to go!
