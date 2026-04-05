---
title: MHCVA Seren AI
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
short_description: Intelligent Psychiatric Voice Assistant (RAG + Dual LLM)
---

# SEREN AI - Mental Healthcare Voice Assistant

An intelligent, **production-ready** psychiatric assessment system powered by Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG). Dr. Aiden conducts empathetic clinical interviews, analyzes conversation patterns, and provides evidence-based psychological insights grounded in clinical psychopathology literature.

**Live Demo**: [SEREN AI on Hugging Face Spaces](https://huggingface.co/spaces/) *(Add your space URL here)*

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Cloud%20Vector%20DB-blue.svg)](https://www.pinecone.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Inference%20API-yellow.svg)](https://huggingface.co/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Amogh1221/Mental_HealthCare_VoiceAssistant/blob/main/LICENSE)



## Overview

**SEREN AI** is a production-grade mental healthcare assistant built with cutting-edge AI technology:

- **Dual-LLM Architecture**: Two specialized language models working together
  - **LLM1 (Conversational)**: Empathetic psychiatric interviewer with therapeutic rapport-building
  - **LLM2 (Analyst)**: Advanced pattern recognition across 6 psychological domains
  
- **Cloud-Powered RAG**: Semantic search over 122k+ psychiatric Q&A pairs via Pinecone Vector Database
  
- **Advanced Voice Processing**:
  - **STT (Speech-to-Text)**: Faster-Whisper for real-time audio transcription
  - **TTS (Text-to-Speech)**: Browser-native Web Speech API for natural voice output
  
- **Cloud Infrastructure**:
  - **Inference**: HuggingFace Inference API for LLM hosting
  - **Vector Database**: Pinecone Cloud for scalable semantic search
  - **Deployment**: Hugging Face Spaces with Docker containerization

This system is designed for **educational and research purposes** to demonstrate enterprise-grade AI in healthcare.

---

## Key Features

### Advanced Voice Interface
- Real-time Speech Recognition: Powered by Faster-Whisper (optimized for speed)
- Natural Voice Output: Browser-native TTS with zero server-side latency
- Cross-platform: Works on Chrome, Edge, Safari
- Offline-capable: Whisper runs locally on CPU (int8 quantization)

### Intelligent Dual-LLM System
- LLM1 (Qwen/Qwen2.5-7B-Instruct): Fluid conversational model
- LLM2 (meta-llama/Llama-3.3-70B-Instruct): State-of-the-art clinical reasoning
- Decision Logic: Intelligent CONTINUE/ANALYZE intent routing
- Context Awareness: Full conversation history with intelligent trimming

### Clinical Knowledge Integration
- **122k+ Q&A Pairs**: From "Sims' Symptoms in the Mind" textbook
- **Semantic Search**: Pinecone cloud vector database for instant retrieval
- **Metadata Tracking**: Source PDF, page numbers, confidence scores
- **Multiple Indexing Modes**: Assistant-only, Q&A pairs, or separate documents

### Multi-Domain Pattern Analysis
When analysis is triggered, LLM2 identifies patterns across:
- Emotional Themes: Mood states, anhedonia, emotional regulation
- Thinking Patterns: Rumination, catastrophizing, intrusive thoughts
- Behavioral Patterns: Sleep, appetite, social withdrawal, self-care
- Interpersonal Dynamics: Relationship patterns, social functioning
- Stressors: Identified triggers and life challenges
- Unclear Areas: Information gaps for targeted exploration

### Session Management
- **Persistent Sessions**: UUID-based conversation tracking
- **Smart History Trimming**: Maintains last 100 messages for performance
- **Automatic Cleanup**: Sessions are memory-resident (ephemeral)
- **Reset Capability**: Start fresh sessions instantly

### Production-Grade UI
- Tailwind CSS Styling: Modern, responsive design
- Real-time Feedback: Typing indicators, recording animations
- Mobile Ready: Works on tablets and phones
- Accessibility: WCAG-compliant with keyboard navigation

### Enterprise Deployment
- **Docker Support**: Production-ready containerization
- **Hugging Face Spaces**: One-click deployment
- **Environment Variables**: Full configuration via .env
- **Health Checks**: API monitoring and status endpoints

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                       Frontend (Browser)                           │
│  - HTML5 Tailwind UI                                               │
│  - Web Audio API (Microphone)                                      │
│  - Web Speech API (TTS)                                            │
└────────────────────┬───────────────────────────────────────────────┘
                     │ HTTP/WebSocket
                     ▼
┌────────────────────────────────────────────────────────────────────┐
│              FastAPI Backend (0.0.0.0:7860)                        │
│  - Session Management                                              │
│  - Request/Response Routing                                        │
└──────┬──────────────────────────────────────────────────────┬──────┘
       │                                                      │
       ▼                                                      ▼
  ┌─────────────┐                                  ┌──────────────────┐
  │  Whisper    │                                  │  HuggingFace     │
  │   (Local)   │                                  │   Inference API  │
  │   STT       │                                  │                  │
  │  CPU:int8   │                                  │  • LLM1 (7B)    │
  └─────────────┘                                  │  • LLM2 (70B)   │
                                                   └────────┬─────────┘
                                                            │
                                                   ┌────────▼───────────┐
                                                   │   Pinecone Cloud   │
                                                   │                    │
                                                   │  Vector DB         │
                                                   │  • 122k+ embeddings│
                                                   │  • Semantic Search │
                                                   │  • Cosine Metrics  │
                                                   └────────────────────┘
```

### Request Flow

1. **User Input**
   - Voice → Whisper STT → Text (local, fast)
   - OR manual text input

2. **Message Processing** (main.py)
   - Store in session history
   - Call LLM1 for conversational response
   - Decision: CONTINUE or ANALYZE

3. **CONTINUE Path** (Simple Chat)
   - LLM1 generates response
   - Add to history, return to user

4. **ANALYZE Path** (Clinical Analysis)
   - Retrieve clinical context from Pinecone
   - Call LLM2 for pattern analysis
   - Format analysis into clinical briefing
   - Call LLM1 again with briefing for informed response
   - Return enhanced response to user

5. **Audio Output**
   - Browser Web Speech API (no server overhead)
   - Natural, low-latency speech synthesis

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend Framework** | FastAPI | RESTful API, async processing |
| **LLM Inference** | HuggingFace Inference API | Cloud-hosted model serving |
| **LLM Models** | Qwen 2.5 (7B) & Llama 3.3 (70B) | Language understanding & generation |
| **Vector Database** | Pinecone Cloud | Semantic search over clinical knowledge |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | Convert text to vectors |
| **Speech Recognition** | Faster-Whisper (tiny.en) | Real-time audio transcription |
| **Frontend Framework** | HTML5 + Tailwind CSS | Responsive UI |
| **Text-to-Speech** | Web Speech API (Browser) | Native TTS (no server cost) |
| **Containerization** | Docker | Production deployment |
| **Deployment Platform** | Hugging Face Spaces | Free, auto-scaling inference |
| **Package Manager** | pip | Python dependency management |

---

## Prerequisites

### Required
- **Python 3.11+** (specified in Dockerfile)
- **FFmpeg** (for Whisper audio processing)
- **Git** (for cloning)

### API Keys & Credentials
1. **HuggingFace API Token** - [Get token](https://huggingface.co/settings/tokens)
   - Permissions: Read (for model inference)
   
2. **Pinecone API Key** - [Create free account](https://www.pinecone.io/)
   - Free tier includes 1 serverless index
   - Index name: `mhcva-index` (or customize)

3. **Pinecone Index Setup**
   - Dimension: 384 (all-MiniLM-L6-v2)
   - Metric: cosine
   - Cloud: AWS, Region: us-east-1 (or customize)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Amogh1221/Mental_HealthCare_VoiceAssistant.git
cd Mental_HealthCare_VoiceAssistant
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r Requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# HuggingFace Configuration
HUGGINGFACE_API_TOKEN=hf_your_token_here
LLM1_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM2_MODEL=meta-llama/Llama-3.3-70B-Instruct
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=mhcva-index

# Server Configuration
API_HOST=0.0.0.0
API_PORT=7860
```

### 5. Build Pinecone Vector Database

This step uploads the psychiatric knowledge base to Pinecone Cloud:

```bash
python built_vectorDB.py
```

**Setup Flow:**
```
1. Choose indexing mode (default: 1 - Assistant messages only)
2. Download Compumacy/Psych_data from HuggingFace
3. Process documents and create embeddings
4. Upload to Pinecone Cloud
5. Create index if it doesn't exist
```

**Processing Time**: 5-15 minutes (depends on bandwidth)

**Output**: Confirmation that knowledge base is live on Pinecone ✅

---

## Configuration

### LLM Models

Edit `.env` or `llm_engine.py` to change models:

```python
# In .env:
LLM1_MODEL=Qwen/Qwen2.5-7B-Instruct            # fluid conversation
LLM2_MODEL=meta-llama/Llama-3.3-70B-Instruct    # specialist reasoning
```

### Embedding Model

Edit `.env`:

```bash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Alternatives:
# sentence-transformers/all-mpnet-base-v2 (better quality, slower)
# sentence-transformers/paraphrase-MiniLM-L6-v2 (fastest)
```

### Vector Database Retrieval

Edit `rag_engine.py`:

```python
def retrieve(self, query: str, k: int = 8):  # k = number of results
    # MMR ensures diverse clinical context (avoids redundant symptom info)
    results = self.vectorstore.max_marginal_relevance_search(query, k=k)
```

### Whisper Audio Settings

Edit `llm_engine.py`:

```python
self.whisper = WhisperModel(
    "tiny.en",              # Options: tiny, base, small, medium, large
    device="cpu",           # "gpu" if CUDA available
    compute_type="int8",    # "float16" for higher accuracy
    download_root=models_dir
)

# Transcription settings:
segments, info = self.whisper.transcribe(
    audio_file,
    beam_size=1,            # 1 for speed, 5 for accuracy
    vad_filter=True         # Removes silence automatically
)
```

---

## Usage

### 1. Start the Application

```bash
uvicorn main:app --reload
```

Output:
```
INFO:     Uvicorn running on http://0.0.0.0:7860
INFO:     Application startup complete
```

### 2. Access the Interface

Open browser: **`http://localhost:7860`**

### 3. Using the Application

**Voice Mode:**
1. Click **Microphone** button
2. Speak naturally about your concerns
3. Click again to **stop recording**
4. Whisper transcribes, Dr. Aiden responds
5. Browser speaks response aloud

**Text Mode:**
1. Type your message in the **input box**
2. Press **Enter** or click **Send**
3. See Dr. Aiden's response appear
4. Optionally have browser read it aloud

**Session Management:**
1. Each conversation is a unique **session ID**
2. Click **"New Session"** to start fresh
3. Conversation history persists during session

---

## Project Structure

```
Mental_HealthCare_VoiceAssistant/
├── main.py                      # FastAPI app with endpoints
├── llm_engine.py               # LLM logic, Whisper STT, TTS
├── rag_engine.py               # Pinecone retrieval
├── built_vectorDB.py           # Vector DB builder
│
├── templates/
│   └── index.html              # Frontend UI (Tailwind)
│
├── models/                      # (Generated at runtime)
│   └── whisper/               # Cached Whisper models
│
├── Dockerfile                  # Docker configuration
├── Requirements.txt            # Python dependencies
├── .env                        # Configuration (create this)
├── .gitignore                  # Git ignore rules
├── .dockerignore              # Docker ignore rules
└── README.md                   # This file
```

---

## API Endpoints

### `POST /start`
**Initialize a new session**

**Response:**
```json
{
  "assistant_message": "Hello, I'm Dr. Aiden...",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

### `POST /reset`
**Reset current session and start fresh**

**Request:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response:**
```json
{
  "assistant_message": "Hello again...",
  "session_id": "new-uuid-here"
}
```

---

### `POST /chat_text`
**Send a text message**

**Request:**
```json
{
  "message": "I've been feeling sad for the past month",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response:**
```json
{
  "assistant_message": "I'm sorry to hear you've been struggling...",
  "intent": "CONTINUE"
}
```

Note: `intent` is either:
- `"CONTINUE"` - Regular conversation continues
- `"ANALYZE"` - (Internal) Analysis triggered, response includes insights

---

### `POST /transcribe`
**Transcribe audio file to text**

**Request:** (multipart/form-data)
- `audio` (file): WAV, MP3, OGG, etc.

**Response:**
```json
{
  "text": "I've been having trouble sleeping"
}
```

---



## Core Components

### LLM1: Conversational Psychiatrist (Dr. Aiden)

**Model**: Qwen/Qwen2.5-7B-Instruct (hosted on HuggingFace)

**Responsibilities**:
- Conduct empathetic psychiatric interviews
- Build therapeutic rapport
- Ask follow-up questions systematically
- Decide when to trigger analysis
- Provide psychoeducation

**System Prompt**: Detailed clinical guidelines
- Conversational tone guidelines
- CONTINUE vs ANALYZE decision logic
- Safety protocols for suicidal ideation
- Ethical boundaries

**Output Format**:
```json
{
  "assistant_message": "Empathetic response to patient",
  "intent": "CONTINUE | ANALYZE"
}
```

---

### LLM2: Clinical Pattern Analyst

**Model**: meta-llama/Llama-3.3-70B-Instruct (hosted on HuggingFace)

**Responsibilities**:
- Analyze conversation patterns
- Integrate clinical knowledge from RAG
- Identify psychological themes
- Suggest areas for exploration
- Guide treatment planning

**System Prompt**: Evidence-based clinical analysis framework
- 6-domain analysis structure
- Temporal context inclusion
- Safety concern flagging
- Clinical significance prioritization

**Output Format**:
```json
{
  "emotional_themes": ["Theme 1", "Theme 2", ...],
  "thinking_patterns": ["Pattern 1", "Pattern 2", ...],
  "behavioral_patterns": ["Behavior 1", "Behavior 2", ...],
  "interpersonal_dynamics": ["Dynamic 1", "Dynamic 2", ...],
  "stressors": ["Stressor 1", "Stressor 2", ...],
  "unclear_areas": ["Gap 1", "Gap 2", ...]
}
```

---

### RAG Engine (Pinecone)

**Vector Database**: Pinecone Serverless Cloud

**Dataset**:
- Source: Compumacy/Psych_data (HuggingFace)
- Content: 122k+ psychiatric Q&A pairs
- Origin: "Sims' Symptoms in the Mind" textbook
- Metadata: Source PDF, page number, confidence score

**Embedding Model**:
- `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Speed: ~1ms per query

**Indexing Modes** (selectable during setup):
1. **Assistant-only** (default): Only clinical answers (~122k docs)
2. **Q&A pairs**: Combined questions + answers (~122k docs)
3. **Both separate**: Individual questions and answers (~244k docs)

**Retrieval**:
- Maximal Marginal Relevance (MMR) search
- Configurable k (number of results)
- Default k=8 to pull a diverse array of clinical insights

---

### Speech Processing

#### Speech-to-Text (Whisper)
- **Model**: Faster-Whisper (tiny.en)
- **Quantization**: int8 (CPU-optimized)
- **Speed**: Real-time on standard CPU
- **Accuracy**: ~90% for clear English speech
- **Storage**: ~140 MB downloaded on first run

#### Text-to-Speech (Browser)
- **Engine**: Web Speech API (native browser)
- **Cost**: Zero server-side overhead
- **Latency**: <100ms response time
- **Quality**: Depends on OS default voices
- **Fallback**: Silent if browser doesn't support TTS

---

## Deployment Guide

### Local Development

```bash
# Terminal 1: Start backend
uvicorn main:app --reload --host 0.0.0.0 --port 7860

# Terminal 2: Access frontend
# Open http://localhost:7860
```

### Docker Build & Run

```bash
# Build image
docker build -t mental-health-assistant:latest .

# Run container
docker run -p 7860:7860 \
  -e HUGGINGFACE_API_TOKEN=hf_your_token \
  -e PINECONE_API_KEY=your_pinecone_key \
  -e PINECONE_INDEX_NAME=mhcva-index \
  mental-health-assistant:latest
```

### Hugging Face Spaces Deployment

**Step 1: Create New Space**
- Go to [huggingface.co/spaces/new](https://huggingface.co/spaces/new)
- Owner: Your HuggingFace username
- Space name: `mental-healthcare-voice-assistant` (or choose your own)
- Runtime: **Docker**
- Visibility: Public or Private

**Step 2: Clone Your Space**
```bash
git clone https://huggingface.co/spaces/your-username/mental-healthcare-voice-assistant
cd mental-healthcare-voice-assistant
```

**Step 3: Add Files**
```bash
# Copy all project files
cp -r /path/to/Mental_HealthCare_VoiceAssistant/* .

# Ensure Dockerfile exists and is configured for port 7860
# (Already done in this repo)
```

**Step 4: Set Environment Variables**
- Go to Space Settings → Repository secrets
- Add:
  - `HUGGINGFACE_API_TOKEN` = your token
  - `PINECONE_API_KEY` = your key
  - `PINECONE_INDEX_NAME` = mhcva-index

**Step 5: Push to Deploy**
```bash
git add .
git commit -m "Deploy mental healthcare assistant"
git push
```

**Step 6: Monitor Logs**
- Check Space logs as it builds
- Wait for "Running" status (2-5 minutes)
- Access at: `https://huggingface.co/spaces/your-username/mental-healthcare-voice-assistant`

### Environment Variables

```bash
# HuggingFace Configuration
HUGGINGFACE_API_TOKEN=hf_xxxxxxxxxxx      # Required
LLM1_MODEL=Qwen/Qwen2.5-7B-Instruct       # Optional
LLM2_MODEL=meta-llama/Llama-3.3-70B-Instruct # Optional

# Pinecone Configuration
PINECONE_API_KEY=xxxxxxxxxxxxxxx          # Required
PINECONE_INDEX_NAME=mhcva-index           # Optional

# Server Configuration
API_HOST=0.0.0.0                          # Required for HF Spaces
API_PORT=7860                             # Required for HF Spaces
```

---

## Troubleshooting

### HuggingFace API Errors

**Error**: `Unauthorized. Invalid token.`
```bash
# Solution: Verify token at https://huggingface.co/settings/tokens
# Token must have "Read" permission
# Regenerate if needed
```

**Error**: `Model not found`
```bash
# Solution: Model must exist on HuggingFace Hub
# Check: https://huggingface.co/Qwen (official models)
# Or use smaller models if quota limited
```

### Pinecone Connection Issues

**Error**: `Error creating index: Connection refused`
```bash
# Solution: Verify Pinecone credentials
# 1. Check PINECONE_API_KEY is correct
# 2. Ensure account has free tier access
# 3. Verify index name is unique
```

**Error**: `Index dimension mismatch`
```bash
# Solution: Rebuild index with correct dimensions
# all-MiniLM-L6-v2 requires dimension: 384
# Delete old index and rerun built_vectorDB.py
```

### Whisper Audio Issues

**Error**: `FFmpeg not found`
```bash
# Windows: Download from ffmpeg.org
# Mac: brew install ffmpeg
# Linux: sudo apt-get install ffmpeg
```

**Error**: `CUDA out of memory`
```bash
# Solution: Whisper uses CPU by default (int8)
# If forcing GPU, reduce compute_type="float32"
# Or use smaller model ("tiny.en")
```

### Docker Issues

**Error**: `Port 7860 already in use`
```bash
# Solution: Use different port
docker run -p 8000:7860 mental-health-assistant:latest
# Then access at http://localhost:8000
```

**Error**: `Docker build fails with network error`
```bash
# Solution: Network issues during dependency install
# Retry build: docker build --no-cache -t ...
# Or build locally first, then push to registry
```

### Session/History Issues

**Sessions constantly reset**
- Sessions are in-memory only (reset on server restart)
- This is intentional for privacy
- Deploy on persistent infrastructure if needed

**History growing too large**
- History auto-trims to last 100 messages
- If still slow, reduce MAX_HISTORY in main.py

---

## Privacy and Ethics

### Important Disclaimers

⚠️ **THIS IS NOT A REPLACEMENT FOR PROFESSIONAL MENTAL HEALTH CARE**

- **Educational & Research Only**: For learning purposes
- **No Medical Diagnoses**: Cannot replace psychiatrists
- **Not Emergency Ready**: Seek immediate help in crisis
- **No Data Storage**: Sessions deleted on server restart
- **No Guarantees**: Use at your own risk

### Data Privacy

✅ **What We Don't Do**:
- No data is stored persistently
- No user profiling or tracking
- No third-party data sharing
- No cloud logging (HF API doesn't log conversations)
- No analytics on conversations

⚠️ **What You Should Know**:
- Whisper transcription happens locally (no upload)
- LLM API calls go to HuggingFace servers (see their privacy policy)
- Vector search queries go to Pinecone (see their privacy policy)
- Browser TTS uses your OS voice (no cloud upload)

### Ethical Guidelines

✅ **Our Approach**:
- System acknowledges limitations as AI
- Encourages professional help when appropriate
- Takes suicidal ideation very seriously
- Maintains non-judgmental, culturally sensitive stance
- Never claims to diagnose or prescribe

---

## Limitations

1. **Not a Diagnostic Tool**: Cannot diagnose psychiatric conditions
2. **Limited Context**: May miss nuances in complex cases
3. **English Only**: Currently supports English language only
4. **No Real-time Conversation**: Some latency due to API calls
5. **No Crisis Intervention**: Cannot provide emergency response
6. **Pattern Matching Only**: Identifies patterns but cannot confirm diagnoses
7. **No Continuity**: Each session is independent (no long-term tracking)
8. **API Dependent**: Requires HuggingFace and Pinecone connectivity
9. **Bias Potential**: May reflect biases in training data
10. **No Phone Integration**: Web-only (no SMS/phone support)

---

## Roadmap

### Phase 1 (Current - Q2 2026)
- [x] Dual-LLM architecture
- [x] Cloud deployment (HF Spaces)
- [x] Pinecone RAG integration
- [x] Voice interface with Whisper
- [x] Production Docker setup

### Phase 2 (Q3 2026)
- [ ] Multi-language support (Spanish, Mandarin, French)
- [ ] Enhanced crisis detection with live resources
- [ ] Export conversation summaries (PDF/JSON)
- [ ] Session persistence with encrypted storage
- [ ] Advanced analytics dashboard

### Phase 3 (Q4 2026)
- [ ] Mobile app (React Native)
- [ ] Therapist collaboration features
- [ ] Real-time supervisor notifications
- [ ] Integration with EHR systems
- [ ] Long-term progress tracking

### Phase 4 (2027)
- [ ] Video/webcam integration
- [ ] Multimodal analysis (emotion detection)
- [ ] Advanced visualization tools
- [ ] Research publication tools
- [ ] Institutional licensing

---

## Contributing

Contributions welcome! Please follow these guidelines:

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR-USERNAME/Mental_HealthCare_VoiceAssistant.git
cd Mental_HealthCare_VoiceAssistant

# Create feature branch
git checkout -b feature/your-feature

# Make changes
# ...

# Test locally
pytest tests/  # (if tests exist)

# Format code
black *.py
flake8 *.py

# Commit and push
git add .
git commit -m "Add: Your feature description"
git push origin feature/your-feature

# Open Pull Request on GitHub
```

### Contribution Areas

- **Bug Fixes**: Report and fix issues
- **Documentation**: Improve README, add examples
- **Features**: New models, UX improvements
- **Performance**: Optimize API calls, caching
- **Testing**: Add unit/integration tests
- **Deployment**: Docker improvements, cloud configs

---

## License

This project is licensed under the **MIT License** - see [LICENSE](https://github.com/Amogh1221/Mental_HealthCare_VoiceAssistant/blob/main/LICENSE) for details.

---

## Acknowledgments

- **Femi Oyebode** - "Sims' Symptoms in the Mind" clinical foundation
- **Compumacy** - Psychiatric knowledge dataset
- **Hugging Face** - Model hosting and inference API
- **Pinecone** - Vector database infrastructure
- **Meta/OpenAI** - Whisper speech recognition
- **Alibaba/Qwen** - High-quality language models
- **FastAPI** - Modern web framework
- **Tailwind CSS** - UI styling

---

## Support and Contact

- **Issues**: [GitHub Issues](https://github.com/Amogh1221/Mental_HealthCare_VoiceAssistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Amogh1221/Mental_HealthCare_VoiceAssistant/discussions)
- **Email**: Add your contact email

---

## Crisis Resources

**If you or someone you know is in crisis, seek immediate professional help:**

### United States
- **National Suicide Prevention Lifeline**: 988 (call or text)
- **Crisis Text Line**: Text "HOME" to 741741
- **SAMHSA National Helpline**: 1-800-662-4357 (24/7, free, confidential)

### United Kingdom
- **Samaritans**: 116 123 (24/7)
- **Crisis Text Line**: Text SHOUT to 85258

### Canada
- **Suicide Prevention Lifeline**: 1-833-456-4566
- **Crisis Text Line**: Text HELLO to 741741

### International
- **International Association for Suicide Prevention**: [https://www.iasp.info/resources/Crisis_Centres/](https://www.iasp.info/resources/Crisis_Centres/)
- **Befrienders International**: [https://www.befrienders.org/](https://www.befrienders.org/)

---

## References and Resources

### Technical Documentation
1. [FastAPI Docs](https://fastapi.tiangolo.com/)
2. [Pinecone Docs](https://docs.pinecone.io/)
3. [HuggingFace Inference API](https://huggingface.co/inference-api)
4. [Faster-Whisper GitHub](https://github.com/guillaumekln/faster-whisper)
5. [Web Speech API MDN](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API)

### Clinical References
1. Oyebode, F. (2022). *Sims' Symptoms in the Mind: Textbook of Descriptive Psychopathology* (7th ed.). Elsevier.
2. Kaplan, H. I., & Sadock, B. J. (2014). *Kaplan and Sadock's Synopsis of Psychiatry* (11th ed.). Wolters Kluwer.
3. Diagnostic and Statistical Manual of Mental Disorders (DSM-5-TR). (2013). American Psychiatric Association.

### Research Papers
- *Natural Language Processing for Clinical Decision Support* - IEEE Review
- *Conversational AI in Healthcare* - Journal of Medical AI
- *Ethical Considerations in AI-Assisted Mental Health* - Ethics in AI

---

## Quick Start Summary

```bash
# 1. Clone
git clone https://github.com/Amogh1221/Mental_HealthCare_VoiceAssistant.git

# 2. Install
pip install -r Requirements.txt

# 3. Configure
# Create .env with HuggingFace token and Pinecone credentials

# 4. Build Vector DB
python built_vectorDB.py

# 5. Run
uvicorn main:app --host 0.0.0.0 --port 7860

# 6. Access
# Open http://localhost:7860
```

---

**Built for mental health awareness and education**

**If you find this project helpful, please star the repository!**

---

*Last Updated: April 5, 2026*
*Version: 2.0.0 (Production Cloud Deployment)*
*Status: Actively Maintained*
