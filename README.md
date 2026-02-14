# 🧠 SEREN AI - Mental Healthcare Voice Assistant

An intelligent psychiatric assessment system powered by Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG). SEREN AI (featuring Dr. Aiden) conducts empathetic clinical interviews through natural voice interactions, analyzes conversation patterns using dual-LLM architecture, and provides evidence-based psychological insights grounded in clinical psychopathology literature.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-purple.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 🌟 Features

- 🎙️ **Voice-Based Interface**: Natural speech recognition and text-to-speech for conversational interactions
- 💬 **Dual Input Modes**: 
  - **Voice Mode** (Primary): Hands-free conversation with Dr. Aiden
  - **Text Mode** (Secondary): Type-based chat for privacy or accessibility
- 🤖 **Dual-LLM Architecture**: 
  - **LLM1 (Llama3.2:3b)**: Empathetic psychiatric interviewer
  - **LLM2 (Nemotron-mini)**: Clinical pattern analyst
- 📚 **RAG-Enhanced**: Retrieves relevant clinical knowledge from 122K+ Q&A pairs based on "Sims' Symptoms in the Mind"
- 🔄 **Dynamic Analysis**: Automatically analyzes conversation patterns across 6 psychological domains when sufficient data is gathered
- 💾 **Session Management**: Persistent conversation history with seamless reset functionality
- 🎨 **Modern UI**: Beautiful loading animation, gradient branding, and responsive Tailwind CSS interface
- ⚡ **Real-time Processing**: Immediate feedback with typing indicators, status updates, and sticky controls
- 🛡️ **Privacy-First**: 100% local processing with Ollama - no data leaves your machine

---

## 🏗️ System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                   SEREN AI Web Interface                     │
│              (Voice Input + Text Chat + TTS)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                         │
│                   (Session Management)                       │
└─────────┬─────────────────────────────────┬─────────────────┘
          │                                 │
          ▼                                 ▼
┌──────────────────────┐         ┌──────────────────────────┐
│   LLM1: Dr. Aiden    │         │    Vector Database       │
│  (Psychiatrist)      │         │  (Clinical Knowledge)    │
│  - Conversation      │         │  - 122k+ Q&A pairs       │
│  - Empathy           │         │  - Sims' Symptoms        │
│  - Decision Making   │         │  - ChromaDB              │
└──────────┬───────────┘         └──────────┬───────────────┘
           │                                │
           │ (When ANALYZE triggered)       │
           ▼                                │
┌──────────────────────────────────────────▼────────────────┐
│              LLM2: Clinical Analyst                        │
│  - 6-Domain Pattern Recognition                            │
│  - Clinical Context Integration                            │
│  - Evidence-Based Analysis                                 │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
              (Enriched Response to User)
```

### Workflow

1. **Beautiful Loading Experience**: 
   - Animated SEREN AI logo with gradient glow effect
   - Session initialization in background
   - "Start Session" button when ready

2. **Initial Interview Phase**: 
   - LLM1 conducts empathetic psychiatric interview
   - Gathers symptoms, duration, severity, and functional impact
   - User can switch between voice and text input modes
   - Decides between CONTINUE (more questions) or ANALYZE (sufficient data)

3. **Analysis Phase** (when ANALYZE triggered):
   - Recent conversation retrieved from vector database for clinical context
   - LLM2 analyzes conversation and identifies patterns across 6 domains:
     - Emotional themes
     - Thinking patterns
     - Behavioral patterns
     - Interpersonal dynamics
     - Stressors
     - Unclear areas needing exploration

4. **Informed Response**:
   - LLM1 receives analysis and clinical context
   - Provides psychoeducation and targeted follow-up questions
   - Continues therapeutic conversation with deeper understanding
   - Responses are both displayed and spoken (in voice mode)

---

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Ollama**: [Download and install](https://ollama.ai/)
- **Ollama Models**: Download required models:
```bash
ollama pull llama3.2:3b
ollama pull nemotron-mini
ollama pull nomic-embed-text
```

- **Browser**: Chrome or Edge (recommended for voice features)
  - Brave: Works with additional configuration (see Troubleshooting)
  - Safari/Firefox: Limited speech recognition support

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/seren-ai.git
cd seren-ai
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Build Vector Database

This step downloads the psychiatric knowledge base (122K+ Q&A pairs) and builds a vector database:
```bash
python build_vectordb.py
```

**Interactive Setup:**
- Choose indexing mode: `1` (Assistant messages only - recommended)
- Direct document creation: `y` (faster - recommended)

**Output:**
- Creates `MH_db/` directory (~2-3 GB)
- Indexes 122,324 clinical Q&A pairs
- Takes 10-30 minutes depending on hardware

---

## 🎮 Usage

### Start the Application
```bash
uvicorn main:app --reload
```

The application will be available at: **http://localhost:8000**

### Using the Interface

#### 1. **Loading Screen**
   - Beautiful SEREN AI logo animation
   - Automatic session initialization
   - Click "Start Session" when ready

#### 2. **Voice Mode** (Default - 🎙️)
   - Grant microphone permissions when prompted
   - Click the microphone button to start recording
   - Speak naturally about your concerns
   - Click again to stop and send
   - Dr. Aiden will respond with voice + text

#### 3. **Text Mode** (Alternative - 💬)
   - Click the "💬 Text" toggle button
   - Type your message in the input field
   - Press Enter or click send button
   - Responses are text-only (no speech)

#### 4. **Session Controls**
   - **New Session**: Click top-right button to start fresh
   - **Sticky Controls**: Input controls stay at bottom (no scrolling needed)
   - **Auto-scroll**: Chat automatically scrolls to latest message

---

## 📁 Project Structure
```
seren-ai/
├── main.py                 # FastAPI application & API endpoints
├── llm_engine.py          # LLM configuration and system prompts
├── rag_engine.py          # Vector database retrieval logic
├── build_vectordb.py      # Script to build knowledge base
├── templates/
│   └── main.html          # Frontend UI (single-file web app)
├── MH_db/                 # Vector database (generated)
├── requirements.txt       # Python dependencies
├── presentation.html      # PBL presentation slides
└── README.md             # This file
```

---

## 🔧 Configuration

### LLM Models

Edit `llm_engine.py` to change models:
```python
# LLM1: Conversational psychiatrist
self.llm1 = ChatOllama(
    model="llama3.2:3b",      # Change model here
    temperature=0.6,
    format="json"
)

# LLM2: Clinical analyst
self.llm2 = ChatOllama(
    model="nemotron-mini",    # Change model here
    temperature=0.2,
    format="json"
)
```

### Embedding Model

Edit `build_vectordb.py` and `rag_engine.py`:
```python
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

### Retrieval Settings

Edit `rag_engine.py`:
```python
def retrieve(self, query: str, k: int = 5):  # Adjust k for more/fewer results
```

---

## 🧪 API Endpoints

### `POST /start`
Initializes a new session and returns the opening greeting.

**Response:**
```json
{
  "assistant_message": "Hello, I'm Dr. Aiden...",
  "session_id": "uuid-here"
}
```

### `POST /reset`
Resets the current session and starts a new one.

**Request:**
```json
{
  "session_id": "uuid-here"
}
```

**Response:**
```json
{
  "assistant_message": "Hello, I'm Dr. Aiden...",
  "session_id": "new-uuid-here"
}
```

### `POST /chat_text`
Sends a message and receives a response.

**Request:**
```json
{
  "message": "I've been feeling sad lately",
  "session_id": "uuid-here"
}
```

**Response:**
```json
{
  "assistant_message": "I'm sorry to hear that...",
  "intent": "CONTINUE"  // or "ANALYZE"
}
```

---

## 🎯 Core Components

### LLM1: Conversational Psychiatrist (Dr. Aiden)

**Model**: `llama3.2:3b` (configurable)

**Responsibilities**:
- Conduct empathetic psychiatric interviews
- Build rapport and trust
- Ask relevant follow-up questions
- Decide when to analyze (CONTINUE vs ANALYZE)
- Provide psychoeducation and support

**Output Format**:
```python
{
  "assistant_message": str,           # Response to patient
  "intent": "CONTINUE" | "ANALYZE"   # Next action
}
```

### LLM2: Clinical Pattern Analyst

**Model**: `nemotron-mini` (configurable)

**Responsibilities**:
- Analyze conversation patterns
- Integrate clinical knowledge from RAG
- Identify psychological patterns across 6 domains

**Output Format**:
```python
{
  "emotional_themes": List[str],
  "thinking_patterns": List[str],
  "behavioral_patterns": List[str],
  "interpersonal_dynamics": List[str],
  "stressors": List[str],
  "unclear_areas": List[str]
}
```

### RAG Engine

**Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`

**Knowledge Base**: 
- **Source**: [Compumacy/Psych_data](https://huggingface.co/datasets/Compumacy/Psych_data)
- **Content**: 122,324 psychiatric Q&A pairs from "Sims' Symptoms in the Mind"
- **Vector DB**: ChromaDB with persistent storage
- **Retrieval**: Top-5 most relevant clinical contexts per query

---

## 📊 Dataset Information

**Source**: [Compumacy/Psych_data](https://huggingface.co/datasets/Compumacy/Psych_data)

**Structure**:
```python
{
  'user_message': str,        # Clinical question
  'assistant_message': str,   # Evidence-based answer
  'metadata': {
    'source_pdf': str,        # Reference textbook
    'page_number': int,       # Page in source
    'confidence_score': float # Answer quality
  }
}
```

**Content**: Clinical knowledge extracted from "Sims' Symptoms in the Mind: Textbook of Descriptive Psychopathology" by Femi Oyebode (2022, Elsevier)

**Statistics**:
- Total Q&A pairs: 122,324
- Source: Authoritative psychiatric textbook
- Coverage: Descriptive psychopathology, symptom recognition, clinical assessment

---

## ⚙️ System Requirements

### Minimum Requirements
- **RAM**: 8 GB
- **Storage**: 5 GB free space
- **CPU**: 4 cores (Intel i5 or equivalent)
- **GPU**: Not required (CPU-only works fine)
- **Internet**: Required for initial setup only

### Recommended Requirements
- **RAM**: 16 GB
- **Storage**: 10 GB free space
- **CPU**: 8 cores (Intel i7/Ryzen 7 or better)
- **GPU**: NVIDIA GPU with 6+ GB VRAM (for faster inference)

---

## 🐛 Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama (if not running)
ollama serve
```

### Speech Recognition Not Working

**Chrome/Edge (Recommended):**
- Grant microphone permissions when prompted
- Ensure HTTPS or localhost is used

**Brave Browser:**
1. Click the shield icon (🛡️) in address bar
2. Allow microphone access
3. Set "Fingerprinting Protection" to "Allow all fingerprinting"
4. **Alternative**: Use Text Mode (💬) which works perfectly

**Safari/Firefox:**
- Limited Speech Recognition support
- Use Text Mode (💬) as alternative

### Vector Database Issues
```bash
# Rebuild vector database
python build_vectordb.py

# Check if database exists
ls -la MH_db/  # Linux/Mac
dir MH_db\    # Windows
```

### Memory Issues

If you encounter memory errors:
- Reduce `BATCH_SIZE` in `build_vectordb.py` (default: 1000 → try 500)
- Use smaller embedding models
- Reduce retrieval `k` value in `rag_engine.py` (default: 5 → try 3)

### Port Already in Use
```bash
# Kill process on port 8000
# Linux/Mac:
lsof -ti:8000 | xargs kill -9

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

---

## 🌐 Browser Compatibility

| Feature | Chrome/Edge | Brave (Configured) | Safari | Firefox |
|---------|------------|-------------------|--------|---------|
| Text Chat | ✅ Perfect | ✅ Perfect | ✅ Perfect | ✅ Perfect |
| Voice Input | ✅ Perfect | ⚠️ Needs Setup | ❌ Limited | ❌ Limited |
| Text-to-Speech | ✅ Perfect | ⚠️ Needs Setup | ✅ Works | ✅ Works |
| UI/Animations | ✅ Perfect | ✅ Perfect | ✅ Perfect | ✅ Perfect |

**Recommended**: Chrome or Edge for best experience

---

## 🔒 Privacy & Ethics

### Important Disclaimers

⚠️ **This is NOT a replacement for professional mental health care**

- This system is for **educational and research purposes only**
- It does **not provide medical diagnosis or treatment**
- Users experiencing mental health issues should **consult licensed professionals**
- In crisis situations, **contact emergency services or crisis hotlines immediately**

### Data Privacy

- ✅ **All conversations are stored in-memory only** (deleted on server restart)
- ✅ **100% local processing** - No data transmitted to external servers
- ✅ **Ollama runs locally** - No API calls to cloud services
- ✅ **No personally identifiable information is collected**
- ✅ **Sessions are ephemeral** - No persistent user data
- ✅ **Open source** - Fully auditable code

### Ethical Considerations

- The system **acknowledges its limitations as an AI**
- **Encourages users to seek professional help** when appropriate
- **Takes suicidal ideation seriously** and recommends immediate professional intervention
- Maintains a **non-judgmental, culturally sensitive approach**
- **Disclaimers are shown** before session starts

---

## 🚧 Limitations

1. **Not a Diagnostic Tool**: Cannot provide clinical diagnoses
2. **Limited Context**: May miss nuances in complex cases
3. **Language Support**: Currently English only
4. **Knowledge Cutoff**: Based on training data up to model's cutoff date
5. **No Crisis Intervention**: Not equipped for emergency situations
6. **Pattern Recognition**: May identify patterns but cannot confirm clinical conditions
7. **Voice Limitations**: Requires supported browser (Chrome/Edge)
8. **Local Processing**: Performance depends on hardware specifications

---

## 🛣️ Roadmap

### Planned Features
- [ ] Multi-language support (Hindi, Spanish, French)
- [ ] Enhanced crisis detection with immediate resource links
- [ ] Export conversation summaries (PDF/TXT)
- [ ] Integration with additional clinical databases
- [ ] Mobile app version (iOS/Android)
- [ ] Therapist collaboration features (anonymized data sharing)
- [ ] Advanced visualization of psychological patterns
- [ ] Long-term progress tracking across sessions
- [ ] Offline mode with pre-downloaded models
- [ ] Custom voice selection for Dr. Aiden

### Under Consideration
- [ ] Multi-user support with authentication
- [ ] Integration with wearables for mood tracking
- [ ] Group therapy session support
- [ ] Custom LLM fine-tuning on specialized datasets

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/AmazingFeature`
3. **Commit your changes**: `git commit -m 'Add some AmazingFeature'`
4. **Push to the branch**: `git push origin feature/AmazingFeature`
5. **Open a Pull Request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests (if available)
pytest tests/

# Format code
black .
flake8 .
```

### Contribution Areas

- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation improvements
- 🌍 Translations
- 🎨 UI/UX enhancements
- 🧪 Test coverage
- 🔧 Performance optimizations

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Sims' Symptoms in the Mind** by Femi Oyebode - Clinical knowledge foundation
- **Compumacy/Psych_data** - Psychiatric knowledge dataset on HuggingFace
- **LangChain** - RAG framework and LLM orchestration
- **Ollama** - Local LLM inference engine
- **FastAPI** - High-performance web framework
- **Tailwind CSS** - Modern UI styling
- **ChromaDB** - Vector database for embeddings
- **HuggingFace** - Embedding models and datasets

---

## 📞 Support & Contact

- **Email**: ag14gupta@gmail.com
- **GitHub Issues**: [Report a bug or request a feature](https://github.com/yourusername/seren-ai/issues)
- **Discussions**: [Join the community](https://github.com/yourusername/seren-ai/discussions)

---

## ⚕️ Crisis Resources

If you or someone you know is in crisis:

### India
- **KIRAN Mental Health Helpline**: 1800-599-0019 (24/7, Toll-Free)
- **Vandrevala Foundation**: 1860-2662-345 / 1800-2333-330
- **iCall**: 9152987821 (Mon-Sat, 8 AM - 10 PM)

### United States
- **National Suicide Prevention Lifeline**: 988
- **Crisis Text Line**: Text HOME to 741741

### International
- **Global Crisis Resources**: https://findahelpline.com
- **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/

---

## 📚 References

1. Oyebode, F. (2022). *Sims' Symptoms in the Mind: Textbook of Descriptive Psychopathology* (7th ed.). Elsevier.
2. [LangChain Documentation](https://python.langchain.com/)
3. [Ollama Documentation](https://ollama.ai/docs)
4. [ChromaDB Documentation](https://docs.trychroma.com/)
5. [FastAPI Documentation](https://fastapi.tiangolo.com/)
6. [Compumacy/Psych_data Dataset](https://huggingface.co/datasets/Compumacy/Psych_data)

---

## 🎓 Academic Use

This project was developed as part of:
- **Institution**: Manipal University Jaipur
- **Department**: Computer Science & Engineering
- **Course**: Project-Based Learning (PBL) 2026
- **Project Guide**: Dr. Arpita Baronia
- **Developer**: Amogh Gupta (23FE10CSE00819)

---

<div align="center">

### 💙 Built with care for mental health awareness and education

**SEREN AI** - *Serene support, when you need it*

---

⭐ **Star this repo if you find it helpful!** ⭐

[🌟 Star on GitHub](https://github.com/yourusername/seren-ai) • [🐛 Report Bug](https://github.com/yourusername/seren-ai/issues) • [💡 Request Feature](https://github.com/yourusername/seren-ai/issues)

---

*Remember: AI is a tool to assist, not replace, professional mental healthcare.*

</div>
