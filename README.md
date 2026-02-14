#  AI Psychiatrist - Dr. Aiden

An intelligent psychiatric assessment system powered by Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG). Dr. Aiden conducts empathetic clinical interviews, analyzes conversation patterns, and provides evidence-based psychological insights grounded in clinical psychopathology literature.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

##  Features

-  Voice-Based Interface: Natural speech recognition and text-to-speech for conversational interactions
-  Dual-LLM Architecture: 
  - LLM1 (Conversational Agent): Empathetic psychiatric interviewer
  - LLM2 (Clinical Analyst): Pattern recognition and clinical analysis
-  RAG-Enhanced: Retrieves relevant clinical knowledge from Sims' Symptoms in the Mind
-  Dynamic Analysis: Automatically analyzes conversation patterns when sufficient data is gathered
-  Session Management: Persistent conversation history across interactions
-  Modern UI: Beautiful, responsive Tailwind CSS interface
-  Real-time Processing: Immediate feedback with typing indicators and status updates

---

##  System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                       │
│                  (Voice + Text Chat UI)                     │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                        │
│                   (Session Management)                      │
└───────────┬─────────────────────────────────┬───────────────┘
            │                                 │
            ▼                                 ▼
┌──────────────────────┐         ┌──────────────────────────┐
│   LLM1: Dr. Aiden    │         │    Vector Database       │
│  (Psychiatrist)      │         │  (Clinical Knowledge)    │
│  - Conversation      │         │  - 122k+ Q&A pairs       │
│  - Empathy           │         │  - Sims' Symptoms        │
│  - Decision Making   │         └──────────┬───────────────┘
└──────────┬───────────┘                    │
           │                                │
           │ (When ANALYZE triggered)       │
           ▼                                │
┌───────────────────────────────────────────▼───────────────┐
│              LLM2: Clinical Analyst                       │
│  - Pattern Recognition                                    │
│  - Clinical Context Integration                           │
│  - Evidence-Based Analysis                                │
└────────────────────────┬──────────────────────────────────┘
                         │
                         ▼
            (Enriched Response to User)
```

### Workflow

1. **Initial Interview Phase**: 
   - LLM1 conducts empathetic psychiatric interview
   - Gathers symptoms, duration, severity, and functional impact
   - Decides between CONTINUE (more questions) or ANALYZE (sufficient data)

2. **Analysis Phase** (when ANALYZE triggered):
   - Recent conversation retrieved from vector database for clinical context
   - LLM2 analyzes conversation and identifies:
     - Emotional themes
     - Thinking patterns
     - Behavioral patterns
     - Interpersonal dynamics
     - Stressors
     - Unclear areas needing exploration

3. **Informed Response**:
   - LLM1 receives analysis and clinical context
   - Provides psychoeducation and targeted follow-up questions
   - Continues therapeutic conversation with deeper understanding

---

##  Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- Ollama models downloaded:
```bash
  ollama pull llama3.2:3b
  ollama pull nemotron-mini
  ollama pull nomic-embed-text
```

---

##  Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ai-psychiatrist.git
cd ai-psychiatrist
```

### 2. Create Virtual Environment
```bash
python -m venv venv

venv\Scripts\activate

source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Build Vector Database

This step downloads the psychiatric knowledge base and builds a vector database:
```bash
python build_vectordb.py
```

**Options during setup:**
- Choose indexing mode (recommended: 1 - Assistant messages only)
- Direct document creation (faster, recommended: y)

This will create a `MH_db/` directory containing your vector database (~2-3 GB).

**Note**: This process may take 10-30 minutes depending on your hardware.

---

##  Usage

### Start the Application
```bash
uvicorn main:app --reload
```

The application will be available at: `http://localhost:8000`

### Using the Interface

1. **Grant Microphone Permissions**: Click "Allow" when prompted by your browser
2. **Start Speaking**: Click the microphone button to begin
3. **Conversation**: 
   - Speak naturally about your concerns
   - Dr. Aiden will ask follow-up questions
   - Click the microphone again to stop recording
4. **Analysis**: After sufficient conversation, Dr. Aiden will analyze patterns and provide insights
5. **New Session**: Click "New Session" to start over

---

##  Project Structure
```
ai-psychiatrist/
├── main.py                 # FastAPI application & API endpoints
├── llm_engine.py          # LLM configuration and prompts
├── rag_engine.py          # Vector database retrieval logic
├── build_vectordb.py      # Script to build knowledge base
├── templates/
│   └── main.html          # Frontend UI (Tailwind CSS)
├── MH_db/                 # Vector database (created by build_vectordb.py)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

---

##  Configuration

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

##  API Endpoints

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

##  Core Components

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
  "assistant_message": str,      # Response to patient
  "intent": "CONTINUE" | "ANALYZE"  # Next action
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
- Source: [Compumacy/Psych_data](https://huggingface.co/datasets/Compumacy/Psych_data)
- Content: 122k+ psychiatric Q&A pairs from Sims' Symptoms in the Mind
- Vector DB: ChromaDB with persistent storage

---

##  Dataset Information

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

---

## ⚙️ System Requirements

### Minimum Requirements
- **RAM**: 8 GB
- **Storage**: 5 GB free space
- **CPU**: 4 cores
- **GPU**: Not required (but recommended for faster processing)

### Recommended Requirements
- **RAM**: 16 GB
- **Storage**: 10 GB free space
- **CPU**: 8 cores
- **GPU**: NVIDIA GPU with 6+ GB VRAM (for GPU acceleration)

---

##  Privacy & Ethics

### Important Disclaimers

 **This is NOT a replacement for professional mental health care**

- This system is for educational and research purposes only
- It does not provide medical diagnosis or treatment
- Users experiencing mental health issues should consult licensed professionals
- In crisis situations, contact emergency services or crisis hotlines

### Data Privacy

- All conversations are stored in-memory only
- No data is transmitted to external servers (except Ollama locally)
- Sessions are ephemeral and deleted on server restart
- No personally identifiable information is collected

### Ethical Considerations

- The system acknowledges its limitations as an AI
- Encourages users to seek professional help when appropriate
- Takes suicidal ideation seriously and recommends immediate professional intervention
- Maintains a non-judgmental, culturally sensitive approach

---

##  Limitations

1. **Not a Diagnostic Tool**: Cannot provide clinical diagnoses
2. **Limited Context**: May miss nuances in complex cases
3. **Language Support**: Currently English only
4. **Knowledge Cutoff**: Based on training data up to model's cutoff date
5. **No Crisis Intervention**: Not equipped for emergency situations
6. **Pattern Recognition**: May identify patterns but cannot confirm clinical conditions

---

##  Roadmap

- [ ] Multi-language support
- [ ] Enhanced crisis detection and resources
- [ ] Export conversation summaries
- [ ] Integration with more clinical databases
- [ ] Mobile app version
- [ ] Therapist collaboration features
- [ ] Advanced visualization of patterns
- [ ] Long-term progress tracking

---

##  Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
```

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Acknowledgments

- **Sims' Symptoms in the Mind** by Femi Oyebode - Clinical knowledge foundation
- **Compumacy/Psych_data** - Psychiatric knowledge dataset
- **LangChain** - RAG framework
- **Ollama** - Local LLM inference
- **FastAPI** - Web framework
- **Tailwind CSS** - UI styling

---

##  Support & Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/ai-psychiatrist/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-psychiatrist/discussions)
- **Email**: your.email@example.com

---

## Crisis Resources

If you or someone you know is in crisis:

**India**  
- KIRAN Mental Health Helpline: 1800-599-0019 (24/7)

**Global**  
- Find local crisis helplines at: https://findahelpline.com

---

##  References

1. Oyebode, F. (2022). *Sims' Symptoms in the Mind: Textbook of Descriptive Psychopathology* (7th ed.). Elsevier.
2. [LangChain Documentation](https://python.langchain.com/)
3. [Ollama Documentation](https://ollama.ai/docs)
4. [ChromaDB Documentation](https://docs.trychroma.com/)

---

<div align="center">

**Built with ❤️ for mental health awareness and education**

⭐ **Star this repo if you find it helpful!** ⭐

</div>
