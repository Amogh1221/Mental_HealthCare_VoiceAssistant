# 🎙️ AI Mental Health Voice Assistant

A voice-only AI mental health interview system built with:

-   Streamlit (UI)
-   Ollama (LLM backend)
-   ChromaDB (RAG vector store)
-   Faster-Whisper (Speech-to-Text)
-   Browser SpeechSynthesis (Text-to-Speech)

The assistant conducts a calm, natural, spoken interview to understand
emotional experiences. It does NOT diagnose or give advice. It focuses
purely on structured conversational exploration.

------------------------------------------------------------------------

## 🧠 Architecture Overview

### 1. RAG Pipeline (`vectordb.py`)

-   Loads text data from `RAG_data.txt`
-   Splits entries using a delimiter
-   Embeds using `sentence-transformers/all-MiniLM-L6-v2`
-   Stores embeddings in ChromaDB (`MH_db`)

### 2. Interview LLM (LLM1)

-   Model: `llama3.2:3b` (via Ollama)
-   Structured JSON output
-   Controls conversation flow
-   Produces:
    -   Spoken assistant response
    -   Intent (`CONTINUE` or `ANALYZE`)

### 3. Internal Reasoner (LLM2)

-   Model: `nemotron-mini`
-   Extracts:
    -   Behavioral patterns
    -   Contextual considerations
-   Never speaks to the user

### 4. Voice Pipeline

-   Audio input via `audio_recorder_streamlit`
-   Speech-to-text via Faster-Whisper
-   Text-to-speech via browser SpeechSynthesis API

------------------------------------------------------------------------

## 📁 Project Structure

. ├── app.py
  ├── vectordb.py
  ├── RAG_data.txt
  ├── MH_db/
  ├── README.md
  └── requirements.txt

------------------------------------------------------------------------

## ⚙️ Setup Instructions

### 1. Install Ollama

Download from: https://ollama.com

Pull required models:

``` bash
ollama pull llama3.2:3b
ollama pull nemotron-mini
```

### 2. Create Virtual Environment

``` bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate    # Windows
```

### 3. Install Dependencies

``` bash
pip install -r requirements.txt
```

### 4. Build Vector Database

Ensure `RAG_data.txt` exists.

``` bash
python vectordb.py
```

This creates:

MH_db/

### 5. Run Application

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 🔊 How It Works

1.  Click **Start Conversation**
2.  Speak naturally
3.  Audio is transcribed
4.  LLM generates structured response
5.  Browser speaks reply
6.  Conversation continues until stopped

No text interaction is required.

------------------------------------------------------------------------

## 🛑 Design Constraints

The assistant:

-   Does NOT diagnose
-   Does NOT give advice
-   Does NOT use clinical labels
-   Does NOT judge or validate strongly
-   Asks at most one question per turn

------------------------------------------------------------------------

## 🚧 Limitations

-   Runs on CPU by default (Whisper int8 mode)
-   Requires Ollama running locally
-   Not a replacement for licensed mental health care
-   Browser TTS quality varies by system

------------------------------------------------------------------------

## ⚠️ Disclaimer

This project is an experimental AI conversational system. It is not
medical software and should not be used as a substitute for professional
care.

------------------------------------------------------------------------
