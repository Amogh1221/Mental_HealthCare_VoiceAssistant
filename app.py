import streamlit as st
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma

from faster_whisper import WhisperModel
from audio_recorder_streamlit import audio_recorder
import tempfile
import streamlit.components.v1 as components

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI Mental Health Interview (Voice Only)",
    layout="centered"
)

# -------------------- VECTOR STORE --------------------
vector_store = Chroma(
    collection_name="Docs",
    persist_directory="MH_db",
    embedding_function=None
)

def retrieve_rag(query: str, k: int = 4) -> str:
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n".join(doc.page_content for doc in docs)

# -------------------- OUTPUT SCHEMAS --------------------
class LLM1Output(BaseModel):
    assistant_message: str
    intent: Literal["CONTINUE", "ANALYZE"]

class LLM2Output(BaseModel):
    patterns: List[str] = Field(default_factory=list)
    considerations: List[str] = Field(default_factory=list)

# -------------------- PROMPTS --------------------
LLM1_SYSTEM_PROMPT = """
You are having a natural, calm, one-on-one spoken conversation.

You are interviewing the user to understand their mental and emotional experiences.
Everything you say will be spoken aloud.
The user will not see any text.

You MUST respond ONLY with valid JSON in this exact format:

{
  "assistant_message": "natural spoken response",
  "intent": "CONTINUE | ANALYZE"
}

CONVERSATION STYLE RULES:
- Speak like a real person, not a questionnaire.
- Use simple, clear, conversational language.
- Keep sentences short and easy to follow.
- Sound attentive, calm, and neutral.
- Do not rush the conversation.
- Avoid sounding clinical, formal, or robotic.

QUESTION RULES:
- Ask AT MOST one question per response.
- You may sometimes respond without a question if it feels natural.
- Do not repeat the user’s words back to them.
- Do not summarize unless it feels conversationally appropriate.

CONTENT RULES:
- Do NOT diagnose, label, or name conditions.
- Do NOT give advice or solutions.
- Do NOT judge or validate strongly (“that’s good” / “that’s bad”).
- Avoid medical or technical terms.
- Focus on understanding experiences, feelings, and context.

INTENT RULES:
- intent = CONTINUE → when you still need more understanding.
- intent = ANALYZE → only when the conversation feels complete and no further questions are needed.

STRICT OUTPUT RULES:
- Output JSON ONLY.
- No markdown.
- No explanations.
- No extra keys.

"""

LLM2_SYSTEM_PROMPT = """
You are an internal reasoning engine.
You do NOT speak to the user.

Your task:
- Identify patterns in the conversation.
- Note alternative explanations or contextual factors.

Rules:
- Do NOT diagnose or name disorders.
- Do NOT give advice or recommendations.
- Output ONLY valid JSON.
"""

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_llms():
    llm_interviewer = ChatOllama(
        model="llama3.2:3b",
        temperature=0.4,
        system_prompt=LLM1_SYSTEM_PROMPT,
        format="json"
    )

    llm_reasoner = ChatOllama(
        model="nemotron-mini",
        temperature=0.2,
        system_prompt=LLM2_SYSTEM_PROMPT,
        format="json"
    )

    return (
        llm_interviewer.with_structured_output(LLM1Output),
        llm_reasoner.with_structured_output(LLM2Output)
    )

@st.cache_resource
def load_whisper():
    return WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )

llm1, llm2 = load_llms()
whisper_model = load_whisper()

# -------------------- SESSION STATE --------------------
if "conversation_active" not in st.session_state:
    st.session_state.conversation_active = False

if "interview_history" not in st.session_state:
    st.session_state.interview_history = []

if "last_audio_bytes" not in st.session_state:
    st.session_state.last_audio_bytes = None

# -------------------- HELPERS --------------------
def transcribe_audio(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    segments, _ = whisper_model.transcribe(audio_path)
    return " ".join(seg.text for seg in segments).strip()

def browser_speak(text: str):
    components.html(
        f"""
        <script>
        if (window.speechSynthesis) {{
            speechSynthesis.cancel();
            const msg = new SpeechSynthesisUtterance({text!r});
            msg.rate = 1.0;
            msg.pitch = 1.0;
            msg.volume = 1.0;

            const voices = speechSynthesis.getVoices();
            const preferred = voices.find(v =>
                v.name.includes("Google") || v.name.includes("Microsoft")
            );
            if (preferred) msg.voice = preferred;

            speechSynthesis.speak(msg);
        }}
        </script>
        """,
        height=0
    )

def run_llm1(user_input: str) -> LLM1Output:
    st.session_state.interview_history.append({
        "role": "user",
        "content": user_input
    })

    response: LLM1Output = llm1.invoke(st.session_state.interview_history)

    st.session_state.interview_history.append({
        "role": "assistant",
        "content": response.assistant_message
    })

    return response

# -------------------- UI --------------------
st.title("🎙️ AI Mental Health ")

col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Start Conversation"):
        st.session_state.conversation_active = True
        st.session_state.last_audio_bytes = None

with col2:
    if st.button("⏹ End Conversation"):
        st.session_state.conversation_active = False
        components.html("<script>speechSynthesis.cancel();</script>", height=0)

# -------------------- VOICE PIPELINE --------------------
if st.session_state.conversation_active:
    audio_bytes = audio_recorder(
        pause_threshold=2.0,
        sample_rate=16000
    )

    if audio_bytes and audio_bytes != st.session_state.last_audio_bytes:
        st.session_state.last_audio_bytes = audio_bytes

        with st.spinner("Listening..."):
            user_text = transcribe_audio(audio_bytes)

        if user_text:
            llm_response = run_llm1(user_text)
            browser_speak(llm_response.assistant_message)

else:
    st.info("Click **Start Conversation** and speak.")
