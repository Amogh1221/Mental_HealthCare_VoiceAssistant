from fastapi import FastAPI, Request, UploadFile, File, Response
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import uuid
import json
from typing import Dict, List
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from rag_engine import rag_engine
from llm_engine import llm_engine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

sessions: Dict[str, List[Dict]] = {}
MAX_HISTORY = 100


# Pydantic models for request validation
class ResetRequest(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    message: str
    session_id: str


def trim_history(history):
    if len(history) > MAX_HISTORY:
        return history[-MAX_HISTORY:]
    return history


def format_list(items):
    """Helper to format list items for display"""
    if not items:
        return "None identified yet"
    return "\n  • " + "\n  • ".join(items)


def create_new_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = []

    opening_context = [
        {
            "role": "user",
            "content": "Start the psychiatric session and greet the patient naturally."
        }
    ]

    llm1_response = llm_engine.psychiatrist_response(opening_context)

    sessions[session_id].append({
        "role": "assistant",
        "content": llm1_response.assistant_message
    })

    return session_id, llm1_response.assistant_message


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})


@app.post("/start")
def start_session():
    session_id, opening_message = create_new_session()

    return {
        "assistant_message": opening_message,
        "session_id": session_id
    }


@app.post("/reset")
def reset_session(request: ResetRequest):
    # Delete old session if it exists
    if request.session_id in sessions:
        del sessions[request.session_id]

    # Create new session
    session_id, opening_message = create_new_session()

    return {
        "assistant_message": opening_message,
        "session_id": session_id
    }


@app.post("/chat_text")
def chat_text(request: ChatRequest):
    
    history = sessions.get(request.session_id)

    if history is None:
        session_id, opening_message = create_new_session()
        history = sessions[session_id]
        request.session_id = session_id

    # Add user message to history
    history.append({
        "role": "user",
        "content": request.message
    })

    history = trim_history(history)
    sessions[request.session_id] = history

    # Get Dr. Aiden's conversational response
    llm1_response = llm_engine.psychiatrist_response(history)

    # CONTINUE path (Simple chat)
    if llm1_response.intent == "CONTINUE":
        history.append({
            "role": "assistant",
            "content": llm1_response.assistant_message
        })
        history = trim_history(history)
        sessions[request.session_id] = history

        return {
            "assistant_message": llm1_response.assistant_message,
            "intent": llm1_response.intent
        }

    # ANALYZE path (Trigger Reasoning Specialist)
    if llm1_response.intent == "ANALYZE":
        # 1. Retrieve clinical context from Pinecone cloud
        recent_text = "\n".join(
            [msg["content"] for msg in history[-10:]]
        )
        retrieved_context = rag_engine.retrieve(recent_text)

        # 2. Call LLM2 (Analyst) with history and clinical context
        # We wrap the context in a user message so it reaches the analyst's brain
        llm2_input = history[-10:] + [{"role": "user", "content": f"Clinical Context for Analysis:\n{retrieved_context}\n\nPlease perform pattern analysis."}]
        llm2_response = llm_engine.internal_reasoning(llm2_input)

        # 3. Format the result for the internal psychiatrist briefing
        analysis_briefing = f"""[Internal Clinical Analysis - For Treatment Planning]

Emotional Themes:
{format_list(llm2_response.emotional_themes)}

Thinking Patterns:
{format_list(llm2_response.thinking_patterns)}

Behavioral Patterns:
{format_list(llm2_response.behavioral_patterns)}

Interpersonal Dynamics:
{format_list(llm2_response.interpersonal_dynamics)}

Identified Stressors:
{format_list(llm2_response.stressors)}

Areas Requiring Further Exploration:
{format_list(llm2_response.unclear_areas)}

Based on this clinical insight, provide your next therapeutic response to the patient."""

        # 4. Get final Dr. Aiden response based on the briefing
        briefing_history = history + [{"role": "user", "content": analysis_briefing}]
        llm1_final_response = llm_engine.psychiatrist_response(briefing_history)

        # 5. Save the conversational outcome to history
        history.append({
            "role": "assistant",
            "content": llm1_final_response.assistant_message
        })

        history = trim_history(history)
        sessions[request.session_id] = history

        return {
            "assistant_message": llm1_final_response.assistant_message,
            "intent": "CONTINUE"  # Always return to chat mode
        }

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Transcribe uploaded audio file to text using Whisper."""
    audio_bytes = await audio.read()
    text = llm_engine.transcribe_audio(audio_bytes)
    return {"text": text if text else ""}

@app.get("/speech")
async def speech(text: str):
    """Generate professional speech from text using Parler-TTS."""
    audio_content = llm_engine.generate_speech(text)
    if audio_content:
        return Response(content=audio_content, media_type="audio/wav")
    return Response(content="Speech generation failed", status_code=500)