from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import uuid
from typing import Dict, List

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
def reset_session(session_id: str = Body(...)):

    if session_id in sessions:
        del sessions[session_id]

    session_id, opening_message = create_new_session()

    return {
        "assistant_message": opening_message,
        "session_id": session_id
    }


@app.post("/chat_text")
def chat_text(
    message: str = Body(...),
    session_id: str = Body(...)
):

    history = sessions.get(session_id)

    if history is None:
        session_id, opening_message = create_new_session()
        history = sessions[session_id]

    history.append({
        "role": "user",
        "content": message
    })

    history = trim_history(history)
    sessions[session_id] = history

    llm1_response = llm_engine.psychiatrist_response(history)

    if llm1_response.intent == "CONTINUE":
        history.append({
            "role": "assistant",
            "content": llm1_response.assistant_message
        })
        history = trim_history(history)
        sessions[session_id] = history

        return {
            "assistant_message": llm1_response.assistant_message,
            "intent": llm1_response.intent
        }

    if llm1_response.intent == "ANALYZE":
        recent_text = "\n".join(
            [msg["content"] for msg in history[-10:]]
        )
        retrieved_context = rag_engine.retrieve(recent_text)

        llm2_context = f"""Analyze the following conversation and provide pattern analysis.

[Recent Conversation History]
{chr(10).join([f"{msg['role'].upper()}: {msg['content']}" for msg in history[-10:]])}

[Retrieved Clinical Context from Knowledge Base]
{retrieved_context}

Based on the conversation and clinical context above, identify patterns across all six domains."""

        llm2_input = [{"role": "user", "content": llm2_context}]
        llm2_response = llm_engine.internal_reasoning(llm2_input)

        analysis_content = f"""[Internal Clinical Analysis - For Treatment Planning]

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

Based on this analysis, provide your next therapeutic response to the patient."""

        enriched_history = history + [
            {
                "role": "user",
                "content": analysis_content
            }
        ]

        llm1_final_response = llm_engine.psychiatrist_response(enriched_history)

        history.append({
            "role": "assistant",
            "content": llm1_final_response.assistant_message
        })

        history = trim_history(history)
        sessions[session_id] = history

        return {
            "assistant_message": llm1_final_response.assistant_message,
            "intent": "CONTINUE"
        }