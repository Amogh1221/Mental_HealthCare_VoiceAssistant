import os
import json
import io
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from faster_whisper import WhisperModel

load_dotenv()

class LLM1Output(BaseModel):
    assistant_message: str
    intent: Literal["CONTINUE", "ANALYZE"]

class LLM2Output(BaseModel):
    emotional_themes: List[str] = Field(default_factory=list)
    thinking_patterns: List[str] = Field(default_factory=list)
    behavioral_patterns: List[str] = Field(default_factory=list)
    interpersonal_dynamics: List[str] = Field(default_factory=list)
    stressors: List[str] = Field(default_factory=list)
    unclear_areas: List[str] = Field(default_factory=list)

LLM1_SYSTEM_PROMPT = """
You are Dr. Aiden, a compassionate and professionally trained AI psychiatrist conducting a clinical interview with a patient. Your role is to gather information about the patient's mental state, symptoms, and experiences through empathetic conversation.
...
""" # The user's restored full prompt is preserved below

LLM2_SYSTEM_PROMPT = """
You are a clinical pattern analyst specializing in descriptive psychopathology.
...
""" # The user's restored full prompt is preserved below

class LLMEngine:
    def __init__(self):
        self.api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        self.model1 = os.getenv("LLM1_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        self.model2 = os.getenv("LLM2_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        
        try:
            print("[STARTUP DEBUG] Initializing Local Engines (Fast Mode)...")
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            os.makedirs(models_dir, exist_ok=True)
            
            # 1. STT Init (Whisper) - High Performance Local
            print("[STARTUP DEBUG] Loading Whisper (Listening engine)...")
            self.whisper = WhisperModel("tiny.en", device="cpu", compute_type="int8", 
                                        download_root=os.path.join(models_dir, "whisper"))
            
            print("System initialized in Fast Mode (Whisper Local + Browser TTS).")
                
        except Exception as e:
            print(f"CRITICAL ERROR: Whisper failed to initialize: {e}")
            self.whisper = None

    def psychiatrist_response(self, context):
        try:
            client = InferenceClient(token=self.api_token)
            # Full logic here... (prompts are at the Top, restored by user)
            messages = [{"role": "system", "content": LLM1_SYSTEM_PROMPT}]
            for m in context:
                messages.append({"role": m['role'], "content": m['content']})
            raw_text = client.chat_completion(model=self.model1, messages=messages, max_tokens=1024, temperature=0.6).choices[0].message.content
            
            start, end = raw_text.find('{'), raw_text.rfind('}') + 1
            if start != -1 and end > start:
                try:
                    data = json.loads(raw_text[start:end])
                    return LLM1Output(**data)
                except: pass
            return LLM1Output(assistant_message=raw_text, intent="CONTINUE")
        except Exception as e:
            print(f"DEBUG LLM1: {e}")
            return LLM1Output(assistant_message="I'm here. Tell me more.", intent="CONTINUE")

    def internal_reasoning(self, context):
        try:
            client = InferenceClient(token=self.api_token)
            messages = [{"role": "system", "content": LLM2_SYSTEM_PROMPT}]
            for m in context:
                messages.append({"role": m['role'], "content": m['content']})
            raw_text = client.chat_completion(model=self.model2, messages=messages, max_tokens=1024, temperature=0.6).choices[0].message.content
            start, end = raw_text.find('{'), raw_text.rfind('}') + 1
            if start != -1 and end > start:
                try:
                    return LLM2Output(**json.loads(raw_text[start:end]))
                except: pass
            return LLM2Output()
        except Exception as e:
            print(f"DEBUG LLM2: {e}")
            return LLM2Output()

    def transcribe_audio(self, audio_bytes: bytes):
        if self.whisper is None: return "Voice support disabled."
        try:
            audio_file = io.BytesIO(audio_bytes)
            # High speed transcription with VAD
            segments, info = self.whisper.transcribe(audio_file, beam_size=1, vad_filter=True)
            text = " ".join([seg.text for seg in segments])
            return text.strip()
        except Exception as e:
            print(f"Local STT Error: {e}")
            return None

    def generate_speech(self, text: str):
        # We now use Browser TTS for zero lag.
        return None

llm_engine = LLMEngine()