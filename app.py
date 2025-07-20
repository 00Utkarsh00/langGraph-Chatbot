import os
import pathlib
from uuid import uuid4
from io import BytesIO
from xmlrpc import client as xmlrpc_client

from fastapi import FastAPI, HTTPException, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from openai import OpenAI
from dotenv import load_dotenv

from utils.config import THREAD_ID, DEFAULT_PDF
from utils.loader import build_retriever
import utils.tools
from utils.graph_builder import build_graph

# Load environment
load_dotenv()
client = OpenAI()

# Ensure PDF is present
pdf_path = pathlib.Path(DEFAULT_PDF)
if not pdf_path.exists():
    raise RuntimeError(f"PDF not found at {pdf_path}")

# Build retriever & graph
utils.tools.RETRIEVER = build_retriever(str(pdf_path))
graph = build_graph()
cfg   = {"configurable": {"thread_id": THREAD_ID}}

# --- FastAPI setup ---
app = FastAPI(title="NVIDIA LangGraph Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Models ---
class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    reply: str


class AudioRequest(BaseModel):
    text: str

class AudioResponse(BaseModel):
    reply: str
    audio_url: str


# text endpoint
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    state = {"messages": [{"role": "user", "content": req.text}]}
    try:
        result = graph.invoke(state, config=cfg)
    except Exception as e:
        raise HTTPException(500, detail=f"Agent error: {e}")

    # Find the last assistant message that isn’t a tool call
    reply = None
    for msg in reversed(result.get("messages", [])):
        if msg.type == "ai" and not getattr(msg, "tool_calls", None):
            reply = msg.content
            break

    if reply is None:
        raise HTTPException(500, detail="No valid assistant reply generated")

    return ChatResponse(reply=reply)


# voice endpoint
@app.post("/audio-chat", response_model=AudioResponse)
async def audio_chat(input: AudioRequest):
    text = input.text.strip()
    if not text:
        raise HTTPException(400, detail="Text input is required.")

    # 1) Invoke your LLM agent
    state = {"messages": [{"role": "user", "content": text}]}
    result = graph.invoke(state, config=cfg)

    # 2) Extract the AI reply
    reply = None
    for msg in reversed(result.get("messages", [])):
        if msg.type == "ai" and not getattr(msg, "tool_calls", None):
            reply = msg.content
            break

    if reply is None:
        raise HTTPException(500, detail="No valid assistant reply generated")

    # 3) Ensure audio output directory exists
    audio_dir = pathlib.Path(__file__).parent / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # 4) Choose a unique filename
    filename = f"{uuid4().hex}.mp3"
    speech_file_path = audio_dir / filename

    # 5) Generate TTS and stream to file
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=reply,
        instructions="Speak in a cheerful and positive tone.",
    ) as response:
        response.stream_to_file(speech_file_path)

    # 6) Return the URL for the frontend
    return AudioResponse(
        reply=reply,
        audio_url=f"/audio/{filename}"
    )


# --- Static file serving for TTS outputs ---
@app.get("/audio/{file_name}")
def get_audio(file_name: str):
    audio_path = pathlib.Path(__file__).parent / "audio" / file_name
    if not audio_path.exists() or not audio_path.is_file():
        raise HTTPException(404, detail="Audio file not found")
    return FileResponse(audio_path, media_type="audio/mpeg")
