import os
import pathlib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.config import THREAD_ID, DEFAULT_PDF
from utils.loader import build_retriever
import utils.tools
from utils.graph_builder import build_graph

pdf_path = pathlib.Path(DEFAULT_PDF)
if not pdf_path.exists():
    raise RuntimeError(f"PDF not found at {pdf_path}")

utils.tools.RETRIEVER = build_retriever(str(pdf_path))
graph = build_graph()
cfg   = {"configurable": {"thread_id": THREAD_ID}}

app = FastAPI(title="NVIDIA LangGraph Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       
    allow_credentials=True,
    allow_methods=["*"],      
    allow_headers=["*"],       
)

class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    state = {"messages": [{"role": "user", "content": req.text}]}
    try:
        result = graph.invoke(state, config=cfg)
    except Exception as e:
        raise HTTPException(500, detail=f"Agent error: {e}")

    reply = None
    for msg in reversed(result.get("messages", [])):
        if msg.type == "ai" and not getattr(msg, "tool_calls", None):
            reply = msg.content
            break

    if reply is None:
        raise HTTPException(500, detail="No valid assistant reply generated")

    return ChatResponse(reply=reply)
