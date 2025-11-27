from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agents.orchestrator import handle_message

SessionState = Dict[str, Any]
SESSIONS: Dict[str, SessionState] = {}

app = FastAPI(title="Nephrology Assistant API") # instance of fastApi


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # for assignment/demo; can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel): #pydantic base model
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    agent: str   # "receptionist" or "clinical"


@app.get("/") #get request
async def health_check():
    return {"status": "ok", "message": "Nephrology assistant backend is running"}


@app.post("/chat", response_model=ChatResponse) #post request
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint.
    The frontend must keep using the same session_id for one conversation.
    """
    state: SessionState = SESSIONS.get(payload.session_id, {})

    reply, new_state = handle_message(payload.message, state) # calling handle_message function in orchestrator

    # detect which agent responded 
    agent_name = new_state.get("mode", "receptionist")

    SESSIONS[payload.session_id] = new_state

    return ChatResponse(
        session_id=payload.session_id,
        reply=reply,
        agent=agent_name,
    )
