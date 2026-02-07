"""
Agents Routes - Create and manage Llama Stack agents with tool support
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uuid

from app.services.llama_stack import get_llama_stack_service, LlamaStackService
from llama_stack_client import Agent, AgentEventLogger


router = APIRouter(tags=["agents"])

# In-memory storage for agents and sessions (in production, use a database)
_agents: Dict[str, Agent] = {}
_sessions: Dict[str, Dict[str, Any]] = {}


class CreateAgentRequest(BaseModel):
    """Request to create a new agent."""
    instructions: Optional[str] = Field(None, description="System instructions for the agent")
    model: Optional[str] = Field(None, description="Model identifier")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Tools to enable")


class CreateSessionRequest(BaseModel):
    """Request to create a new session for an agent."""
    agent_id: str = Field(..., description="Agent identifier")
    session_name: Optional[str] = Field(None, description="Optional session name")


class AgentTurnRequest(BaseModel):
    """Request to create a turn in an agent session."""
    agent_id: str = Field(..., description="Agent identifier")
    session_id: str = Field(..., description="Session identifier")
    message: str = Field(..., description="User message")
    stream: bool = Field(False, description="Whether to stream the response")


class QuickAgentRequest(BaseModel):
    """Quick agent request for one-shot agent interactions."""
    message: str = Field(..., description="User message")
    instructions: Optional[str] = Field(None, description="Agent instructions")
    model: Optional[str] = Field(None, description="Model identifier")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Tools to enable")


@router.post("/agents")
async def create_agent(request: CreateAgentRequest) -> Dict[str, Any]:
    """
    Create a new Llama Stack agent.
    
    Agents combine LLM reasoning with tools for taking actions.
    """
    try:
        service = get_llama_stack_service()
        
        agent = service.create_agent(
            instructions=request.instructions,
            model=request.model,
            tools=request.tools,
        )
        
        agent_id = f"agent-{uuid.uuid4().hex[:8]}"
        _agents[agent_id] = agent
        
        return {
            "agent_id": agent_id,
            "model": request.model or service.get_default_llm(),
            "instructions": request.instructions,
            "tools": request.tools or [],
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents")
async def list_agents() -> List[Dict[str, Any]]:
    """List all created agents."""
    return [
        {"agent_id": agent_id}
        for agent_id in _agents.keys()
    ]


@router.post("/agents/sessions")
async def create_session(request: CreateSessionRequest) -> Dict[str, Any]:
    """
    Create a new session for an agent.
    
    Sessions maintain conversation state across multiple turns.
    """
    try:
        if request.agent_id not in _agents:
            raise HTTPException(status_code=404, detail=f"Agent {request.agent_id} not found")
        
        agent = _agents[request.agent_id]
        service = get_llama_stack_service()
        
        session_id = service.create_agent_session(agent, request.session_name)
        
        _sessions[session_id] = {
            "agent_id": request.agent_id,
            "session_name": request.session_name,
            "turns": [],
        }
        
        return {
            "session_id": session_id,
            "agent_id": request.agent_id,
            "session_name": request.session_name,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/sessions")
async def list_sessions() -> List[Dict[str, Any]]:
    """List all sessions."""
    return [
        {
            "session_id": session_id,
            "agent_id": data.get("agent_id"),
            "session_name": data.get("session_name"),
            "turn_count": len(data.get("turns", [])),
        }
        for session_id, data in _sessions.items()
    ]


@router.post("/agents/turns")
async def create_turn(request: AgentTurnRequest) -> Dict[str, Any]:
    """
    Create a turn in an agent session.
    
    A turn represents one user message and the agent's response,
    including any tool calls the agent makes.
    """
    try:
        if request.agent_id not in _agents:
            raise HTTPException(status_code=404, detail=f"Agent {request.agent_id} not found")
        
        agent = _agents[request.agent_id]
        service = get_llama_stack_service()
        
        if request.stream:
            def generate():
                stream = agent.create_turn(
                    messages=[{"role": "user", "content": request.message}],
                    session_id=request.session_id,
                    stream=True,
                )
                for event in AgentEventLogger().log(stream):
                    if hasattr(event, 'text'):
                        yield f"data: {event.text}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        
        result = service.agent_turn(
            agent=agent,
            session_id=request.session_id,
            message=request.message,
            stream=False,
        )
        
        # Track the turn
        if request.session_id in _sessions:
            _sessions[request.session_id]["turns"].append({
                "turn_id": result.get("turn_id"),
                "user_message": request.message,
                "agent_response": result.get("output_message", {}).get("content"),
            })
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/quick")
async def quick_agent(request: QuickAgentRequest) -> Dict[str, Any]:
    """
    Quick one-shot agent interaction.
    
    Creates a temporary agent, runs a single turn, and returns the result.
    Useful for simple agent tasks without managing sessions.
    """
    try:
        service = get_llama_stack_service()
        
        agent = service.create_agent(
            instructions=request.instructions,
            model=request.model,
            tools=request.tools,
        )
        
        session_id = service.create_agent_session(agent)
        
        result = service.agent_turn(
            agent=agent,
            session_id=session_id,
            message=request.message,
            stream=False,
        )
        
        return {
            "response": result.get("output_message", {}).get("content", ""),
            "turn_id": result.get("turn_id"),
            "steps": result.get("steps", []),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
