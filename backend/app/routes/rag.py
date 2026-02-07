"""
RAG Routes - Vector databases and retrieval-augmented generation
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import uuid

from app.services.llama_stack import get_llama_stack_service
from llama_stack_client import AgentEventLogger


router = APIRouter(tags=["rag"])

# In-memory storage for vector DBs and RAG agents
_vector_dbs: Dict[str, Dict[str, Any]] = {}
_rag_agents: Dict[str, Any] = {}


class CreateVectorDBRequest(BaseModel):
    """Request to create a new vector database."""
    vector_db_id: Optional[str] = Field(None, description="Optional vector DB identifier")
    embedding_model: Optional[str] = Field(None, description="Embedding model to use")


class InsertDocumentsRequest(BaseModel):
    """Request to insert documents into a vector database."""
    vector_db_id: str = Field(..., description="Vector database identifier")
    documents: List[Dict[str, Any]] = Field(..., description="Documents to insert")
    chunk_size_in_tokens: int = Field(512, description="Chunk size for document splitting")


class DocumentInput(BaseModel):
    """Document input model."""
    content: str = Field(..., description="Document content (text or URL)")
    document_id: Optional[str] = Field(None, description="Optional document identifier")
    mime_type: str = Field("text/plain", description="MIME type of the content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class InsertDocumentsSimpleRequest(BaseModel):
    """Simplified request to insert documents."""
    vector_db_id: str = Field(..., description="Vector database identifier")
    contents: List[str] = Field(..., description="List of text contents to insert")
    chunk_size_in_tokens: int = Field(512, description="Chunk size")


class CreateRAGAgentRequest(BaseModel):
    """Request to create a RAG-enabled agent."""
    vector_db_ids: List[str] = Field(..., description="Vector database IDs to search")
    instructions: Optional[str] = Field(None, description="Agent instructions")
    model: Optional[str] = Field(None, description="Model identifier")


class RAGQueryRequest(BaseModel):
    """Request to query a RAG agent."""
    rag_agent_id: str = Field(..., description="RAG agent identifier")
    message: str = Field(..., description="User query")
    session_id: Optional[str] = Field(None, description="Optional session ID for multi-turn")


class QuickRAGRequest(BaseModel):
    """Quick RAG request for one-shot queries."""
    vector_db_id: str = Field(..., description="Vector database to search")
    message: str = Field(..., description="User query")
    model: Optional[str] = Field(None, description="Model identifier")


@router.post("/rag/vector-dbs")
async def create_vector_db(request: CreateVectorDBRequest) -> Dict[str, Any]:
    """
    Create and register a new vector database for RAG.
    
    Vector databases store document embeddings for retrieval.
    """
    try:
        service = get_llama_stack_service()
        
        vector_db_id = service.register_vector_db(
            vector_db_id=request.vector_db_id,
            embedding_model=request.embedding_model,
        )
        
        _vector_dbs[vector_db_id] = {
            "embedding_model": request.embedding_model or service.get_default_embedding_model(),
            "document_count": 0,
        }
        
        return {
            "vector_db_id": vector_db_id,
            "embedding_model": _vector_dbs[vector_db_id]["embedding_model"],
            "status": "created",
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/vector-dbs")
async def list_vector_dbs() -> List[Dict[str, Any]]:
    """List all registered vector databases."""
    return [
        {
            "vector_db_id": db_id,
            **data,
        }
        for db_id, data in _vector_dbs.items()
    ]


@router.post("/rag/documents")
async def insert_documents(request: InsertDocumentsRequest) -> Dict[str, Any]:
    """
    Insert documents into a vector database.
    
    Documents can be text content or URLs to be fetched.
    """
    try:
        service = get_llama_stack_service()
        
        result = service.insert_documents(
            vector_db_id=request.vector_db_id,
            documents=request.documents,
            chunk_size_in_tokens=request.chunk_size_in_tokens,
        )
        
        # Update document count
        if request.vector_db_id in _vector_dbs:
            _vector_dbs[request.vector_db_id]["document_count"] += len(request.documents)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/documents/simple")
async def insert_documents_simple(request: InsertDocumentsSimpleRequest) -> Dict[str, Any]:
    """
    Simplified endpoint to insert text documents.
    
    Just provide a list of text strings to be indexed.
    """
    try:
        service = get_llama_stack_service()
        
        documents = [
            {
                "document_id": f"doc-{uuid.uuid4().hex[:8]}",
                "content": content,
                "mime_type": "text/plain",
                "metadata": {},
            }
            for content in request.contents
        ]
        
        result = service.insert_documents(
            vector_db_id=request.vector_db_id,
            documents=documents,
            chunk_size_in_tokens=request.chunk_size_in_tokens,
        )
        
        if request.vector_db_id in _vector_dbs:
            _vector_dbs[request.vector_db_id]["document_count"] += len(documents)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/agents")
async def create_rag_agent(request: CreateRAGAgentRequest) -> Dict[str, Any]:
    """
    Create a RAG-enabled agent.
    
    RAG agents can search vector databases to answer questions.
    """
    try:
        service = get_llama_stack_service()
        
        agent = service.create_rag_agent(
            vector_db_ids=request.vector_db_ids,
            instructions=request.instructions,
            model=request.model,
        )
        
        rag_agent_id = f"rag-agent-{uuid.uuid4().hex[:8]}"
        session_id = service.create_agent_session(agent)
        
        _rag_agents[rag_agent_id] = {
            "agent": agent,
            "sessions": {session_id: []},
            "default_session": session_id,
            "vector_db_ids": request.vector_db_ids,
        }
        
        return {
            "rag_agent_id": rag_agent_id,
            "session_id": session_id,
            "vector_db_ids": request.vector_db_ids,
            "model": request.model or service.get_default_llm(),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/agents")
async def list_rag_agents() -> List[Dict[str, Any]]:
    """List all RAG agents."""
    return [
        {
            "rag_agent_id": agent_id,
            "vector_db_ids": data.get("vector_db_ids", []),
            "session_count": len(data.get("sessions", {})),
        }
        for agent_id, data in _rag_agents.items()
    ]


@router.post("/rag/query")
async def query_rag_agent(request: RAGQueryRequest) -> Dict[str, Any]:
    """
    Query a RAG agent.
    
    The agent will search the vector database and generate a response.
    """
    try:
        if request.rag_agent_id not in _rag_agents:
            raise HTTPException(status_code=404, detail=f"RAG agent {request.rag_agent_id} not found")
        
        rag_data = _rag_agents[request.rag_agent_id]
        agent = rag_data["agent"]
        service = get_llama_stack_service()
        
        # Use provided session or default
        session_id = request.session_id or rag_data["default_session"]
        
        result = service.agent_turn(
            agent=agent,
            session_id=session_id,
            message=request.message,
            stream=False,
        )
        
        return {
            "response": result.get("output_message", {}).get("content", ""),
            "turn_id": result.get("turn_id"),
            "session_id": session_id,
            "steps": result.get("steps", []),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/quick")
async def quick_rag_query(request: QuickRAGRequest) -> Dict[str, Any]:
    """
    Quick one-shot RAG query.
    
    Creates a temporary RAG agent, queries it, and returns the result.
    """
    try:
        service = get_llama_stack_service()
        
        agent = service.create_rag_agent(
            vector_db_ids=[request.vector_db_id],
            model=request.model,
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
