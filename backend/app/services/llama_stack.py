"""
Llama Stack Service - Unified wrapper for all Llama Stack APIs

This module provides a comprehensive wrapper around the llama-stack-client SDK,
exposing all built-in Llama Stack APIs:
- Inference (chat completions)
- Models (list, get)
- Agents (create, sessions, turns)
- Vector DBs & RAG (register, insert documents, query)
- Embeddings
- Safety (shields)
"""

from typing import Optional, List, Dict, Any, Generator, AsyncGenerator
import uuid

from llama_stack_client import LlamaStackClient as BaseLlamaStackClient
from llama_stack_client import Agent, AgentEventLogger

from app.core.config import settings


class LlamaStackService:
    """
    Unified service for all Llama Stack APIs.
    Provides methods for inference, agents, RAG, embeddings, and safety.
    """
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.LLAMA_STACK_BASE_URL
        self.client = BaseLlamaStackClient(base_url=self.base_url)
        self._default_model: Optional[str] = None
        self._default_embedding_model: Optional[str] = None
    
    # ==================== Models API ====================
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models from the Llama Stack server."""
        models = self.client.models.list()
        return [
            {
                "id": m.id if hasattr(m, 'id') else str(m),
                "identifier": m.identifier if hasattr(m, 'identifier') else m.id if hasattr(m, 'id') else str(m),
                "model_type": m.custom_metadata.get("model_type") if hasattr(m, 'custom_metadata') and m.custom_metadata else None,
                "provider_id": m.custom_metadata.get("provider_id") if hasattr(m, 'custom_metadata') and m.custom_metadata else None,
                "metadata": m.custom_metadata if hasattr(m, 'custom_metadata') else {},
            }
            for m in models
        ]
    
    def get_default_llm(self) -> str:
        """Get the default LLM model identifier."""
        if self._default_model:
            return self._default_model
        
        models = self.client.models.list()
        for m in models:
            if hasattr(m, 'custom_metadata') and m.custom_metadata:
                if m.custom_metadata.get("model_type") == "llm":
                    self._default_model = m.id
                    return self._default_model
        
        # Fallback to configured default
        self._default_model = settings.DEFAULT_MODEL
        return self._default_model
    
    def get_default_embedding_model(self) -> str:
        """Get the default embedding model identifier."""
        if self._default_embedding_model:
            return self._default_embedding_model
        
        models = self.client.models.list()
        for m in models:
            if hasattr(m, 'custom_metadata') and m.custom_metadata:
                if m.custom_metadata.get("model_type") == "embedding":
                    self._default_embedding_model = m.id
                    return self._default_embedding_model
        
        # Fallback to configured default
        self._default_embedding_model = settings.DEFAULT_EMBEDDING_MODEL
        return self._default_embedding_model
    
    # ==================== Inference API ====================
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion using the Llama Stack inference API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier (defaults to configured model)
            stream: Whether to stream the response
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Chat completion response
        """
        model_id = model or self.get_default_llm()
        
        params = {
            "model": model_id,
            "messages": messages,
            "stream": stream,
        }
        
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        params.update(kwargs)
        
        response = self.client.chat.completions.create(**params)
        
        if stream:
            return response  # Return generator for streaming
        
        # Format response
        return {
            "id": response.id if hasattr(response, 'id') else str(uuid.uuid4()),
            "model": model_id,
            "choices": [
                {
                    "index": i,
                    "message": {
                        "role": choice.message.role if hasattr(choice.message, 'role') else "assistant",
                        "content": choice.message.content if hasattr(choice.message, 'content') else str(choice.message),
                    },
                    "finish_reason": choice.finish_reason if hasattr(choice, 'finish_reason') else "stop",
                }
                for i, choice in enumerate(response.choices)
            ] if hasattr(response, 'choices') else [],
            "usage": response.usage if hasattr(response, 'usage') else {},
        }
    
    def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream chat completion responses."""
        model_id = model or self.get_default_llm()
        
        response = self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            stream=True,
            **kwargs
        )
        
        for chunk in response:
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta if hasattr(chunk.choices[0], 'delta') else None
                if delta and hasattr(delta, 'content') and delta.content:
                    yield delta.content
    
    # ==================== Agents API ====================
    
    def create_agent(
        self,
        instructions: Optional[str] = None,
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Agent:
        """
        Create a new Llama Stack agent.
        
        Args:
            instructions: System instructions for the agent
            model: Model to use (defaults to configured model)
            tools: List of tools to enable for the agent
            
        Returns:
            Agent instance
        """
        model_id = model or self.get_default_llm()
        agent_instructions = instructions or settings.DEFAULT_AGENT_INSTRUCTIONS
        
        agent = Agent(
            self.client,
            model=model_id,
            instructions=agent_instructions,
            tools=tools or [],
        )
        
        return agent
    
    def create_agent_session(self, agent: Agent, session_name: Optional[str] = None) -> str:
        """Create a new session for an agent."""
        name = session_name or f"session-{uuid.uuid4().hex[:8]}"
        session_id = agent.create_session(session_name=name)
        return session_id
    
    def agent_turn(
        self,
        agent: Agent,
        session_id: str,
        message: str,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a turn in an agent session.
        
        Args:
            agent: The agent instance
            session_id: Session identifier
            message: User message
            stream: Whether to stream the response
            
        Returns:
            Agent response
        """
        messages = [{"role": "user", "content": message}]
        
        if stream:
            return agent.create_turn(
                messages=messages,
                session_id=session_id,
                stream=True,
            )
        
        response = agent.create_turn(
            messages=messages,
            session_id=session_id,
            stream=False,
        )
        
        return {
            "turn_id": response.turn_id if hasattr(response, 'turn_id') else str(uuid.uuid4()),
            "session_id": session_id,
            "output_message": {
                "role": "assistant",
                "content": response.output_message.content if hasattr(response, 'output_message') else str(response),
            },
            "steps": [
                {
                    "step_id": step.step_id if hasattr(step, 'step_id') else str(i),
                    "step_type": step.step_type if hasattr(step, 'step_type') else "inference",
                }
                for i, step in enumerate(response.steps if hasattr(response, 'steps') else [])
            ],
        }
    
    # ==================== Vector Store & RAG API ====================
    
    def register_vector_db(
        self,
        vector_db_id: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> str:
        """
        Register a new vector store for RAG.
        
        Args:
            vector_db_id: Optional identifier for the vector store
            embedding_model: Embedding model to use
            
        Returns:
            Vector store identifier
        """
        db_id = vector_db_id or f"vectordb-{uuid.uuid4().hex[:8]}"
        embed_model = embedding_model or self.get_default_embedding_model()
        
        # Use vector_stores.create API (v0.5.0+)
        result = self.client.vector_stores.create(
            name=db_id,
            embedding_model=embed_model,
            chunking_strategy={
                "type": "auto",
            },
        )
        
        # Return the identifier from the response
        return result.id if hasattr(result, 'id') else db_id
    
    def insert_documents(
        self,
        vector_db_id: str,
        documents: List[Dict[str, Any]],
        chunk_size_in_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Insert documents into a vector store for RAG.
        
        Args:
            vector_db_id: Vector store identifier
            documents: List of documents with 'content' and optional 'metadata'
            chunk_size_in_tokens: Chunk size for document splitting
            
        Returns:
            Insert result
        """
        # Format documents for vector_io.insert API (v0.5.0+)
        chunks = []
        for doc in documents:
            doc_id = doc.get("document_id", f"doc-{uuid.uuid4().hex[:8]}")
            content = doc["content"]
            metadata = doc.get("metadata", {})
            
            chunks.append({
                "content": content,
                "metadata": {
                    "document_id": doc_id,
                    **metadata,
                },
            })
        
        self.client.vector_io.insert(
            vector_store_id=vector_db_id,
            chunks=chunks,
        )
        
        return {
            "status": "success",
            "vector_db_id": vector_db_id,
            "document_count": len(chunks),
        }
    
    def create_rag_agent(
        self,
        vector_db_ids: List[str],
        instructions: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Agent:
        """
        Create a RAG-enabled agent with knowledge search capabilities.
        
        Args:
            vector_db_ids: List of vector store IDs to search
            instructions: Agent instructions
            model: Model to use
            
        Returns:
            RAG Agent instance
        """
        model_id = model or self.get_default_llm()
        agent_instructions = instructions or "You are a helpful assistant. Use the RAG tool to answer questions as needed."
        
        # Use vector_store_ids for v0.5.0+
        tools = [
            {
                "name": "builtin::rag/knowledge_search",
                "args": {"vector_store_ids": vector_db_ids},
            }
        ]
        
        agent = Agent(
            self.client,
            model=model_id,
            instructions=agent_instructions,
            tools=tools,
        )
        
        return agent
    
    # ==================== Embeddings API ====================
    
    def create_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create embeddings for the given texts.
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use
            
        Returns:
            Embeddings response
        """
        model_id = model or self.get_default_embedding_model()
        
        # Use the embeddings.create API (v0.5.0+)
        response = self.client.embeddings.create(
            model=model_id,
            input=texts,
        )
        
        # Handle response format
        embeddings_data = []
        if hasattr(response, 'data'):
            for i, item in enumerate(response.data):
                embeddings_data.append({
                    "index": i,
                    "embedding": item.embedding if hasattr(item, 'embedding') else item,
                })
        elif hasattr(response, 'embeddings'):
            for i, emb in enumerate(response.embeddings):
                embeddings_data.append({
                    "index": i,
                    "embedding": emb.embedding if hasattr(emb, 'embedding') else emb,
                })
        
        return {
            "model": model_id,
            "embeddings": embeddings_data,
        }
    
    # ==================== Safety API ====================
    
    def run_shields(
        self,
        messages: List[Dict[str, str]],
        shields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run safety shields on messages.
        
        Args:
            messages: Messages to check
            shields: List of shield identifiers to use
            
        Returns:
            Safety check results
        """
        # List available shields if not specified
        if not shields:
            available_shields = self.client.shields.list()
            shields = [s.identifier if hasattr(s, 'identifier') else str(s) for s in available_shields]
        
        if not shields:
            return {
                "status": "skipped",
                "message": "No shields available",
                "results": [],
            }
        
        results = []
        for shield_id in shields:
            try:
                result = self.client.safety.run_shield(
                    shield_id=shield_id,
                    messages=messages,
                )
                results.append({
                    "shield_id": shield_id,
                    "violation": result.violation if hasattr(result, 'violation') else None,
                    "is_safe": not (result.violation if hasattr(result, 'violation') else False),
                })
            except Exception as e:
                results.append({
                    "shield_id": shield_id,
                    "error": str(e),
                    "is_safe": True,  # Assume safe on error
                })
        
        return {
            "status": "completed",
            "results": results,
            "is_safe": all(r.get("is_safe", True) for r in results),
        }
    
    def list_shields(self) -> List[Dict[str, Any]]:
        """List available safety shields."""
        try:
            shields = self.client.shields.list()
            return [
                {
                    "identifier": s.identifier if hasattr(s, 'identifier') else str(s),
                    "provider_id": s.provider_id if hasattr(s, 'provider_id') else None,
                }
                for s in shields
            ]
        except Exception:
            return []


# Singleton instance for the service
_service_instance: Optional[LlamaStackService] = None


def get_llama_stack_service() -> LlamaStackService:
    """Get or create the singleton LlamaStackService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = LlamaStackService()
    return _service_instance