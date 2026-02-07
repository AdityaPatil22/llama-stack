# Llama Stack API

A comprehensive FastAPI application that integrates with [Llama Stack](https://llama-stack.readthedocs.io/) to provide unified APIs for building AI applications.

## Features

This application exposes all Llama Stack built-in APIs:

- **Inference** - Chat completions with streaming support
- **Models** - List and manage available models
- **Agents** - Create intelligent agents with tool support and session management
- **RAG** - Vector databases and retrieval-augmented generation
- **Embeddings** - Create text embeddings for semantic search
- **Safety** - Content moderation with safety shields

## Prerequisites

1. **Ollama** - Install from [ollama.com](https://ollama.com/download)
2. **Python 3.12+**
3. **Llama Stack Server** running locally

## Quick Start

### 1. Start Ollama with a model

```bash
ollama pull llama3.2:3b
ollama run llama3.2:3b --keepalive 60m
```

### 2. Start Llama Stack Server

```bash
# Install and run Llama Stack server
pip install llama-stack
OLLAMA_URL=http://localhost:11434 llama stack run starter
```

The Llama Stack server will be available at `http://localhost:8321`.

### 3. Start this API server

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

## API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger documentation.

## API Endpoints

### Inference

```bash
# Simple chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is machine learning?"}'

# Full chat completions (OpenAI-compatible format)
curl -X POST http://localhost:8000/api/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "temperature": 0.7
  }'
```

### Models

```bash
# List all models
curl http://localhost:8000/api/models

# List LLM models only
curl http://localhost:8000/api/models/llm

# List embedding models
curl http://localhost:8000/api/models/embedding
```

### Agents

Agents combine LLM reasoning with tools for taking actions.

```bash
# Quick one-shot agent
curl -X POST http://localhost:8000/api/agents/quick \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the capital of France?",
    "instructions": "You are a geography expert."
  }'

# Create a persistent agent
curl -X POST http://localhost:8000/api/agents \
  -H "Content-Type: application/json" \
  -d '{
    "instructions": "You are a helpful coding assistant.",
    "tools": []
  }'

# Create a session for multi-turn conversations
curl -X POST http://localhost:8000/api/agents/sessions \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "agent-xxxxxxxx"}'

# Create a turn in the session
curl -X POST http://localhost:8000/api/agents/turns \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent-xxxxxxxx",
    "session_id": "session-id",
    "message": "Help me write a Python function"
  }'
```

### RAG (Retrieval-Augmented Generation)

Build knowledge-enhanced agents that can answer questions from your documents.

```bash
# Create a vector database
curl -X POST http://localhost:8000/api/rag/vector-dbs \
  -H "Content-Type: application/json" \
  -d '{}'

# Insert documents (simple)
curl -X POST http://localhost:8000/api/rag/documents/simple \
  -H "Content-Type: application/json" \
  -d '{
    "vector_db_id": "vectordb-xxxxxxxx",
    "contents": [
      "Llama Stack is a unified API for building AI applications.",
      "It supports inference, agents, RAG, and safety features.",
      "Llama Stack works with multiple providers including Ollama."
    ]
  }'

# Insert documents with metadata
curl -X POST http://localhost:8000/api/rag/documents \
  -H "Content-Type: application/json" \
  -d '{
    "vector_db_id": "vectordb-xxxxxxxx",
    "documents": [
      {
        "content": "https://raw.githubusercontent.com/pytorch/torchtune/main/README.md",
        "mime_type": "text/plain",
        "metadata": {"source": "github"}
      }
    ]
  }'

# Create a RAG agent
curl -X POST http://localhost:8000/api/rag/agents \
  -H "Content-Type: application/json" \
  -d '{
    "vector_db_ids": ["vectordb-xxxxxxxx"],
    "instructions": "Answer questions using the provided knowledge base."
  }'

# Query the RAG agent
curl -X POST http://localhost:8000/api/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "rag_agent_id": "rag-agent-xxxxxxxx",
    "message": "What is Llama Stack?"
  }'

# Quick one-shot RAG query
curl -X POST http://localhost:8000/api/rag/quick \
  -H "Content-Type: application/json" \
  -d '{
    "vector_db_id": "vectordb-xxxxxxxx",
    "message": "Tell me about the features"
  }'
```

### Embeddings

Create vector embeddings for text.

```bash
# Create embeddings for multiple texts
curl -X POST http://localhost:8000/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "Machine learning is amazing"]}'

# Create embedding for single text
curl -X POST http://localhost:8000/api/embeddings/single \
  -H "Content-Type: application/json" \
  -d '{"text": "What is artificial intelligence?"}'
```

### Safety

Content moderation with safety shields.

```bash
# List available shields
curl http://localhost:8000/api/safety/shields

# Check message safety
curl -X POST http://localhost:8000/api/safety/check \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "How do I bake a cake?"}
    ]
  }'

# Simple content check
curl -X POST http://localhost:8000/api/safety/check/content \
  -H "Content-Type: application/json" \
  -d '{"content": "This is a test message"}'
```

## Configuration

Environment variables can be set in a `.env` file:

```env
# Llama Stack server URL
LLAMA_STACK_BASE_URL=http://localhost:8321

# Default models
DEFAULT_MODEL=ollama/llama3.2:3b
DEFAULT_EMBEDDING_MODEL=ollama/nomic-embed-text:v1.5

# Agent settings
DEFAULT_AGENT_INSTRUCTIONS=You are a helpful assistant.

# Safety
ENABLE_SAFETY_SHIELDS=true
```

## Project Structure

```
backend/
├── app/
│   ├── core/
│   │   └── config.py          # Configuration settings
│   ├── routes/
│   │   ├── chat.py            # Inference endpoints
│   │   ├── models.py          # Models endpoints
│   │   ├── agents.py          # Agents endpoints
│   │   ├── rag.py             # RAG endpoints
│   │   ├── embeddings.py      # Embeddings endpoints
│   │   └── safety.py          # Safety endpoints
│   ├── services/
│   │   └── llama_stack.py     # Llama Stack SDK wrapper
│   └── main.py                # FastAPI application
└── requirements.txt           # Python dependencies
```

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│   Your Application  │────▶│   This API Server   │────▶│  Llama Stack    │
│   (Frontend/CLI)    │     │   (FastAPI)         │     │  Server         │
└─────────────────────┘     └─────────────────────┘     └────────┬────────┘
                                                                  │
                                                                  ▼
                                                        ┌─────────────────┐
                                                        │     Ollama      │
                                                        │  (LLM Provider) │
                                                        └─────────────────┘
```

## License

MIT