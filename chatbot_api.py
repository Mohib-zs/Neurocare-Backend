from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import os
from dotenv import load_dotenv
import json
import logging
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import PyPDF2
from datetime import datetime
import asyncio
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="NeuroCare Mental Health Chatbot API",
    description="State-of-the-art mental health chatbot with RAG, memory, and context awareness",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChromaDB
chroma_client = chromadb.Client(Settings(
    persist_directory="chroma_db",
    anonymized_telemetry=False
))

# Initialize embedding function
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create or get collection
collection = chroma_client.get_or_create_collection(
    name="mental_health_knowledge",
    embedding_function=embedding_function
)

# Initialize Mistral API
MISTRAL_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
MISTRAL_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY environment variable is not set")

# Configure retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)

# Create session with retry strategy
session = requests.Session()
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# Initialize conversation memory
conversation_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Custom prompt template for mental health conversations
MENTAL_HEALTH_PROMPT = """You are a compassionate and professional mental health chatbot. Your responses should be:
1. Empathetic and supportive
2. Evidence-based and accurate
3. Clear and easy to understand
4. Focused on the user's specific needs
5. Mindful of crisis situations

Current conversation:
{chat_history}

User: {input}
Assistant:"""

# Initialize conversation chain
conversation = ConversationChain(
    memory=conversation_memory,
    prompt=PromptTemplate(
        input_variables=["chat_history", "input"],
        template=MENTAL_HEALTH_PROMPT
    ),
    verbose=True
)

# Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    user_id: str
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    context: Dict[str, Any]
    relevant_docs: Optional[List[Dict[str, Any]]] = None

class Document(BaseModel):
    content: str
    metadata: Dict[str, Any]

# Helper functions
def load_pdf_to_chroma(pdf_path: str):
    """Load PDF content into ChromaDB."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                
                # Split text into chunks
                chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                
                # Add chunks to ChromaDB
                for i, chunk in enumerate(chunks):
                    collection.add(
                        documents=[chunk],
                        metadatas=[{
                            "source": pdf_path,
                            "page": page_num + 1,
                            "chunk": i + 1
                        }],
                        ids=[f"{pdf_path}_{page_num}_{i}"]
                    )
        logger.info(f"Successfully loaded PDF: {pdf_path}")
    except Exception as e:
        logger.error(f"Error loading PDF: {str(e)}")
        raise

async def call_mistral_api(prompt: str, context: Optional[Dict] = None) -> str:
    """Call Mistral API with context-aware prompting."""
    try:
        # Prepare system message
        system_message = """You are a compassionate and professional mental health chatbot. 
        Your responses should be empathetic, evidence-based, and focused on the user's specific needs.
        Always maintain professional boundaries and refer to mental health professionals when appropriate."""

        # Prepare messages
        messages = [
            {"role": "system", "content": system_message}
        ]

        # Add context if available
        if context:
            context_message = f"Context: {json.dumps(context, indent=2)}"
            messages.append({"role": "system", "content": context_message})

        # Add user message
        messages.append({"role": "user", "content": prompt})

        # Prepare API request
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "messages": messages,
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.95
        }

        # Make API call
        response = session.post(
            MISTRAL_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            logger.error(f"API error: {response.text}")
            raise HTTPException(status_code=500, detail="Error calling Mistral API")

    except Exception as e:
        logger.error(f"Error in Mistral API call: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_relevant_context(query: str, n_results: int = 3) -> List[Dict]:
    """Retrieve relevant context from ChromaDB."""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        relevant_docs = []
        for i in range(len(results["documents"][0])):
            relevant_docs.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
            })
        
        return relevant_docs
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        return []

# API endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Main chat endpoint with RAG and memory."""
    try:
        # Get relevant context from ChromaDB
        relevant_docs = get_relevant_context(request.message)
        
        # Prepare context
        context = {
            "timestamp": datetime.now().isoformat(),
            "user_id": request.user_id,
            "relevant_documents": relevant_docs,
            "previous_context": request.context or {}
        }
        
        # Update conversation memory
        conversation.memory.chat_memory.add_user_message(request.message)
        
        # Generate response using Mistral
        response = await call_mistral_api(
            prompt=request.message,
            context=context
        )
        
        # Update conversation memory
        conversation.memory.chat_memory.add_ai_message(response)
        
        # Update context with conversation history
        context["conversation_history"] = conversation.memory.chat_memory.messages
        
        return ChatResponse(
            response=response,
            context=context,
            relevant_docs=relevant_docs
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-document")
async def load_document(document: Document):
    """Load a document into the knowledge base."""
    try:
        # Add document to ChromaDB
        collection.add(
            documents=[document.content],
            metadatas=[document.metadata],
            ids=[f"doc_{datetime.now().timestamp()}"]
        )
        
        return {"message": "Document loaded successfully"}
        
    except Exception as e:
        logger.error(f"Error loading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-pdf")
async def load_pdf(pdf_path: str):
    """Load a PDF file into the knowledge base."""
    try:
        load_pdf_to_chroma(pdf_path)
        return {"message": "PDF loaded successfully"}
        
    except Exception as e:
        logger.error(f"Error loading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "chromadb": "connected",
            "mistral_api": "configured"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 