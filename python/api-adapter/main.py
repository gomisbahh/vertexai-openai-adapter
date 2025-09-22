#!/usr/bin/env python3
"""
FastAPI-based OpenAI-compatible API server for Vertex AI Gemma3 model.

Installation:
pip install fastapi uvicorn google-auth google-auth-oauthlib requests pydantic

Usage:
1. Configure your Vertex AI settings
2. Run: python main.py
3. API will be available at http://localhost:8000
4. Documentation at http://localhost:8000/docs
"""

import os
import time
import uuid
import json
import asyncio
from typing import List, Optional, Dict, Any, Union, AsyncGenerator
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import urllib3
import uvicorn
import requests
from google.auth.transport.requests import Request as AuthRequest
from google.oauth2 import service_account
import google.auth
from dotenv import load_dotenv

load_dotenv()


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Pydantic models for OpenAI API compatibility
class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message author")
    content: str = Field(..., description="The content of the message")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    max_tokens: Optional[int] = Field(default=150, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2, description="Sampling temperature")
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1, description="Nucleus sampling parameter")
    top_k: Optional[int] = Field(default=40, description="Top-k sampling parameter")
    stream: Optional[bool] = Field(default=False, description="Whether to stream responses")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop sequences")

class CompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    prompt: Union[str, List[str]] = Field(..., description="The prompt(s) to generate completions for")
    max_tokens: Optional[int] = Field(default=150, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2, description="Sampling temperature")
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1, description="Nucleus sampling parameter")
    stream: Optional[bool] = Field(default=False, description="Whether to stream responses")

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str] = "stop"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

# Vertex AI Client
class VertexAIClient:
    def __init__(self, project_id: str, location: str, endpoint_id: str):
        self.project_id = project_id
        self.location = location
        self.endpoint_id = endpoint_id

        endpoint_type = os.getenv("ENDPOINT_TYPE", "PRIVATE")
        endpoint_host = os.getenv("ENDPOINT_HOST", "10.132.8.10")

        if endpoint_type == "PRIVATE":
            self.endpoint_url = (
                f"https://{endpoint_host}"
                f"/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:predict"
            )
        else:
        
            self.endpoint_url = (
                f"https://{endpoint_id}.{location}-{project_id}.prediction.vertexai.goog"
                f"/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:predict"
            )
        
        self.credentials = self._setup_credentials()
        
    def _setup_credentials(self):
        try:
            credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
            return credentials
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to setup credentials: {e}")
    
    def _get_access_token(self) -> str:
        self.credentials.refresh(AuthRequest())
        return self.credentials.token

    async def generate_text(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Generate text using Vertex AI endpoint and return the prediction object."""
        try:
            access_token = self._get_access_token()
            #print("token"+access_token)

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }

            # Create payload for Gemma3 to match the provided Java example
            instance = {
                "@requestFormat": "chatCompletions",
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", 150),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 40),
            }
            if "stop" in kwargs and kwargs.get("stop"):
                instance["stop_sequences"] = kwargs["stop"]

            payload = {
                "instances": [instance]
            }

            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                verify=False, 
                timeout=120
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Vertex AI API error: {response.text}"
                )
            
            result = response.json()
            return self._parse_response(result)

        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Request failed: {e}")
    
    def _parse_response(self, response_data: Dict) -> Dict[str, Any]:
        """Parse response from Vertex AI and return the prediction object."""
        try:
            print(response_data)
            # The 'predictions' field from Vertex AI contains the OpenAI-compatible response.
            predictions = response_data.get("predictions")
            if not predictions:
                raise ValueError("No 'predictions' field in Vertex AI response")

            # The API can return a list of predictions or a single prediction object.
            # We handle both cases, taking the first prediction if it's a list.
            if isinstance(predictions, list):
                if not predictions:
                    raise ValueError("Empty 'predictions' list in Vertex AI response")
                prediction_object = predictions[0]
            elif isinstance(predictions, dict):
                prediction_object = predictions
            else:
                raise TypeError(f"'predictions' field is not a dictionary or list: {type(predictions)}")

            if not isinstance(prediction_object, dict):
                 raise TypeError(f"Prediction is not a dictionary: {type(prediction_object)}")

            return prediction_object
        except (ValueError, TypeError, KeyError, IndexError) as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error parsing Vertex AI response: {e}. Full response: {response_data}"
            )

# FastAPI Application
app = FastAPI(
    title="OpenAI-Compatible API for Google Vertex AI",
    description="OpenAI-compatible REST API server for Google Vertex AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    swagger_ui_css_url="/static/custom.css",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory of the current script to build absolute paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# STATIC_DIR = os.path.join(BASE_DIR, "static")

# # Mount static files for theme and custom CSS
# app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Configuration
class Config:
    PROJECT_ID = os.getenv("PROJECT_ID", "egr-appmod")  # Replace with your project ID
    LOCATION = os.getenv("LOCATION", "europe-west1")  # Replace with your location
    ENDPOINT_ID = os.getenv("ENDPOINT_ID", "5010777952984498176")  #private endpoint
    API_KEY = "sk-1234567890abcdef"  # Change this to your desired API key
    AVAILABLE_MODELS = ["google/vertexai/gemma3"]

config = Config()

# Initialize Vertex AI client
vertex_client = VertexAIClient(
    project_id=config.PROJECT_ID,
    location=config.LOCATION,
    endpoint_id=config.ENDPOINT_ID,
)

# Authentication
async def verify_api_key(authorization: Optional[str] = Header(None)):
    # if not authorization:
    #     raise HTTPException(status_code=401, detail="Authorization header required")
    
    # if not authorization.startswith("Bearer "):
    #     raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    # api_key = authorization.split(" ")[1]
    # if api_key != config.API_KEY:
    #     raise HTTPException(status_code=401, detail="Invalid API key")
    
    #return api_key
    return ""

# Utility functions
def estimate_tokens(text: str) -> int:
    """Rough token estimation"""
    return len(text.split())

# API Endpoints
@app.get("/v1/models", response_model=ModelList)
async def list_models(api_key: str = Depends(verify_api_key)):
    """List available models"""
    models = []
    for model_id in config.AVAILABLE_MODELS:
        models.append(ModelInfo(
            id=model_id,
            created=1677610602,
            owned_by="AI Team"
        ))
    
    return ModelList(data=models)

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
):
    """Create a chat completion"""
    if request.model not in config.AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not found")
    
    try:
        # Convert messages to list of dicts
        messages = [msg.dict() for msg in request.messages]

        # Generate response
        vertex_response = await vertex_client.generate_text(
            messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop
        )
        # The model name in the request might be an alias. We should use it.
        vertex_response['model'] = request.model
        # The response from Vertex is already in the desired format.
        # We can pass it directly to the Pydantic model for validation and response.
        return ChatCompletionResponse(**vertex_response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest,
    api_key: str = Depends(verify_api_key)
):
    """Create a text completion"""
    if request.model not in config.AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not found")
    
    try:
        # Handle single prompt or list of prompts
        if isinstance(request.prompt, list):
            prompt = request.prompt[0]  # Use first prompt for simplicity
        else:
            prompt = request.prompt
        
        messages = [{"role": "user", "content": prompt}]

        # Generate response
        vertex_response = await vertex_client.generate_text(
            messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Adapt the chat completion response to a text completion response
        response_text = vertex_response['choices'][0]['message']['content']
        
        # Get usage from vertex response
        usage_data = vertex_response.get('usage', {})
        prompt_tokens = usage_data.get('prompt_tokens', estimate_tokens(prompt))
        completion_tokens = usage_data.get('completion_tokens', estimate_tokens(response_text))
        
        return CompletionResponse(
            id=vertex_response.get('id', f"cmpl-{uuid.uuid4().hex[:10]}"),
            created=vertex_response.get('created', int(time.time())),
            model=request.model,
            choices=[CompletionChoice(
                index=0,
                text=response_text,
                finish_reason=vertex_response['choices'][0].get('finish_reason', 'stop')
            )],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": int(time.time())}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "OpenAI-Compatible API for Vertex AI Gemma3",
        "version": "1.0.0",
        "endpoints": {
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions",
            "completions": "/v1/completions",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    print("🚀 Starting OpenAI-Compatible API Server for Vertex AI Gemma3")
    print("=" * 60)
    print(f"📍 Server URL: http://localhost:8000")
    print(f"📋 API Base: http://localhost:8000/v1")
    print(f"🔑 API Key: {config.API_KEY}")
    print(f"📚 Models: {', '.join(config.AVAILABLE_MODELS)}")
    print(f"📖 Documentation: http://localhost:8000/docs")
    print("\n🔧 Configuration:")
    print(f"  Project ID: {config.PROJECT_ID}")
    print(f"  Location: {config.LOCATION}")
    print(f"  Endpoint ID: {config.ENDPOINT_ID}")
    print("\n📝 Example cURL:")
    print("curl -X POST http://localhost:8000/v1/chat/completions \\")
    print(f"  -H 'Authorization: Bearer {config.API_KEY}' \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{")
    print('    "model": "google/vertexai/gemma3",')
    print('    "messages": [{"role": "user", "content": "Hello!"}],')
    print('    "max_tokens": 150')
    print("  }'")
    print("=" * 60)

    # # Check for theme and static files using absolute paths
    # if not os.path.isdir(STATIC_DIR) or not os.path.isfile(os.path.join(STATIC_DIR, "background.jpg")):
    #     print("\n\u26a0\ufe0f WARNING: 'theme/background.jpg' not found. The Swagger UI background will not be customized.")
    # if not os.path.isdir(STATIC_DIR) or not os.path.isfile(os.path.join(STATIC_DIR, "custom.css")):
    #     print("\u26a0\ufe0f WARNING: 'static/custom.css' not found. The Swagger UI will not have custom styling.")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")