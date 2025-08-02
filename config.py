import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_version: str = "v1"
    
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = "gpt-4-1106-preview"
    max_tokens: int = 2048
    temperature: float = 0.1
    
    # Pinecone Configuration (optional)
    pinecone_api_key: Optional[str] = os.getenv("PINECONE_API_KEY")
    pinecone_environment: Optional[str] = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
    pinecone_index_name: str = "document-retrieval"
    
    # FAISS Configuration
    use_faiss: bool = True  # Set to False to use Pinecone instead
    faiss_index_path: str = "faiss_index"
    
    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dimension: int = 768
    
    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_document_size: int = 50 * 1024 * 1024  # 50MB
    
    # API Security
    bearer_token: str = "88def649851a3e0861a60905001a92f0a9cdd621ade7686aa6be07cc91f1ed9b"
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()