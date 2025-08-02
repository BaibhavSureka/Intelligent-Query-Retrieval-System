#!/usr/bin/env python3
"""
Startup script for the LLM-Powered Query Retrieval System
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def check_environment():
    """Check if environment is properly configured"""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or environment")
        sys.exit(1)
    
    print("✅ Environment variables configured")

def install_dependencies():
    """Install dependencies if needed"""
    try:
        import fastapi
        import openai
        import sentence_transformers
        import faiss
        print("✅ Core dependencies available")
    except ImportError as e:
        print(f"⚠️  Installing missing dependency: {e.name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def create_directories():
    """Create necessary directories"""
    directories = ["faiss_index", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Required directories created")

def run_application():
    """Run the FastAPI application"""
    print("🚀 Starting LLM-Powered Query Retrieval System...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📊 Health check: http://localhost:8000/health")
    print("📖 API docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Import and run
    try:
        from main import app
        import uvicorn
        from config import settings
        
        uvicorn.run(
            app,
            host=settings.api_host,
            port=settings.api_port,
            log_level=settings.log_level.lower()
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

def main():
    """Main startup sequence"""
    print("🔧 LLM-Powered Query Retrieval System - Startup")
    print("=" * 50)
    
    # Pre-flight checks
    check_python_version()
    
    # Load environment variables from .env if it exists
    if Path(".env").exists():
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded .env file")
    
    check_environment()
    install_dependencies()
    create_directories()
    
    print("=" * 50)
    
    # Start the application
    run_application()

if __name__ == "__main__":
    main()