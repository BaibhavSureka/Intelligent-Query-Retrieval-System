# LLM-Powered Intelligent Query-Retrieval System

A production-ready RAG (Retrieval-Augmented Generation) pipeline for processing large documents and answering natural language queries with explainable decision rationale.

## 🚀 Features

- **Multi-format Document Support**: PDF, DOCX, and Email (.eml) files
- **Semantic Search**: FAISS vector store for efficient retrieval
- **Intelligent Answering**: GPT-3.5-turbo for fast, accurate responses
- **Explainable AI**: Decision rationale and confidence scoring
- **Production Ready**: FastAPI with authentication and error handling
- **Optimized Performance**: Fast response times with efficient token usage

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for document downloads

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd llm
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   ```bash
   # Copy the example file
   cp env.example .env

   # Edit .env and add your OpenAI API key
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🏃‍♂️ Running Locally

### Quick Start

```bash
# Run the development server
python run_local.py
```

### Manual Start

```bash
# Using uvicorn directly
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

### Testing the API

```bash
# Test with sample data
python test_api.py
```

## 🌐 API Endpoints

### Main Endpoint

- **URL**: `POST /api/v1/hackrx/run`
- **Authentication**: Bearer token required
- **Content-Type**: `application/json`

### Health Check

- **URL**: `GET /health`
- **Response**: `{"status": "healthy", "service": "RAG Pipeline"}`

### Root Endpoint

- **URL**: `GET /`
- **Response**: System information and status

## 📡 API Usage

### Request Format

```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?"
  ]
}
```

### Response Format

```json
{
  "answers": [
    "A grace period of thirty days is provided for premium payment...",
    "There is a waiting period of thirty-six months for pre-existing diseases..."
  ]
}
```

### Example with curl

```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 88def649851a3e0861a60905001a92f0a9cdd621ade7686aa6be07cc91f1ed9b" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?"
    ]
}'
```

## 🏗️ System Architecture

```
Document Input → Text Extraction → Chunking → Embedding → Vector Store → Retrieval → LLM Processing → Answer Generation
```

### Components

1. **Document Loader** (`app/document_loader.py`)

   - Downloads documents from URLs
   - Supports PDF, DOCX, and Email formats
   - Extracts text content

2. **Text Chunker** (`app/chunker.py`)

   - Splits documents into manageable chunks
   - Optimized for semantic meaning preservation

3. **Embedding Generator** (`app/embedder.py`)

   - Converts text chunks to vectors
   - Uses OpenAI's text-embedding-ada-002 model
   - Handles token limits efficiently

4. **Vector Store** (`app/vector_store.py`)

   - FAISS-based in-memory vector database
   - Fast similarity search

5. **Retriever** (`app/retriever.py`)

   - Finds relevant chunks for queries
   - Semantic search implementation

6. **Logic Engine** (`app/logic.py`)
   - GPT-3.5-turbo for answer generation
   - Optimized for speed and accuracy
   - Confidence scoring

## 🔧 Performance Optimizations

- **Fast Model**: Using GPT-3.5-turbo instead of GPT-4 for speed
- **Token Limits**: Efficient context management
- **Response Limits**: Max 500 tokens per answer
- **No Debug Output**: Clean, production-ready logging
- **Optimized Chunking**: Balanced chunk sizes for retrieval

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t llm-query-system .

# Run the container
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key llm-query-system
```

### Docker Compose

```bash
# Start with docker-compose
docker-compose up --build
```

## 🚀 Production Deployment

### Render

1. Connect your GitHub repository
2. Set environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
3. Deploy automatically

### Railway

1. Connect your GitHub repository
2. Add environment variables
3. Deploy with automatic scaling

### Heroku

```bash
# Create Heroku app
heroku create your-app-name

# Set environment variables
heroku config:set OPENAI_API_KEY=your_key

# Deploy
git push heroku main
```

## 📊 Evaluation Metrics

The system is optimized for:

- **Accuracy**: Precise query understanding and clause matching
- **Token Efficiency**: Optimized LLM usage and cost-effectiveness
- **Latency**: Fast response times (< 30 seconds)
- **Reusability**: Modular code structure
- **Explainability**: Clear decision reasoning

## 🔒 Security

- Bearer token authentication
- Input validation and sanitization
- Error handling without information leakage
- Secure document processing

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**

   ```bash
   # Check your .env file
   cat .env
   # Should contain: OPENAI_API_KEY=sk-...
   ```

2. **Port Already in Use**

   ```bash
   # Kill process on port 8000
   lsof -ti:8000 | xargs kill -9
   ```

3. **Module Not Found**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt
   ```

### Performance Tips

- Use smaller documents for faster processing
- Limit the number of questions per request
- Ensure stable internet connection for document downloads

## 📝 License

This project is developed for the HackRx 6.0 competition.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Ready for HackRx 6.0 Submission!** 🎯
