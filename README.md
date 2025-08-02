# LLM-Powered Intelligent Query-Retrieval System

A comprehensive system for processing documents (PDFs, DOCX, emails) and answering queries using LLM and vector search technology. Built for HackRx 6.0 competition.

## 🚀 Features

- **Multi-format Document Processing**: Supports PDF, DOCX, and email documents
- **Semantic Search**: Uses FAISS or Pinecone for vector-based similarity search
- **LLM Integration**: GPT-4 powered query processing and answer generation
- **Clause Matching**: Intelligent extraction and matching of document clauses
- **Explainable AI**: Provides decision rationale and confidence scores
- **RESTful API**: FastAPI-based API with comprehensive endpoints
- **Optimized Performance**: Token-efficient processing with response time optimization

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Docs    │ -> │   LLM Parser     │ -> │ Embedding Search│
│   PDF/DOCX/EML  │    │ Extract queries  │    │ FAISS/Pinecone  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   JSON Output   │ <- │ Logic Evaluation │ <- │ Clause Matching │
│ Structured resp │    │ Decision process │    │ Semantic sim.   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Requirements

- Python 3.10+
- OpenAI API key
- 8GB+ RAM (for embedding models)
- Optional: Pinecone API key (uses FAISS by default)

## 🛠️ Installation

### Local Development

1. **Clone the repository**
```bash
git clone <repository-url>
cd llm-query-retrieval-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Set OpenAI API Key**
```bash
export OPENAI_API_KEY="your-openai-api-key"
# Or add it to .env file
```

5. **Run the application**
```bash
python main.py
```

### Docker Deployment

1. **Build Docker image**
```bash
docker build -t llm-query-system .
```

2. **Run container**
```bash
docker run -d \
  --name llm-query-system \
  -p 8000:8000 \
  -e OPENAI_API_KEY="your-openai-api-key" \
  llm-query-system
```

### Cloud Deployment

#### Heroku
```bash
# Install Heroku CLI and login
heroku create your-app-name
heroku config:set OPENAI_API_KEY="your-openai-api-key"
git push heroku main
```

#### Railway
```bash
# Connect to Railway and deploy
railway login
railway new
railway add
railway deploy
```

## 📖 API Documentation

### Base URL
- Local: `http://localhost:8000`
- Production: `https://your-domain.com`

### Authentication
All endpoints require Bearer token authentication:
```
Authorization: Bearer 88def649851a3e0861a60905001a92f0a9cdd621ade7686aa6be07cc91f1ed9b
```

### Main Endpoint

#### POST `/hackrx/run`
Process documents and answer questions

**Request:**
```json
{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "Does this policy cover maternity expenses?",
        "What is the waiting period for cataract surgery?"
    ]
}
```

**Response:**
```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment after the due date.",
        "Yes, the policy covers maternity expenses after 24 months of continuous coverage.",
        "The policy has a specific waiting period of two (2) years for cataract surgery."
    ]
}
```

### Additional Endpoints

#### GET `/health`
Health check endpoint

#### POST `/api/v1/query`
Process single query (for testing)

#### GET `/api/v1/documents/{url}/info`
Get document processing information

#### DELETE `/api/v1/documents/{url}`
Remove document from cache

#### DELETE `/api/v1/cache`
Clear all document cache

## 🧪 Testing

### Sample Request
```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer 88def649851a3e0861a60905001a92f0a9cdd621ade7686aa6be07cc91f1ed9b" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
      "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
      "What is the waiting period for pre-existing diseases (PED) to be covered?"
    ]
  }'
```

### Python Test Script
```python
import requests

url = "http://localhost:8000/hackrx/run"
headers = {
    "Authorization": "Bearer 88def649851a3e0861a60905001a92f0a9cdd621ade7686aa6be07cc91f1ed9b",
    "Content-Type": "application/json"
}

data = {
    "documents": "https://example.com/policy.pdf",
    "questions": [
        "What is covered under this policy?",
        "What are the exclusions?"
    ]
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `PINECONE_API_KEY` | Pinecone API key (optional) | - |
| `PINECONE_ENVIRONMENT` | Pinecone environment | us-west1-gcp |
| `API_HOST` | API host | 0.0.0.0 |
| `API_PORT` | API port | 8000 |
| `LOG_LEVEL` | Logging level | INFO |

### System Configuration

Key settings in `config.py`:

- **Embedding Model**: `sentence-transformers/all-mpnet-base-v2`
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Max Tokens**: 2048
- **Temperature**: 0.1 (for consistency)

## 🏆 Performance Optimizations

### Token Efficiency
- Smart chunking with overlap
- Context-aware retrieval
- Optimized prompts
- Token counting and monitoring

### Latency Optimization
- Document caching
- Vector index persistence
- Parallel processing
- Efficient re-ranking

### Accuracy Improvements
- Multi-stage retrieval
- Clause pattern matching
- Question enhancement
- Confidence scoring

## 🔍 Features Deep Dive

### Document Processing
- **PDF**: PyMuPDF + PyPDF2 fallback
- **DOCX**: python-docx with table support
- **Email**: Email parser with reply extraction
- **Text Chunking**: Sentence-aware splitting

### Vector Search
- **FAISS**: Local vector storage (default)
- **Pinecone**: Cloud vector database (optional)
- **Embeddings**: Sentence transformers
- **Similarity**: Cosine similarity

### LLM Processing
- **Model**: GPT-4 (configurable)
- **Prompting**: Domain-specific templates
- **Context**: Top-K retrieval results
- **Output**: Structured JSON responses

### Clause Matching
- **Pattern Recognition**: Regex-based extraction
- **Relevance Scoring**: Jaccard similarity
- **Context Extraction**: Surrounding text
- **Ranking**: Multi-factor scoring

## 🚨 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   ```
   Solution: Set OPENAI_API_KEY environment variable
   ```

2. **Memory Issues**
   ```
   Solution: Reduce chunk_size or use smaller embedding model
   ```

3. **Slow Performance**
   ```
   Solution: Use Pinecone instead of FAISS for large documents
   ```

4. **Document Processing Fails**
   ```
   Solution: Check document URL accessibility and format support
   ```

### Logs and Monitoring

- Logs are written to console and can be redirected
- Health check endpoint: `/health`
- Configuration endpoint: `/api/v1/config`

## 📚 Technical Details

### Dependencies
- **FastAPI**: Web framework
- **OpenAI**: LLM integration
- **FAISS**: Vector similarity search
- **SentenceTransformers**: Text embeddings
- **PyMuPDF**: PDF processing
- **python-docx**: DOCX processing

### Architecture Patterns
- **Dependency Injection**: FastAPI dependencies
- **Factory Pattern**: Vector store creation
- **Strategy Pattern**: Document processors
- **Observer Pattern**: Logging system

## 📄 License

This project is developed for HackRx 6.0 competition.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📞 Support

For issues and questions:
- Check troubleshooting section
- Review logs for error details
- Ensure all dependencies are installed
- Verify API keys are correctly set

---

Built with ❤️ for HackRx 6.0 using FastAPI, GPT-4, and FAISS/Pinecone
