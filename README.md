# LLM-Powered Intelligent Query–Retrieval System

A production-ready RAG (Retrieval-Augmented Generation) pipeline for processing large documents and answering natural language queries with explainable responses.

## 🚀 Features

### Core Capabilities

- **Multi-format Document Processing**: PDF, DOCX, and Email (.eml) files
- **Semantic Search**: FAISS-based vector similarity search
- **Explainable AI**: Detailed responses with supporting clauses and rationale
- **Confidence Scoring**: Built-in confidence assessment for answers
- **Authentication**: Bearer token-based API security
- **Error Handling**: Comprehensive validation and error responses

### Technical Specifications

- **Backend**: FastAPI with async processing
- **Vector Database**: FAISS for efficient similarity search
- **LLM**: OpenAI GPT-4 for answer generation
- **Embeddings**: OpenAI text-embedding-ada-002
- **Token Management**: Optimized chunking and context management

## 📋 Requirements

- Python 3.11+
- OpenAI API key
- Internet connection for document downloads

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/BaibhavSureka/-Intelligent-Query-Retrieval-System-.git
   cd -Intelligent-Query-Retrieval-System-
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Copy the example environment file and add your OpenAI API key:
   ```bash
   cp env.example .env
   ```
   
   Then edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-actual-openai-api-key-here
   ```

## 🚀 Usage

### Start the API Server

```bash
uvicorn app.api:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoint

**POST** `/api/v1/hackrx/run`

**Headers:**

```
Authorization: Bearer 88def649851a3e0861a60905001a92f0a9cdd621ade7686aa6be07cc91f1ed9b
Content-Type: application/json
```

**Request Body:**

```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "Does this policy cover knee surgery?",
    "What is the waiting period for pre-existing diseases?"
  ]
}
```

**Response:**

```json
{
  "answers": [
    "Yes, this policy covers knee surgery under clause 4.2...",
    "The waiting period is 36 months as stated in clause 6.1..."
  ],
  "detailed_answers": [
    {
      "answer": "Yes, this policy covers knee surgery...",
      "supporting_clauses": [
        { "text": "4.2. Surgical Procedures", "location": "Chunk 3" }
      ],
      "decision_rationale": "The answer was derived from clause 4.2...",
      "confidence_score": 0.85
    }
  ],
  "processing_time": 2.34,
  "document_processed": "https://example.com/policy.pdf"
}
```

## 📁 Project Structure

```
llm/
├── app/
│   ├── api.py              # FastAPI endpoints with authentication
│   ├── schemas.py          # Pydantic models for request/response
│   ├── document_loader.py  # PDF/DOCX/Email document processing
│   ├── chunker.py          # Text chunking and preprocessing
│   ├── embedder.py         # OpenAI embeddings with token management
│   ├── vector_store.py     # FAISS vector database interface
│   ├── retriever.py        # Semantic search and retrieval
│   └── logic.py            # LLM-based answer generation
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── env.example             # Environment variables template
└── README.md
```

## 🔧 System Architecture

1. **Document Input**: PDF/DOCX/Email URL processing
2. **Text Extraction**: Multi-format document parsing
3. **Chunking**: Intelligent text segmentation
4. **Embedding**: OpenAI embedding generation
5. **Vector Storage**: FAISS similarity indexing
6. **Query Processing**: Natural language question parsing
7. **Semantic Search**: Relevant chunk retrieval
8. **Answer Generation**: GPT-4 with explainability
9. **Response Formatting**: Structured JSON output

## 🛡️ Error Handling

The system includes comprehensive error handling for:

- Invalid document URLs
- Unsupported file formats
- Empty or corrupted documents
- API rate limits
- Network connectivity issues
- Token limit exceeded

## 📊 Performance Features

- **Token Efficiency**: Optimized chunking and context management
- **Latency**: Fast retrieval with FAISS vector search
- **Scalability**: Modular architecture for easy extension
- **Monitoring**: Processing time tracking and debug logging

## 🔐 Security

- Bearer token authentication
- Input validation and sanitization
- Error message sanitization
- Secure document processing

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t llm-rag .
docker run -p 8000:8000 -e OPENAI_API_KEY=your-key llm-rag
```

## 📈 Evaluation Metrics

The system is designed to meet HackRx 6.0 evaluation criteria:

- **Accuracy**: Precision in query understanding and clause matching
- **Token Efficiency**: Optimized LLM usage and cost-effectiveness
- **Latency**: Fast response times and real-time performance
- **Reusability**: Modular code structure and extensibility
- **Explainability**: Clear decision reasoning and clause traceability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:

1. Check the debug logs for detailed error information
2. Verify your OpenAI API key is set correctly
3. Ensure document URLs are accessible
4. Check that questions are properly formatted

## 🚀 Quick Start

1. **Clone and setup:**
   ```bash
   git clone https://github.com/BaibhavSureka/-Intelligent-Query-Retrieval-System-.git
   cd -Intelligent-Query-Retrieval-System-
   pip install -r requirements.txt
   cp env.example .env
   # Edit .env with your OpenAI API key
   ```

2. **Run the server:**
   ```bash
   uvicorn app.api:app --reload
   ```

3. **Test with Postman:**
   - URL: `POST http://localhost:8000/api/v1/hackrx/run`
   - Headers: `Authorization: Bearer 88def649851a3e0861a60905001a92f0a9cdd621ade7686aa6be07cc91f1ed9b`
   - Body: Your document URL and questions

---

**Built for HackRx 6.0 - Bajaj Finserv Health**
