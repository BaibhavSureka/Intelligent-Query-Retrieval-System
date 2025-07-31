# LLM-Powered Intelligent Query-Retrieval System

A production-ready RAG (Retrieval-Augmented Generation) pipeline for processing large documents and answering natural language queries with explainable decision rationale. This system works with any type of document including legal contracts, technical manuals, research papers, policies, and more.

## 🚀 Features

- **Multi-format Document Support**: PDF, DOCX, and Email (.eml) files
- **Universal Document Processing**: Works with any type of document (legal, technical, academic, business, etc.)
- **Semantic Search**: FAISS vector store for efficient retrieval
- **Intelligent Answering**: GPT-3.5-turbo for fast, accurate responses
- **Explainable AI**: Decision rationale and confidence scoring
- **Production Ready**: FastAPI with authentication and error handling
- **Optimized Performance**: Fast response times with efficient token usage
- **Cloud Deployable**: Ready for Render, Railway, or other Python hosting platforms

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for document downloads

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/BaibhavSureka/Intelligent-Query-Retrieval-System.git
   cd Intelligent-Query-Retrieval-System
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

```bash
# Run the development server
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

## ☁️ Deployment

### Render Deployment

1. **Connect your GitHub repository to Render**
2. **Create a new Web Service**
3. **Configure the service:**
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.api:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables**: Add your `OPENAI_API_KEY`

### Railway Deployment

1. **Connect your GitHub repository to Railway**
2. **Railay will automatically detect Python and deploy**
3. **Add environment variables in Railway dashboard**

### Other Platforms

This application is compatible with any Python hosting platform that supports FastAPI applications.

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
    "What are the key terms and conditions?",
    "What is the scope of coverage?",
    "What are the limitations and exclusions?"
  ]
}
```

### Response Format

```json
{
  "answers": [
    "The key terms include a 30-day grace period and coverage up to $100,000.",
    "The scope covers medical expenses, hospitalization, and outpatient care.",
    "Limitations include pre-existing conditions and cosmetic procedures."
  ]
}
```

### Example with curl

```bash
curl -X POST "https://your-api-url.com/api/v1/hackrx/run" \
  -H "Authorization: Bearer your_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What are the main terms and conditions?"]
  }'
```

## 🏗️ Architecture

### Core Components

- **Document Loader**: Handles PDF, DOCX, and email file processing
- **Text Chunker**: Splits documents into manageable chunks for embedding
- **Embedder**: Converts text chunks to vector embeddings
- **Vector Store**: FAISS-based semantic search engine
- **Retriever**: Finds relevant chunks for user queries
- **Logic Engine**: GPT-3.5-turbo powered answer generation
- **API Layer**: FastAPI with authentication and error handling

### Supported Document Types

- **Legal Documents**: Contracts, agreements, terms of service
- **Technical Documents**: Manuals, specifications, procedures
- **Academic Papers**: Research papers, reports, studies
- **Business Documents**: Policies, procedures, guidelines
- **Medical Documents**: Reports, guidelines, protocols
- **Any Text-based Document**: The system adapts to any domain

### Data Flow

1. **Document Upload**: User provides document URL
2. **Text Extraction**: Document loader extracts text content
3. **Chunking**: Text is split into semantic chunks
4. **Embedding**: Chunks are converted to vector embeddings
5. **Storage**: Embeddings stored in FAISS vector database
6. **Query Processing**: User questions trigger semantic search
7. **Answer Generation**: Relevant chunks + LLM = accurate answers

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `PORT`: Server port (default: 8000)

### Authentication

The API uses Bearer token authentication. Set your token in the request headers:

```
Authorization: Bearer your_token_here
```

## 📊 Performance

- **Response Time**: < 5 seconds for typical queries
- **Token Efficiency**: Optimized chunking and context management
- **Accuracy**: High precision with explainable decision rationale
- **Scalability**: Vector-based retrieval scales with document size

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- OpenAI for GPT-3.5-turbo
- FAISS for vector similarity search
- FastAPI for the web framework
- Render for cloud deployment support
