# Rag(ChatBot) - Document-Based Chatbot System

A sophisticated document-based chatbot system that allows users to upload documents and interact with them through natural language queries. The system uses advanced embedding techniques and Large Language Models (LLMs) to provide intelligent responses based on document content.

## 🚀 Features

- **Document Upload & Processing**: Support for multiple file formats (PDF, TXT, CSV, MD)
- **Intelligent Document Retrieval**: Uses FAISS vector store for semantic search
- **Context-Aware Chatbot**: GPT-4 powered responses based on document content
- **Web Interface**: Both FastAPI backend and Streamlit frontend
- **Document Management**: Track, view, and manage uploaded documents
- **Exception Logging**: Comprehensive error tracking and logging system
- **Chat History**: Persistent chat history with user tracking
- **Multi-Document Support**: Handle multiple documents with unique hash codes

## 🏗️ Architecture

### Backend (FastAPI)
- RESTful API endpoints for document management and chat functionality
- SQLAlchemy ORM for database operations
- FAISS vector store for efficient similarity search
- OpenAI GPT-4 integration for natural language processing

### Frontend (Streamlit)
- Interactive web interface for document upload
- Real-time chat interface
- Document management dashboard

### Data Layer
- PostgreSQL database for metadata storage
- FAISS indexes for vector embeddings
- File system storage for uploaded documents

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL database
- OpenAI API key
- ngrok (for external access, optional)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CopyHaiJi
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   DATABASE_URL=postgresql://username:password@localhost:5432/copyhaiji_db
   ```

4. **Set up the database**
   ```bash
   # Create PostgreSQL database
   createdb copyhaiji_db
   
   # The application will automatically create tables on first run
   ```

5. **Create uploads directory**
   ```bash
   mkdir uploads
   ```

## 🚀 Running the Application

### Start the FastAPI Backend
```bash
# From the CopyHaiJi directory
python main.py
```
The API server will start on `http://localhost:8000`

### Start the Streamlit Frontend
```bash
# From the CopyHaiJi directory
streamlit run streamlit.py
```
The web interface will be available at `http://localhost:8501`

## 📡 API Endpoints

### Document Management
- `POST /upload-document` - Upload and process a new document
- `GET /documents` - List all uploaded documents
- `GET /documents/{document_id}` - Get specific document details
- `DELETE /documents/{document_id}` - Delete a document

### Chat Functionality
- `POST /chat` - Send a message to the chatbot
- `GET /chat/history/{user_id}` - Get chat history for a specific user
- `GET /chat/history` - Get all chat history

### System Monitoring
- `GET /health` - Health check endpoint
- `GET /api/exceptions/table` - View exception logs
- `DELETE /api/exceptions/cleanup` - Clean up old exception logs

## 💬 Using the Chatbot

1. **Upload a Document**: Use the Streamlit interface or API to upload a document
2. **Get Hash Code**: Each document gets a unique hash code for identification
3. **Start Chatting**: Ask questions about your document content
4. **Context-Aware Responses**: The bot will answer based only on the uploaded document

### Example API Usage

```python
import requests

# Upload a document
files = {'file': open('document.pdf', 'rb')}
data = {'chunk_size': 500, 'chunk_overlap': 200}
response = requests.post('http://localhost:8000/upload-document', files=files, data=data)

# Chat with the document
chat_data = {
    "message": "What is the main topic of this document?",
    "hash_code": "your_document_hash_code"
}
response = requests.post('http://localhost:8000/chat', json=chat_data)
print(response.json()['response'])
```

## 🔧 Configuration

### Document Processing Parameters
- **chunk_size**: Size of text chunks for embedding (default: 500)
- **chunk_overlap**: Overlap between chunks (default: 200)
- **similarity_search_k**: Number of similar chunks to retrieve (default: 4)

### LLM Settings
- **Model**: GPT-4 (configurable in `Chatbot_retriver.py`)
- **Temperature**: 0.2 (for consistent responses)

## 📁 Project Structure

```
CopyHaiJi/
├── main.py                 # FastAPI application entry point
├── streamlit.py           # Streamlit web interface
├── requirements.txt       # Python dependencies
├── uploads/              # Directory for uploaded files
└── Router/
    ├── Chatbot_retriver.py  # Core chatbot logic and retrieval
    ├── database.py          # Database configuration
    ├── embedding.py         # Document embedding and processing
    ├── relations.py         # SQLAlchemy models
    ├── table_creater.py     # Database table creation
    ├── exception_utils.py   # Error logging utilities
    └── embedding/           # FAISS vector store files
        ├── document_index/  # Global document index
        └── [hash_dirs]/     # Individual document embeddings
```

## 🔒 Security Features

- **PII Protection**: Automatic masking of personally identifiable information
- **Input Validation**: Comprehensive request validation with error logging
- **Exception Handling**: Robust error handling with database logging
- **File Type Validation**: Restricted file upload types

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Ensure your OpenAI API key is set in the `.env` file
   - Verify the key has sufficient credits

2. **Database Connection Error**
   - Check PostgreSQL is running
   - Verify database URL in environment variables
   - Ensure database exists

3. **File Upload Issues**
   - Check file permissions in uploads directory
   - Verify supported file formats (PDF, TXT, CSV, MD)

4. **Embedding Generation Fails**
   - Check internet connection for model downloads
   - Ensure sufficient disk space for FAISS indexes

### Debug Mode
Enable debug logging by setting environment variable:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## 📊 Monitoring

The system includes comprehensive logging:
- Exception logs stored in database
- Chat history tracking
- Document processing status
- API request/response logging

Access logs via the `/api/exceptions/table` endpoint.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-4 API
- LangChain for document processing framework
- FAISS for efficient vector similarity search
- FastAPI for the robust web framework
- Streamlit for the interactive web interface

