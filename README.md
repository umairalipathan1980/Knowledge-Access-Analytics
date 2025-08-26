# Generic RAG Application

A powerful, customizable Retrieval-Augmented Generation (RAG) application that allows you to create and query your own knowledge bases using PDF documents.

## Features

### 🧠 **Smart Knowledge Base Management**
- **Create Knowledge Bases**: Upload PDF documents to create custom knowledge bases
- **Multiple Knowledge Bases**: Manage and switch between different knowledge bases
- **PDF Document Parsing**: Uses Docling library for advanced PDF parsing with OCR support

### 🔍 **Advanced Search Capabilities**
- **Document Retrieval**: Search through your knowledge base documents
- **Internet Search**: Get up-to-date information from the web
- **Hybrid Search**: Combine document and web search for comprehensive answers
- **Document Grading**: AI-powered relevance scoring of retrieved documents

### 🤖 **Flexible AI Integration**
- **Customizable Answer Styles**: Concise, Moderate, or Explanatory responses
- **Streaming Responses**: Real-time response generation

### 📊 **User-Friendly Interface**
- **Streamlit Web Interface**: Clean, intuitive web-based interface
- **Interactive Chat**: Conversational interface with follow-up questions
- **Document References**: Citations linking back to source documents
- **Debug Information**: Optional debug logs for troubleshooting

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd generic_RAG
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up API keys** in Streamlit secrets or environment variables:
```bash
# Required
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=lsv2_pt_your_langchain_api_key_here
AZURE_API_KEY=your_azure_openai_api_key_here

# Optional for LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langchain_api_key
```

4. **Create data directory**:
```bash
mkdir data
```

## Usage

### Running the Application

```bash
streamlit run app.py
```

### Creating Your First Knowledge Base

1. **Launch the application** - you'll see the knowledge base selection screen
2. **Click "Create New Knowledge Base"**
3. **Enter a name** for your knowledge base
4. **Upload PDF files** - select one or more PDF documents
5. **Choose embedding model** - select from available options
6. **Click "Create Knowledge Base"** - wait for processing to complete

### Querying Your Knowledge Base

1. **Ask questions** using the chat interface
2. **Adjust settings** in the sidebar:
   - **Answer Style**: Choose between Concise, Moderate, or Explanatory
   - **Internet Search**: Enable/disable web search
3. **Use follow-up questions** generated automatically based on responses

## File Structure

```
generic_RAG/
├── app.py                     # Main Streamlit application
├── agentic_rag.py             # Core RAG logic and workflows
├── pdf_parser.py              # PDF parsing using Docling
├── st_callback.py             # Streamlit callback handlers
├── requirements.txt           # Python dependencies
├── data/                      # Data directory
│   ├── chroma_db_*/          # Chroma vector databases
│   └── temp_pdfs/            # Temporary PDF storage
└── README.md                  # This file
```

## How It Works

### 1. **Document Processing Pipeline**
```
PDF Upload → Docling Parser → Markdown → Text Splitting → Embeddings → Chroma DB
```

### 2. **Query Processing Workflow**
```
User Question → Route Decision → Document Retrieval/Web Search → Document Grading → Response Generation
```

### 3. **Agentic Workflow States**
- **Route Question**: Decide between document retrieval, web search, or hybrid
- **Retrieve**: Get relevant documents from the knowledge base
- **Grade Documents**: Score document relevance using AI
- **Web Search**: Get current information from the internet
- **Generate**: Create final response using retrieved information

## Configuration Options

### Models
- **Main Model**: Primary LLM for generation (default: gpt-4o-mini)
- **Routing Model**: LLM for routing decisions (default: gpt-4o-mini)
- **Grading Model**: LLM for document relevance (default: gpt-4o-mini)
- **Embedding Model**: Text embedding model (default: text-embedding-3-large)

### Search Options
- **Document Search Only**: Search only in your knowledge base
- **Internet Search**: Use web search for current information
- **Hybrid Search**: Combine document and web search

### Answer Styles
- **Concise**: Short, direct answers
- **Moderate**: Balanced explanations with context
- **Explanatory**: Detailed responses with examples

## Advanced Features

### Custom Embeddings
```python
# Embedding model:
- text-embedding-3-large (OpenAI)
```


### Vector Database
- Uses Chroma DB for vector storage
- Persistent storage in `data/chroma_db_*` directories
- Supports multiple collections per knowledge base

## Troubleshooting

### Common Issues

1. **PDF Processing Errors**:
   - Ensure PDFs are not password protected
   - Check PDF file size (very large files may timeout)

2. **API Key Errors**:
   - Verify all required API keys are set
   - Check API key permissions and quotas

3. **Streamlit Issues**:
   - Clear browser cache
   - Restart the Streamlit server
   - Check port availability (default: 8501)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the debug logs in the application

---

**Generic RAG Application** - Transform your documents into an intelligent, queryable knowledge base with the power of AI.