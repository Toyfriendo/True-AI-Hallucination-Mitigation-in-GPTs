# End-to-End RAG Pipeline for Reducing Hallucinations

This project implements a complete Retrieval-Augmented Generation (RAG) system that grounds chatbot responses in real documents to reduce hallucinations.

## Features

- **Document Processing**: Automatic chunking and preprocessing of text documents (supports TXT, PDF)
- **File Upload**: Manual document upload through web interface
- **Embedding Generation**: Uses sentence-transformers for high-quality embeddings
- **Vector Storage**: FAISS-based similarity search for efficient retrieval
- **LLM Integration**: OpenAI GPT integration with grounded prompting
- **Web Interface**: Streamlit-based demo application with drag-and-drop upload
- **Comparison Tool**: Side-by-side comparison of grounded vs ungrounded responses

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Step 1: Create Virtual Environment (Recommended)
```bash
python -m venv rag_env
# On Windows:
rag_env\Scripts\activate
# On macOS/Linux:
source rag_env/bin/activate
```

### Step 2: Install Dependencies

**Option A: Install from requirements.txt**
```bash
pip install -r requirements.txt
```

**Option B: Install specific tested versions**
```bash
pip install sentence-transformers==2.7.0
pip install openai==1.12.0
pip install streamlit==1.31.0
pip install pandas==2.2.0
pip install tqdm==4.66.0
pip install python-dotenv==1.0.1
pip install requests==2.31.0
```

**For PDF support, also install:**
```bash
pip install PyPDF2==3.0.1
```

### Step 3: Set up OpenAI API Key

1. Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_api_key_here
```

2. Or set environment variable:
```bash
# Windows
set OPENAI_API_KEY=your_api_key_here
# macOS/Linux
export OPENAI_API_KEY=your_api_key_here
```

## Quick Start

### Method 1: Web Interface (Recommended)
```bash
streamlit run demo_app.py
```
Then open your browser to the displayed URL (usually http://localhost:8501)

### Method 2: Command Line
```bash
python main.py
```

## Usage

### Using the Web Interface

1. **Upload Documents**: 
   - Use the file uploader to add TXT, PDF, DOCX, or MD files
   - Or build from the sample documents directory

2. **Ask Questions**: 
   - Enter your question in the text area
   - Click "Get Grounded Response" for document-based answers
   - Click "Compare Responses" to see grounded vs ungrounded responses

3. **View Sources**: 
   - Check the "Sources Used" section to see which document chunks were referenced
   - Verify the information by reviewing the source content

### Building a Knowledge Base Programmatically

```python
from rag_pipeline import RAGPipeline

# Initialize pipeline
rag = RAGPipeline()

# Build knowledge base from documents
rag.build_knowledge_base('./your_documents_directory')

# Query the system
result = rag.query("Your question here")
print(result['response'])
```

### Comparing Responses

```python
# Compare grounded vs ungrounded responses
comparison = rag.compare_responses("Your question here")
print("Grounded:", comparison['grounded_response'])
print("Ungrounded:", comparison['ungrounded_response'])
```

## Project Structure

- `config.py` - Configuration settings
- `document_processor.py` - Document loading and chunking (TXT, PDF support)
- `embedding_generator.py` - Embedding generation using sentence-transformers
- `vector_store.py` - FAISS-based vector storage and retrieval
- `llm_interface.py` - LLM integration and prompt construction
- `rag_pipeline.py` - Main RAG pipeline orchestration
- `demo_app.py` - Streamlit web interface with file upload
- `main.py` - Command-line demo
- `create_sample_data.py` - Sample document generator
- `requirements.txt` - Python dependencies
- `vector_store/` - Stored vector database files

## Configuration

Edit `config.py` to customize:
- Chunk size and overlap
- Embedding model
- Retrieval parameters
- LLM settings

## Sample Documents

The system includes sample documents on:
- AI and Machine Learning
- Climate Change
- Renewable Energy
- Space Exploration

Run `python create_sample_data.py` to generate these documents.

## Troubleshooting

### Common Issues

1. **OpenAI API Error**: Ensure your API key is correctly set in the `.env` file
2. **Package Conflicts**: Use the specific versions listed above to avoid compatibility issues
3. **PDF Processing Issues**: Install PyPDF2 for PDF support
4. **Torch Warnings**: The `torch.classes` warning is harmless and doesn't affect functionality

### Version Compatibility

This project has been tested with:
- Python 3.12
- The specific package versions listed in the installation section
- Windows 10/11 environment

## Key Benefits

1. **Reduced Hallucinations**: Responses are grounded in actual documents
2. **Source Attribution**: Every response includes source references
3. **Modular Design**: Easy to extend and customize
4. **Efficient Retrieval**: Fast similarity search with FAISS
5. **User-Friendly**: Web interface with drag-and-drop file upload
6. **Multi-Format Support**: Handles TXT, PDF, DOCX, and Markdown files
7. **Response Comparison**: Compare grounded vs ungrounded AI responses

## Requirements

- Python 3.8+
- OpenAI API key
- See `requirements.txt` for full dependency list
