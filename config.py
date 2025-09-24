import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Embedding Model Configuration
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    
    # Chunking Configuration
    CHUNK_SIZE = 300  # tokens
    CHUNK_OVERLAP = 50  # tokens
    
    # Retrieval Configuration
    TOP_K_RETRIEVAL = 5
    
    # Vector Database Configuration
    VECTOR_DB_PATH = './vector_store'
    
    # Document Processing
    SUPPORTED_FORMATS = ['.txt', '.md', '.pdf']
    
    # LLM Configuration
    LLM_MODEL = 'gpt-3.5-turbo'
    MAX_TOKENS = 1000
    TEMPERATURE = 0.1