import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "..", "my_docs")  #change to correct folder name and path
VECTORS_DIR = os.path.join(BASE_DIR, "..", "my_vectors")  #change to correct folder name and path

# AWS Bedrock settings
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))    # tokens per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "2"))  # sentences to overlap between chunks

# ChromaDB
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_docs") #change to correct folder name and path

# Retrieval relevance — cosine distance threshold (0 = identical, 2 = opposite)
# Chunks with distance above this value are discarded before answer generation
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.9"))

# Langfuse observability
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL   = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
