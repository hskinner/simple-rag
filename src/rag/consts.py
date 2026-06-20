CONTAINER_NAME = 'local-pgvector'
POSTGRES_IMAGE = 'pgvector/pgvector:pg16'
EMBEDDING_DIMENSIONS = 768
DEFAULT_TABLE = 'rag_table'

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

INSTRUCTION_PREFIX = """You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:"""