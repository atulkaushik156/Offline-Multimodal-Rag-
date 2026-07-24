from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
embedding_model = OllamaEmbeddings(model="nomic-embed-text",keep_alive=0)
connection_string = os.getenv("pg_vector_connection")
vector_store = PGVector(
    embeddings=embedding_model,
    collection_name="multimodal_docs_nomic",
    connection=connection_string,
    use_jsonb=True,
)