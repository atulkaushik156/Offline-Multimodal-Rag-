from fastapi import FastAPI
from src.ingestion.loader import load_file
from src.embeddings.embedder import embedded_function
from src.retrieval.retriever import multimodal_chain
app = FastAPI()

@app.put("/upload_file")
def upload_file(file_path:str):
    
    embedded_function(load_file(file_path))
    return {"status":"created embeddings successfully"}
@app.get("/chat")
def chat_rag(query:str):
    result = multimodal_chain.invoke(query)
    return {"response":result}

