# Local Multimodal RAG (FastAPI + Ollama)

This is a completely local, offline Multimodal RAG API built to process PDFs and chat with them. 

When you pass it a PDF, it breaks the document down into text, images, and tables. It embeds and stores this data using pgvector and Redis. When you ask a question, it retrieves the relevant pieces and generates an answer using a few specialized local AI models. Everything runs on your machine via Ollama, so no API keys are needed and your data never leaves your computer.

## Tech Stack
*   **API:** FastAPI
*   **Vector DB:** pgvector (via Docker)
*   **Cache:** Redis (via Docker)
*   **LLMs:** Ollama
    *   Text Embeddings: `nomic-embed-text`
    *   Table Processing: `qwen2.5-coder:1.5b`
    *   Image Processing: `moondream`
    *   Multimodal Chat: `llava-phi3`

## Prerequisites
Make sure you have these installed before running the project:
*   Python 3.8+
*   Docker and Docker Compose
*   [Ollama](https://ollama.com/)

## How to Run It

**1. Clone the repo**
```bash
git clone https://github.com/atulkaushik156/Offline-Multimodal-Rag-.git
cd Offline-Multimodal-Rag-
```

**2. Spin up the databases**
This will start the pgvector and Redis containers in the background.
```bash
docker-compose up -d
```

**3. Pull the Ollama models**
Make sure the Ollama app is running on your machine, then pull the required models. (This might take a few minutes).
```bash
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:1.5b
ollama pull moondream
ollama pull llava-phi3
```

**4. Install dependencies**
It is highly recommended to do this inside a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

**5. Start the server**
```bash
uvicorn main:app --reload
```

## Usage

Once the FastAPI server is running, you can test everything directly in your browser using the Swagger UI at:
`http://localhost:8000/docs`

### Endpoints
*   **POST `/upload_file`**: Provide the absolute path to your PDF file here. The API will parse it, extract and chunk the text/images/tables, and save the embeddings to the databases.
*   **POST `/chat`**: Pass your question here. The API will search the databases for relevant context from your PDF and use `llava-phi3` to generate an answer.