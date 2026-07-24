from src.vectordb.redis import redis_client
from src.vectordb.pgvector import vector_store
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from src.llm.llm_client import chat_model_query
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()



def format_multimodal_context(inputs: dict) -> list:
    """
    Takes retrieved documents and the user query, hits Redis for images/tables, 
    and returns a standard HumanMessage list.
    """
    question = inputs["question"]
    retrieved_docs = inputs["docs"]
    
    prompt_content = [{"type": "text", "text": f"Answer the user based on the context:\nQuestion: {question}\n\nContext:"}]
    
    # Your exact same logic remains unchanged here
    for doc in retrieved_docs:
        doc_type = doc.metadata.get("type")
        
        if doc_type == "text":
            prompt_content.append({"type": "text", "text": doc.page_content})
            
        elif doc_type == "image":
            doc_id = doc.metadata.get("doc_id")
            b64_image = redis_client.get(doc_id)
            if isinstance(b64_image, bytes):
                b64_image = b64_image.decode('utf-8')
            prompt_content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
            })
            
        elif doc_type == "table":
            doc_id = doc.metadata.get("doc_id")
            raw_table = redis_client.get(doc_id)
            if isinstance(raw_table, bytes):
                raw_table = raw_table.decode('utf-8')
            prompt_content.append({"type": "text", "text": raw_table})
            
    # Return the exact list format the LLM expects
    return [HumanMessage(content=prompt_content)]


# NEW: Convert your raw vector store into a LangChain retriever component
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# NEW: Construct the formal LCEL pipeline using the | (pipe) operator
multimodal_chain = (
    # 1. Parallel setup: Fetch docs and pass the query through unchanged
    {
        "docs": retriever, 
        "question": RunnablePassthrough() 
    }
    # 2. Format: Use RunnableLambda to insert your custom Python loop into the chain
    | RunnableLambda(format_multimodal_context) 
    # 3. Execution: Send the HumanMessage to the LLM
    | chat_model_query 
    # 4. Parsing: Clean up the output string
    | parser
)