from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate 
import uuid

# ADDED: Import BeautifulSoup to strip heavy HTML and prevent iGPU memory crashes
from bs4 import BeautifulSoup

from src.llm.llm_client import llm_model_image_summery, llm_model_table_summery
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from src.vectordb.pgvector import vector_store
from src.vectordb.redis import redis_client

parser = StrOutputParser()

# ADDED: Helper function to clean Unstructured HTML output before it hits the LLM
def clean_html_table(raw_html: str) -> str:
    """Strips classes, styles, and IDs from HTML to prevent token bloat."""
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    # Remove all attributes from all tags, leaving only the bare structure (<table>, <tr>, <td>)
    for tag in soup.find_all(True):
        tag.attrs = {}  
    return str(soup)


def embedded_function(raw_elements):
    text_docs_for_embedding: List[Document] = [] 
    redis_payload = {}

    for doc in raw_elements:
        doc_id = str(uuid.uuid4())
        category = doc.metadata.get("category")

        if category == "Table":
            # CHANGED: Clean the HTML table before sending it to the summary chain
            cleaned_table_content = clean_html_table(doc.page_content)

            prompt_for_table_summary = PromptTemplate(
                template="""Summarize the following {content_type} accurately. Focus on insights and trends:\n{content}""",
                input_variables=["content_type", "content"]
            )
            summary_chain = prompt_for_table_summary | llm_model_table_summery | parser
            
            # CHANGED: Pass the cleaned HTML string instead of the raw doc.page_content
            summary = summary_chain.invoke({
                "content_type": category,
                "content": cleaned_table_content
            })
            
            text_docs_for_embedding.append(Document(page_content=summary, metadata={"doc_id": doc_id, "type": "table"}))
            
            # CHANGED: Store the cleaned HTML in Redis. This prevents the Retriever from crashing the final LLM later!
            redis_payload[doc_id] = cleaned_table_content

        elif category == "Image":
            base64_image = doc.metadata.get("image_base64", None)
            mime_type = doc.metadata.get("image_mime_type")
        
            if base64_image is None:
                continue
            
            if not mime_type:
                if base64_image.startswith("/9j/"):
                    mime_type = "image/jpeg"
                elif base64_image.startswith("iVBORw0"):
                    mime_type = "image/png"
                else:
                    mime_type = "image/jpeg"
                    
            instructions = "Describe this image in high detail. Focus on text, numbers, and layout."
            
            msg = HumanMessage(content=[
                {"type": "text", "text": instructions},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
            ])
            
            # CHANGED: Because this chain doesn't use StrOutputParser, we must extract the string from the AIMessage object
            image_summary_response = llm_model_image_summery.invoke([msg])
            image_summary_text = getattr(image_summary_response, "content", str(image_summary_response))

            # CHANGED: Include the mime_type in the PGVector metadata so the retriever can reconstruct it dynamically
            text_docs_for_embedding.append(Document(
                page_content=image_summary_text, 
                metadata={"doc_id": doc_id, "type": "image", "mime_type": mime_type}
            ))
            redis_payload[doc_id] = base64_image
            
        else:
            # NORMAL TEXT
            text_docs_for_embedding.append(Document(page_content=doc.page_content, metadata={"type": "text"}))

    if text_docs_for_embedding:
        vector_store.add_documents(text_docs_for_embedding)

    if redis_payload:
        redis_client.mset(redis_payload)