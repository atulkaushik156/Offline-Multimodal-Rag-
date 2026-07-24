from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
load_dotenv()
#table summery generator

llm_model_table_summery = ChatOllama(model = "qwen2.5-coder:1.5b",
    temperature=0,
    num_ctx=2048,
    timeout=300,
    keep_alive=0)
#image summery generator 

llm_model_image_summery = ChatOllama(model="moondream",
    temperature=0,
    # 1024 is plenty for Moondream to look at an image and write a 2-sentence caption
    num_ctx=1024, 
    timeout=300,
    keep_alive=0)

#query chat model

chat_model_query = ChatOllama(model="llava-phi3",
    temperature=0,
    # 2048 to allow it to read the user query + the retrieved images
    num_ctx=2048, 
    timeout=300)