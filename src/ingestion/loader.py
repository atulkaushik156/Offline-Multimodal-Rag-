from langchain_unstructured import UnstructuredLoader
import os
from dotenv import load_dotenv
load_dotenv()
unstructured_api_key = os.getenv("UNSTRUCTURED_API_KEY")
unstructured_url = os.getenv("UNSTRUCTURED_API_URL")

#unstructured loader
def load_file(file_path_given):
    loader = UnstructuredLoader(
        file_path=file_path_given,
        api_key=unstructured_api_key,
        url=unstructured_url,
        partition_via_api=True,          
        strategy="hi_res",               
        extract_image_block_types=["Image", "Table"],
        chunking_strategy="by_title",
        max_characters=1500,
        pdf_infer_table_structure=True,
        extract_image_block_to_payload=True, 
    )
    raw_elements = loader.load()
    return raw_elements