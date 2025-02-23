"""
        pip install \
                llama-index-core==0.11.16 \
                llama-index-embeddings-clip==0.2.0 \
                llama-index-embeddings-huggingface==0.3.1 \
                llama-index-multi-modal-llms-ollama==0.3.3 \
                llama-index-readers-file==0.2.2 \
                llama-index-vector-stores-qdrant==0.3.0 \
                unstructured==0.15.13

"""
from glob import glob
from llama_index.multi_modal_llms.ollama import OllamaMultiModal
from llama_index.core.schema import ImageDocument
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.program import MultiModalLLMCompletionProgram
from pprint import pprint
from pydantic import BaseModel, Field
from time import time
from typing import List

class Entity(BaseModel):
    """Data model for describing an entity"""
    name:str = Field(description="name of the entity")
    description:str = Field(description="brief description of the entity")

class Entities(BaseModel):
    """Data model for detecting entities in the content"""
    entities:List[Entity]

prompt_template_str = """
{query_str}

Return the answer as a Pydantic object. The Pydantic schema is given below:

"""
multi_modal_llm = OllamaMultiModal(model="llava:7b-v1.5-fp16")
image_documents = [ImageDocument(image_path=p) for p in glob("../input/*.png")]

for image_document in image_documents:
    start = time()
    mm_program = MultiModalLLMCompletionProgram.from_defaults(
        verbose=True,
        multi_modal_llm=multi_modal_llm,
        image_documents=[image_document],
        prompt_template_str=prompt_template_str,
        output_parser=PydanticOutputParser(Entities),
    )
    response = mm_program(query_str="""Describe ALL entities present in image.""")
    pprint(response)
