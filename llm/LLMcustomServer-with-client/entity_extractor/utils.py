from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from typing import List

class Entity(BaseModel):
    """Data model for describing an entity"""
    name:str = Field(description="name of the entity")
    description:str = Field(description="brief description of the entity")

class Entities(BaseModel):
    """Data model for detecting entities in the content"""
    entities:List[Entity]

llm = Ollama(
    model="mistral",
    base_url="http://ollama:11434",
    request_timeout=120.0
    )
sllm = llm.as_structured_llm(Entities)

def performExtraction(prompt:str) -> BaseModel:
    response = sllm.chat([
        ChatMessage(role="system", content="You are a very helpful assistant."),
        ChatMessage(role="user", content=prompt),
        ])
    return response.message.content
