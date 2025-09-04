from langchain import hub
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks.base import BaseCallbackManager
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph._node import StateNode
from langgraph.prebuilt import ToolNode

from typing import Optional

from stateAndContextSchema import StateSchema, RetrieverGraphSchema
from trace_ import Trace
from utilsCostAndClient import client

class Tool(Trace):
    def __init__(self, logger, populateMetrics) -> None:
        super().__init__(logger, populateMetrics)
        self.type = "tool"
    @staticmethod
    async def calculate_interest(state:StateSchema) -> StateSchema:
        """calclulates the interest given the principal"""
        return state["principal"]*0.08*5
    @staticmethod
    async def add(a: float, b: float) -> float:
        """Add two numbers together.

        Args:
            a: floating-point number
            b: floating-point number

        Returns:
            floating-point number

        Example:
            >>> add(1, 2)
            >>> 3
        """
        return a + b
    @staticmethod
    async def subtract(a: float, b: float) -> float:
        """Subtract b from a.

        Args:
            a: floating-point number
            b: floating-point number

        Returns:
            floating-point number

        Example:
            >>> subtract(3, 2)
            >>> 1
        """
        return a - b
    @staticmethod
    async def multiply(a: float, b: float) -> float:
        """Multiply two numbers together.

        Args:
            a: floating-point number
            b: floating-point number

        Returns:
            floating-point number

        Example:
            >>> multiply(4, 2)
            >>> 8
        """
        return a * b
    @staticmethod
    async def divide(a: float, b: float) -> float:
        """Divide a by b.

        Args:
            a: floating-point number
            b: floating-point number

        Returns:
            floating-point number

        Raises:
            ValueError: if denominator is 0.

        Example:
            >>> divide(4, 2)
            >>> 2
        """
        if b == 0: raise ValueError("Cannot divide by zero")
        return a / b

class Node(Trace):
    client_with_tools = client.bind_tools([Tool.calculate_interest])

    def __init__(self, logger, populateMetrics) -> None:
        super().__init__(logger, populateMetrics)
        self.type = "node"

    @staticmethod
    async def add(state:StateSchema) -> StateSchema:
        """Add two numbers together."""
        state["add_result"] = state["a"] + state["b"]
        return state
    @staticmethod
    async def subtract(state:StateSchema) -> StateSchema:
        """Subtract the second number from the first."""
        state["sub_result"] = state["a"] - state["b"]
        return state
    @staticmethod
    async def multiply(state:StateSchema) -> StateSchema:
        """Multiply two numbers together."""
        state["mul_result"] = state["a"] * state["b"]
        return state
    @staticmethod
    async def divide(state:StateSchema) -> StateSchema:
        """Divide the first number by the second."""
        state["div_result"] = state["a"] / state["b"]
        return state  # >>> NOTE: zeroDivision unchecked <<<
    @staticmethod
    async def collect(state:StateSchema) -> StateSchema:
        """Format all operation results"""
        state["final_result"] = {
                "addition": state.get("add_result"),
                "subtraction": state.get("sub_result"),
                "multiplication": state.get("mul_result"),
                "division": state.get("div_result")
            }
        return state
    @staticmethod
    async def welcome(state:StateSchema) -> StateSchema:
        """Multiply two numbers together."""
        state["welcome_msg"] = ["HelloO! " + state["name"].upper()]
        return state

class CustomStateGraphBuilder(StateGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def add_tool_node(self, node:str, action:StateNode):
        self.add_node(node, ToolNode([action]))

class Retriever(Trace):
    prompt = hub.pull("rlm/rag-prompt")
    callback_manager = Optional[BaseCallbackManager]
    documents = Optional[Document]
    vector_index = Optional[VectorStoreRetriever]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=25,
        )
    embedding = OllamaEmbeddings(
        model="mistral:7b",
        base_url="http://localhost:11435" # default is 11434
        )

    @staticmethod
    def _create_mythical_creatures_db():
        return [
            Document(
                page_content="""Phoenix - A magnificent bird that cyclically regenerates by bursting into flames upon death and being reborn from the ashes. 
                Known for its healing tears and ability to be reborn, making it a symbol of renewal and resurrection.""",
                metadata={"creature_type": "Bird", "origin": "Greek", "magical_ability": "Rebirth"}
            ),
            Document(
                page_content="""Dragon - Majestic, serpentine creatures with the ability to breathe fire. 
                Often depicted as guardians of great treasures and possessing ancient wisdom.""",
                metadata={"creature_type": "Reptile", "origin": "Global", "magical_ability": "Fire Breathing"}
            ),
            Document(
                page_content="""Unicorn - A horse-like creature with a single, spiraling horn on its forehead. 
                Their horns are said to have the power to heal sickness and purify water.""",
                metadata={"creature_type": "Equine", "origin": "European", "magical_ability": "Healing"}
            ),
            Document(
                page_content="""Kitsune - Japanese fox spirits with intelligence, long life, and magical abilities. 
                They can shapeshift into human form and are known for their trickery and wisdom.""",
                metadata={"creature_type": "Canine", "origin": "Japanese", "magical_ability": "Shapeshifting"}
            ),
            Document(
                page_content="""Kraken - A legendary sea monster of enormous size, said to appear off the coasts of Norway and Greenland. 
                Known to attack ships and drag them to the ocean depths.""",
                metadata={"creature_type": "Cephalopod", "origin": "Norse", "magical_ability": "Whirlpool Creation"}
            )
        ]

    def __init__(self, logger, populateMetrics, callback_manager) -> None:
        super().__init__(logger, populateMetrics)
        self.type = "retriever"
        Retriever.llm = OllamaLLM(
                model="mistral:7b",
                base_url="http://localhost:11435",
                callbacks=callback_manager,
                verbose=True
                )
        Retriever.callback_manager = callback_manager

    @staticmethod
    async def split_documents(state:RetrieverGraphSchema) -> RetrieverGraphSchema:
        Retriever.documents = Retriever.splitter.split_documents(
            Retriever._create_mythical_creatures_db()
            )
        vector_db = FAISS.from_documents(
            Retriever.documents,
            embedding=Retriever.embedding
            )
        Retriever.vector_index = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k":3}
            )
        return state
    @staticmethod
    async def retrieve_docs(state:RetrieverGraphSchema) -> RetrieverGraphSchema:
        query = state.get("query")
        state["retrieved"] = {
            "query":query,
            "docs":[doc.to_json() for doc in Retriever.vector_index.invoke(query)]
            } # NOTE: convert Document -> JSON else serialization error
        return state
    @staticmethod
    async def rag(state:RetrieverGraphSchema) -> RetrieverGraphSchema:
        joinDocs = lambda docs: "\n".join(
            [d["kwargs"]["page_content"] for d in state["retrieved"]["docs"]]
            )
        rag_chain = (
            {
                "context":Retriever.vector_index | joinDocs,
                "question":RunnablePassthrough()
                }
            | Retriever.prompt
            | Retriever.llm
            | StrOutputParser()
            )
        state["answer"] = await rag_chain.ainvoke(
            state.get("query"),
            config={"callbacks": Retriever.callback_manager}
            )
        state["retrieved"] = dict() # reset to reduce payload
        return state    

class Edge:
    def __init__(self) -> None:
        self.nodes = {v.__name__:v.__name__ for (_,v) in Node.__dict__.items()\
         if isinstance(v, staticmethod)}
        self.nodes = self.nodes | {
            v.__name__:v.__name__ for (_,v) in Retriever.__dict__.items()\
                if isinstance(v, staticmethod)
        }
        self.nodes["END"] = END
        self.nodes["START"] = START
    @staticmethod
    async def is_welcome_needed(state:StateSchema) -> StateSchema:
        return "welcome" if state.get("name") else END
    async def __call__(self, edge:str):
        if " --" not in edge:
            start_key, end_key = edge.split(" -> ")
            return dict(
                end_key=self.nodes[end_key],
                start_key=self.nodes[start_key]
                )
        else:
            source = edge.split(" --")[0]
            edge = edge.split(" --")[1].split("-> ")
            path, path_map = edge[0], edge[1].split("|")
            return dict(
                source=self.nodes[source],
                path=self.__getattribute__(path),
                path_map=[self.nodes[p] for p in path_map]
                )
