from json import dumps
from langgraph.graph import END, START
from langgraph.graph._node import StateNode
from langgraph.runtime import get_runtime
from operator import add as reducer
from typing import Annotated, Callable, Tuple, TypedDict

class StateSchema(TypedDict):
    add_result:Annotated[list, reducer]
    sub_result:Annotated[list, reducer]
    mul_result:Annotated[list, reducer]
    div_result:Annotated[list, reducer]
    final_result:Annotated[list, reducer]

class ContextSchema(TypedDict):
    a:int
    b:int

class Node():
    def __init__(self, logger:str) -> None:
        self.logger = logger

    def trace(self, func) -> Callable:
        async def inner(*args, **kwargs):
            func_name = f"{func.__qualname__}"
            await self.logger(f"[START] : {func_name}")
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                await self.logger(f"[ERROR] : {func_name}\n{str(e)}")
                raise # TODO: implement error handling here
            finally: await self.logger(f"[END]   : {func_name}\n{dumps(result, indent=2)}")
        return inner

    @staticmethod
    async def add(state:StateSchema) -> StateSchema:
        """Add two numbers together."""
        runtime = get_runtime(ContextSchema)
        state["add_result"] = [runtime.context["a"] + runtime.context["b"]]
        return state
    @staticmethod
    async def subtract(state:StateSchema) -> StateSchema:
        """Subtract the second number from the first."""
        runtime = get_runtime(ContextSchema)
        state["sub_result"] = [runtime.context["a"] - runtime.context["b"]]
        return state
    @staticmethod
    async def multiply(state:StateSchema) -> StateSchema:
        """Multiply two numbers together."""
        runtime = get_runtime(ContextSchema)
        state["mul_result"] = [runtime.context["a"] * runtime.context["b"]]
        return state
    @staticmethod
    async def divide(state:StateSchema) -> StateSchema:
        """Divide the first number by the second."""
        runtime = get_runtime(ContextSchema)
        state["div_result"] = [runtime.context["a"] / runtime.context["b"]]
        return state  # NOTE: zeroDivision unchecked <--<<---<<<-
    @staticmethod
    async def collect(state:StateSchema) -> StateSchema:
        """Format all operation results"""
        state["final_result"] = [{
                "addition": state.get("add_result"),
                "subtraction": state.get("sub_result"),
                "multiplication": state.get("mul_result"),
                "division": state.get("div_result")
            }]
        return state
    async def __call__(self, node:str) -> Tuple[str, StateNode]:
        return dict(
            node=node,
            action=self.trace(self.__getattribute__(node))
            )

class Edge:
    def __init__(self) -> None:
        self.nodes = {v.__name__:v.__name__ for (_,v) in Node.__dict__.items()\
         if isinstance(v, staticmethod)}
        self.nodes["END"] = END
        self.nodes["START"] = START
    async def __call__(self, edge:str):
        if "--" not in edge:
            start_key, end_key = edge.split(" -> ")
            return dict(
                end_key=self.nodes[end_key],
                start_key=self.nodes[start_key]
                )
        else:
            source = edge.split(" --")[0]
            edge = edge.split(" --")[1].split("-> ")
            path = edge[0]
            path_map = edge[1].split("|")
            return dict(
                source=self.nodes[source],
                path=self.__getattribute__(path),
                path_map=path_map
                )