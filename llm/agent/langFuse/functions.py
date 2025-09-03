from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import Annotated, List, Union
from typing_extensions import TypedDict
from langgraph.graph import END

class StateSchema(TypedDict):
    a:Annotated[int, lambda x, y: y]
    b:Annotated[int, lambda x, y: y]
    name:Annotated[str, lambda x, y: y]
    welcome_msg:Annotated[str, lambda x, y: y]
    add_result:Annotated[float, lambda x, y: x or y]
    sub_result:Annotated[float, lambda x, y: x or y]
    mul_result:Annotated[float, lambda x, y: x or y]
    div_result:Annotated[float, lambda x, y: x or y]
    final_result:Annotated[dict, lambda x, y: y|x]
    messages: Annotated[List[Union[HumanMessage, AIMessage, ToolMessage]],
                        lambda x, y: x + y]

def add(state:StateSchema) -> StateSchema:
     """Add two numbers together."""
     state["add_result"] = state["a"] + state["b"]
     return state

def subtract(state:StateSchema) -> StateSchema:
     """Subtract the second number from the first."""
     state["sub_result"] = state["a"] - state["b"]
     return state

def multiply(state:StateSchema) -> StateSchema:
     """Multiply two numbers together."""
     state["mul_result"] = state["a"] * state["b"]
     return state
 
def divide(state:StateSchema) -> StateSchema:
     """Divide the first number by the second."""
     state["div_result"] = state["a"] / state["b"]
     return state  # >>> NOTE: zeroDivision unchecked <<<

def collect(state:StateSchema) -> StateSchema:
     """Format all operation results"""
     state["final_result"] = {
             "addition": state.get("add_result"),
             "subtraction": state.get("sub_result"),
             "multiplication": state.get("mul_result"),
             "division": state.get("div_result")
         }
     return state

def welcome(state:StateSchema) -> StateSchema:
     """Multiply two numbers together."""
     state["welcome_msg"] = ["HelloO! " + state["name"].upper()]
     return state

def is_welcome_needed(state:StateSchema) -> StateSchema:
    return "welcome" if state.get("name") else END
