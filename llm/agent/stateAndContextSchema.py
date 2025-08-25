from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import Annotated, Dict, List, Union
from typing_extensions import TypedDict

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

class RetrieverGraphSchema(TypedDict):
    query:HumanMessage
    answer:AIMessage
    retrieved:List[Union[str,Dict[str,str]]]
