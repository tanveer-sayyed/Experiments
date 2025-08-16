from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from operator import add as reducer
from typing import Annotated, List, Optional, TypedDict, Union

class StateSchema(TypedDict):
    name:Annotated[list, reducer]
    welcome_msg:Annotated[list, reducer]
    add_result:Annotated[list, reducer]
    sub_result:Annotated[list, reducer]
    mul_result:Annotated[list, reducer]
    div_result:Annotated[list, reducer]
    final_result:Annotated[list, reducer]
    messages: Annotated[List[Union[HumanMessage, AIMessage, ToolMessage]], reducer]

class ParallelNodesGraphContext(TypedDict):
    a:int
    b:int
    name:Optional[str]

class ReactToolGraphContext(TypedDict):
    message:HumanMessage
