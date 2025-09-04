from langgraph.core import StateGraph
from langgraph.graph._node import StateNode
from langgraph.prebuilt import ToolNode

class CustomStateGraphBuilder(StateGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def add_tool_node(self, node:str, action:StateNode):
        self.add_node(node, ToolNode([action]))
