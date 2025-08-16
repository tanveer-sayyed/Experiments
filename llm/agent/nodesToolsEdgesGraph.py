from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph._node import StateNode
from langgraph.prebuilt import ToolNode
from langgraph.runtime import get_runtime

from stateAndContextSchema import StateSchema, ParallelNodesGraphContext
from trace_ import Trace
from utils import client

class Tool(Trace):
    def __init__(self, logger, populateMetrics) -> None:
        super().__init__(logger, populateMetrics)
        self.type = "tool" # must be same as metric attribute
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
        self.type = "node" # must be same as metric attribute

    @staticmethod
    async def add(state:StateSchema) -> StateSchema:
        """Add two numbers together."""
        runtime = get_runtime(ParallelNodesGraphContext)
        state["add_result"] = [runtime.context["a"] + runtime.context["b"]]
        return state
    @staticmethod
    async def subtract(state:StateSchema) -> StateSchema:
        """Subtract the second number from the first."""
        runtime = get_runtime(ParallelNodesGraphContext)
        state["sub_result"] = [runtime.context["a"] - runtime.context["b"]]
        return state
    @staticmethod
    async def multiply(state:StateSchema) -> StateSchema:
        """Multiply two numbers together."""
        runtime = get_runtime(ParallelNodesGraphContext)
        state["mul_result"] = [runtime.context["a"] * runtime.context["b"]]
        return state
    @staticmethod
    async def divide(state:StateSchema) -> StateSchema:
        """Divide the first number by the second."""
        runtime = get_runtime(ParallelNodesGraphContext)
        state["div_result"] = [runtime.context["a"] / runtime.context["b"]]
        return state  # >>> NOTE: zeroDivision unchecked <<<
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
    @staticmethod
    async def welcome(state:StateSchema) -> StateSchema:
        """Multiply two numbers together."""
        runtime = get_runtime(ParallelNodesGraphContext)
        state["welcome_msg"] = ["HelloO! " + runtime.context["name"].upper()]
        return state
    @staticmethod
    async def tool_calling_llm(state:StateSchema):
        runtime = get_runtime(ParallelNodesGraphContext)
        messages = runtime.context["messages"]
        messages = messages[0]["content"] + state["division"]
        response = Node.client_with_tools.invoke(messages)
        return {"messages": [response]}
    @staticmethod
    async def ask_llm(state:StateSchema) -> StateSchema:
        runtime = get_runtime(ParallelNodesGraphContext)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": runtime.context["question1"]},
                ]
            }
        ]
        response = client.chat.completions.create(
                # model=MODEL,
                messages=messages,
                max_tokens=1024,
            )
        answer = response.choices[0].message.content
        state["answer1"] = answer
        return state

class Edge:
    def __init__(self) -> None:
        self.nodes = {v.__name__:v.__name__ for (_,v) in Node.__dict__.items()\
         if isinstance(v, staticmethod)}
        self.nodes["END"] = END
        self.nodes["START"] = START
    @staticmethod
    async def is_welcome_needed(state:StateSchema) -> StateSchema:
        runtime = get_runtime(ParallelNodesGraphContext)
        return "welcome" if runtime.context.get("name") else END
    @staticmethod
    async def is_calulator_tool_needed(state:StateSchema) -> StateSchema:
        return "tool_calling_llm" if state["principal"] else END
    @staticmethod
    async def is_follow_up_needed(state:StateSchema) -> StateSchema:
        runtime = get_runtime(ParallelNodesGraphContext)
        return "ask_llm" if runtime.context.get("follow_up") else END
    @staticmethod
    def should_continue(m_state:MessagesState):
        messages = m_state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "calculate_interest"
        return END
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

class CustomStateGraphBuilder(StateGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def add_tool_node(self, node:str, action:StateNode):
        self.add_node(node, ToolNode([action]))
