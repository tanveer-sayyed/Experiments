from dataclasses import dataclass, field
from typing import List

@dataclass
class Tokens:
    cost:float
    total_tokens:int
    prompt_tokens:int
    completion_tokens:int

@dataclass
class Duration:
    start_time:str
    end_time:str
    duration:str

@dataclass
class Error:
    where:str = str()
    description:str = str()

@dataclass
class LlmMetrics:
    model:str = str()
    call_order:List[str] = field(default_factory=list)
    call_order_duration:List[Duration] = field(init=False)
    tokens:List[Tokens] = field(init=False)
    error:Error = field(default_factory=Error)
    def __post_init__(self):
        self.tokens = list()
        self.call_order_duration = list()

@dataclass
class NodeMetrics:
    call_order:List[str] = field(default_factory=list)
    call_order_duration:List[Duration] = field(init=False)
    error:Error = field(default_factory=Error)
    def __post_init__(self): self.call_order_duration = list()

@dataclass
class ToolMetrics:
    call_order:List[str] = field(default_factory=list)
    call_order_duration:List[Duration] = field(init=False)
    error:Error = field(default_factory=Error)
    def __post_init__(self): self.call_order_duration = list()

@dataclass
class UserInfo:
    name:str = str()
    thread_id:str = str()

@dataclass
class Metrics:
    """ALL leaf attributes are either str or list"""
    user:UserInfo = field(default_factory=UserInfo)
    llm:LlmMetrics = field(default_factory=LlmMetrics)
    node:NodeMetrics = field(default_factory=NodeMetrics)
    tool:ToolMetrics = field(default_factory=ToolMetrics)

# from pprint import pprint
# metrics = Metrics()
# pprint(Metrics())