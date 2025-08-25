from asyncio import Lock
from aiofiles import open as aopen
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Awaitable, List, Tuple

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
    model:List[str] = field(default_factory=list)
    prompt:List[str] = field(default_factory=list)
    generation:List[str] = field(default_factory=list)
    call_order:List[str] = field(default_factory=list)
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
    def __post_init__(self):
        self.call_order_duration = list()

@dataclass
class DocumentMetrics:
    content:List[str] = field(default_factory=list)
    metadata:List[str] = field(default_factory=list)

@dataclass
class RetrieverMetrics:
    query:List[str] = field(default_factory=list)
    num_docs:List[int] = field(default_factory=list)
    call_order:List[str] = field(default_factory=list)
    call_order_duration:List[Duration] = field(init=False)
    error:Error = field(default_factory=Error)
    docs:List[DocumentMetrics] = field(init=False)
    def __post_init__(self):
        self.docs = list()
        self.call_order_duration = list()

@dataclass
class ToolMetrics:
    call_order:List[str] = field(default_factory=list)
    call_order_duration:List[Duration] = field(init=False)
    error:Error = field(default_factory=Error)
    def __post_init__(self):
        self.call_order_duration = list()

@dataclass
class UserInfo:
    name:str = str()
    thread_id:str = str()
    session_id:str = str()

@dataclass
class Metrics:
    """ALL leaf attributes are either str or list"""
    user:UserInfo = field(default_factory=UserInfo)
    node:NodeMetrics = field(default_factory=NodeMetrics)
    llm:LlmMetrics = field(default_factory=LlmMetrics)
    tool:ToolMetrics = field(default_factory=ToolMetrics)
    retriever:RetrieverMetrics = field(default_factory=RetrieverMetrics)

async def monitor(file_name:str) -> Tuple[Awaitable, Awaitable, Metrics]:
    lock = Lock()
    metrics = Metrics()
    async with aopen(f"logs/{file_name}.log", "w") as f: await f.write("READY\n")
    async def logger(message:str):
        async with lock:
            try:
                async with aopen(f"logs/{file_name}.log","a") as f:
                    await f.write(f"{datetime.now()} : {message}\n")
            except Exception as e: print("@LOGGING", e)
    async def populateMetrics(position:str, value:Any):
        nonlocal metrics
        async with lock:
            try:
                position = position.split(".")
                levels = [metrics.__getattribute__(position[1])]
                for key in position[2:]:
                    levels.append(levels[-1].__getattribute__(key))
                if isinstance(levels[-1], list): levels[-1].append(value)
                else: levels[-1] = value
                i = 0 # counter
                while True:
                    try:
                        i -= 1
                        levels[i-1].__setattr__(position[i], levels[i])
                    except IndexError: break
                metrics.__setattr__(position[i], levels[i])
            except Exception as e: print("@METRIC", e)
    return logger, populateMetrics, metrics
