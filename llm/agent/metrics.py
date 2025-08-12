"""
    pip install opentelemetry-sdk

"""

from dataclasses import dataclass, field
from typing import Dict, List

def calculate_cost(response):
    if response.model in ['gpt-4', 'gpt-4-0314']:
        cost = (response.usage.prompt_tokens * 0.03 + response.usage.completion_tokens * 0.06) / 1000
    elif response.model in ['gpt-4-32k', 'gpt-4-32k-0314']:
        cost = (response.usage.prompt_tokens * 0.06 + response.usage.completion_tokens * 0.12) / 1000
    elif 'gpt-3.5-turbo' in response.model:
        cost = response.usage.total_tokens * 0.002 / 1000
    elif 'davinci' in response.model:
        cost = response.usage.total_tokens * 0.02 / 1000
    elif 'curie' in response.model:
        cost = response.usage.total_tokens * 0.002 / 1000
    elif 'babbage' in response.model:
        cost = response.usage.total_tokens * 0.0005 / 1000
    elif 'ada' in response.model:
        cost = response.usage.total_tokens * 0.0004 / 1000
    else:
        cost = 0
    return cost

@dataclass
class Tokens:
    cost:List[float] = field(default_factory=list)
    total_tokens:List[int] = field(default_factory=list)
    prompt_tokens:List[int] = field(default_factory=list)
    completion_tokens:List[int] = field(default_factory=list)

@dataclass
class Duration: # TODO
    start_time:str = ""
    end_time:str = ""
    duration:str = ""

@dataclass
class Error:
    where:str = ""
    description:str = ""

@dataclass
class LlmMetrics:
    model:str = ""
    llm_call_order:List[str] = field(default_factory=list)
    tokens:Tokens = field(default_factory=Tokens)
    error:Error = field(default_factory=Error)

@dataclass
class NodeMetrics:
    node_order:List[str] = field(default_factory=list)
    error:Error = field(default_factory=Error)

@dataclass
class UserInfo:
    name:str = ""
    thread_id:str = ""

@dataclass
class Metrics:
    """all leaf attributes are either str or list"""
    user:UserInfo = field(default_factory=UserInfo)
    llm:LlmMetrics = field(default_factory=LlmMetrics)
    node:NodeMetrics = field(default_factory=NodeMetrics)

metric = Metrics()
