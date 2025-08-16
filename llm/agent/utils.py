import asyncio
from dataclasses import dataclass
from json import JSONEncoder
from langchain_core.runnables.graph import UUID
from langchain_openai import ChatOpenAI
from os import getenv
from pandas import DataFrame
from langchain_ollama import ChatOllama #langchain_core-0.3.74
from typing import Any, Awaitable, Type
from pydantic import BaseModel, Field, create_model
from inspect import signature
from langchain_core.tools import BaseTool

MODEL = "gpt-3.5-turbo"
# client = ChatOpenAI(api_key=getenv("OPENAI_API_KEY"))
# client = ChatOpenAI(api_key="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")
client = ChatOllama(model="mistral:7b", base_url="http://localhost:11435")


class UUIDEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID): return str(obj)
        return JSONEncoder.default(self, obj)

def calculateCost(response):
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

def prettyPrintMetrics(metrics:dataclass):
    print(f"\n{'*'*100}\nMETRIC ::")
    for attr in metrics.__dict__:
        print(f"\n{'='*len(attr)}\n{attr}\n{'='*len(attr)}")
        attr = metrics.__dict__[attr]
        for k in attr.__dict__:
            if k not in ["call_order","call_order_duration","tokens"]:
                print(f"{k} : {attr.__dict__[k]}")
        if "call_order" in attr.__dict__:
            print(DataFrame({
                "call_order":attr.call_order,
                "duration":[x.duration for x in attr.call_order_duration],
                "start_time":[x.start_time for x in attr.call_order_duration],
                "end_time":[x.end_time for x in attr.call_order_duration],
                }).to_markdown())
        if "tokens" in attr.__dict__:
            print(DataFrame({
                "call_order":attr.call_order,
                "cost":[round(x.cost,6) for x in attr.tokens],
                "total_tokens":[x.total_tokens for x in attr.tokens],
                "prompt_tokens":[x.prompt_tokens for x in attr.tokens],
                "completion_tokens":[x.completion_tokens for x in attr.tokens],
                }).to_markdown())

def convertToBaseTool(func:Awaitable, trace:Awaitable) -> BaseTool:
    tool_name = func.__name__
    sig = signature(func)
    fields = dict()
    for param_name, param in sig.parameters.items():
        if param_name == 'self': continue
        field_info = Field(
            ... if param.default == param.empty else param.default,
            description=f"Type: {param.annotation.__name__ \
                if param.annotation != param.empty \
                    else 'Any'}"
        )
        fields[param_name] = (param.annotation \
                              if param.annotation != param.empty \
                                  else Any, field_info)
    args_model = create_model(f"{tool_name.capitalize()}Args", **fields)
    class CustomTool(BaseTool):
        name: str = tool_name
        description: str = func.__doc__
        args_schema:Type[BaseModel] = args_model
        async def _arun(self, *args, **kwargs):
            return await trace(func)(*args, **kwargs)
        def _run(self, *args, **kwargs):
            """Synchronous wrapper for the async function"""
            return asyncio.get_event_loop().run_until_complete(
                self._arun(*args, **kwargs)
                )
    return CustomTool()

# func = t.add
# convertToBaseTool(func=t.add, trace:t.trace)
# c = 