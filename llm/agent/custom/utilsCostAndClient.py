import asyncio
from dataclasses import dataclass
from json import JSONEncoder
from langchain_core.runnables.graph import UUID
from pandas import DataFrame
from pprint import pformat
from langchain_ollama import ChatOllama #langchain_core-0.3.74
from typing import Any, Awaitable, Type
from pydantic import BaseModel, Field, create_model
from inspect import signature
from langchain_core.tools import BaseTool

MODEL = "gpt-3.5-turbo"
client = ChatOllama(
    model="mistral:7b",
    base_url="http://localhost:11435"
    )

class UUIDEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID): return str(obj)
        return JSONEncoder.default(self, obj)

def calculateCost(model, completion_tokens, prompt_tokens):
    total_tokens = completion_tokens + prompt_tokens
    if model in ['gpt-4', 'gpt-4-0314']:
        return (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000
    elif model in ['gpt-4-32k', 'gpt-4-32k-0314']:
        return (prompt_tokens * 0.06 + completion_tokens * 0.12) / 1000
    elif 'gpt-3.5-turbo' in model:
        return total_tokens * 0.002 / 1000
    elif 'davinci' in model:
        return total_tokens * 0.02 / 1000
    elif 'curie' in model:
        return total_tokens * 0.002 / 1000
    elif 'babbage' in model:
        return total_tokens * 0.0005 / 1000
    elif 'ada' in model:
        return total_tokens * 0.0004 / 1000
    else: return 0.0

def convertSecondsToHMS(total_seconds):
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:06}"

def prettyPrintMetrics(metrics:dataclass):
    def save(df:DataFrame, path:str) -> None:
        if not df.empty: temp.to_csv(path, index=False)
    print(f"\n{'*'*100}\nMETRIC ::")
    thread_id = metrics.user.thread_id
    for attr_ in metrics.__dict__:
        print(f"\n\t{'='*12}\n\t{attr_}\n\t{'='*12}")
        attr = metrics.__dict__[attr_]
        for k in attr.__dict__:
            if k not in ["call_order","call_order_duration","tokens"]:
                if k == "docs": print(f"{k} : " + pformat(attr.__dict__[k]))
                else: print(f"{k} : {attr.__dict__[k]}")
        if "call_order" in attr.__dict__:
            print("calls : ")
            temp = DataFrame({
                "call_order":attr.call_order,
                "duration":[x.duration for x in attr.call_order_duration],
                "start_time":[x.start_time for x in attr.call_order_duration],
                "end_time":[x.end_time for x in attr.call_order_duration],
                })
            print(temp.to_markdown())
            save(temp, f"logs/{thread_id}_{attr_}_call_order.csv")
        if "query" in attr.__dict__:
            print("retrieval : ")
            temp = DataFrame({
                "query":attr.query,
                "num_docs":attr.num_docs,
                "docs":[pformat(d) for d in attr.docs],
                })
            print(temp.to_markdown())
            save(temp, f"logs/{thread_id}_{attr_}_call_order.csv")
        if "tokens" in attr.__dict__:
            print("tokens : ")
            temp = DataFrame({
                "call_order":attr.call_order,
                "model":attr.model,
                "cost":[round(x.cost,6) for x in attr.tokens],
                "total_tokens":[x.total_tokens for x in attr.tokens],
                "prompt_tokens":[x.prompt_tokens for x in attr.tokens],
                "completion_tokens":[x.completion_tokens for x in attr.tokens],
                "prompts":[f"{p:<20}" for p in attr.prompt],
                "generations":[f"{g:<20}" for g in attr.generation]
                })
            print(temp.to_markdown())
            save(temp, f"logs/{thread_id}_{attr_}_tokens.csv")
        with open(f"logs/{thread_id}_all_metric.log","w") as f:
            f.write(pformat(metrics))

def convertToBaseTool(func:Awaitable, trace:Awaitable) -> BaseTool:
    tool_name = func.__name__
    sig = signature(func)
    fields = dict()
    for param_name, param in sig.parameters.items():
        if param_name == 'self': continue
        field_info = Field(
            ... if param.default == param.empty else param.default,
            description=f"""Type: {param.annotation.__name__ 
                if param.annotation != param.empty
                    else 'Any'}"""
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
