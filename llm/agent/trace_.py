from datetime import datetime
from json import dumps
from langgraph.graph._node import StateNode
from typing import Awaitable, Tuple

from metrics import Duration

class Trace:
    def __init__(self, logger:Awaitable, populateMetrics:Awaitable) -> None:
        self.type = None
        self.logger = logger
        self.metric = populateMetrics

    def trace(self, func:Awaitable) -> Awaitable:
        async def inner(*args, **kwargs):
            start_time, end_time = str(), str()
            func_name = f"{func.__qualname__}"
            await self.logger(f"[START] : {func_name}")
            await self.metric(f"Metrics.{self.type}.call_order", func_name)
            try:
                start_time = datetime.now()
                result = await func(*args, **kwargs)
                end_time = datetime.now()
                await self.metric(
                    f"Metrics.{self.type}.call_order_duration",
                    Duration(
                        start_time=str(start_time),
                        end_time=str(start_time),
                        duration=str(end_time - start_time)
                        )
                    )
                return result
            except Exception as e:
                await self.logger(f"[ERROR] : {func_name}\n{str(e)}")
                await self.metric(f"Metrics.{self.type}.error.where", func_name)
                await self.metric(f"Metrics.{self.type}.error.description", str(e))
                raise
            finally: await self.logger(f"[END]   : {func_name}\n{dumps(result, indent=2)}")
        return inner

    async def __call__(self, node:str) -> Tuple[str, StateNode]:
        return dict(
            node=node,
            action=self.trace(self.__getattribute__(node))
            )
