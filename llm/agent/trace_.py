from datetime import datetime
from json import dumps
from langgraph.graph._node import StateNode
from typing import Any, Awaitable, Dict, Tuple, Union

from asyncLogsAndMetrics import Duration, DocumentMetrics

class Trace:
    def __init__(self, logger:Awaitable, populateMetrics:Awaitable) -> None:
        self.type = None
        self.logger = logger
        self.metric = populateMetrics

    async def node(self, result:Union[Dict, Any]) -> None: return None

    async def tool(self, result:Union[Dict, Any]) -> None: return None

    async def retriever(self, result:Union[Dict, Any]) -> None:
        retrieved = result.get("retrieved")
        if not retrieved: return None
        await self.metric(f"Metrics.{self.type}.query", retrieved["query"])
        await self.metric(f"Metrics.{self.type}.num_docs", len(retrieved["docs"]))
        docs = list()
        for doc in retrieved["docs"]:
            doc = doc["kwargs"]
            docs.append(DocumentMetrics(
                    content=f"{doc['page_content'][:10]} ... {doc['page_content'][-10:]}",
                    metadata=doc["metadata"]
                    ))
        await self.metric(f"Metrics.{self.type}.docs", docs)

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
                if self.type == "node": await self.node(result)
                elif self.type == "tool": await self.tool(result)
                elif self.type == "retriever": await self.retriever(result)
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
