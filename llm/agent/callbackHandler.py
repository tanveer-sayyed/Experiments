"""
    https://python.langchain.com/api_reference/core/callbacks/langchain_core.callbacks.base.AsyncCallbackHandler.html#langchain_core.callbacks.base.AsyncCallbackHandler
"""

from json import dumps
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.documents.base import Document
from langchain_core.messages.base import BaseMessage
from langchain_core.outputs.chat_generation import ChatGenerationChunk
from langchain_core.outputs.generation import GenerationChunk
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.runnables.graph import UUID
from tenacity import RetryCallState
from typing import Any, Sequence

from utils import UUIDEncoder

class LoggingCallback(AsyncCallbackHandler):
    """Handles logging of graph execution"""

    def __init__(self, logger, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.logger = logger

    async def start_message(self, which:str) -> None:
        await self.logger(f"[START] : {which}")

    async def end_message(self, name, d):
        await self.logger(f"[END]   : {name}\n{d}")

    async def on_agent_action(
        self,
        *,
        action: AgentAction,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
        ) -> None:
        name = "on_agent_action"
        await self.start_message(name)
        d = dumps({
            "action":action,
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "tags":tags,
            }, cls=UUIDEncoder)
        await self.end_message(name, d)

    async def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
        ) -> None:
        name = "on_agent_finish"
        await self.start_message(name)
        d = dumps({
            "finish":finish,
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "tags":tags
            }, cls=UUIDEncoder)
        await self.end_message(name, d)

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any, 
        ) -> None:
        name = "on_chain_end"
        await self.start_message(name)
        d = dumps({
            "outputs":outputs,
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "tags":tags
            }, cls=UUIDEncoder)
        await self.end_message(name, d)

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any, 
        ) -> None:
        name = "on_chain_error"
        await self.start_message(name)
        d = dumps({
            "error":str(error),
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "tags":tags
            }, cls=UUIDEncoder)
        await self.end_message(name, d)

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any, 
        ) -> None:
        name = "on_chain_start"
        await self.start_message(name)
        d = dumps({
            "serialized":serialized,
            "inputs":inputs,
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "tags":tags,
            "metadata":metadata
            }, cls=UUIDEncoder)
        await self.end_message(name, d)

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any, 
        ) -> None:
        name = "on_chat_model_start"
        await self.start_message(name)
        d = dumps({
            "serialized":serialized,
            "messages":messages,
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "tags":tags,
            "metadata":metadata
            }, cls=UUIDEncoder)
        await self.end_message(name, d)

    async def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
        ) -> None:
        name = "on_custom_event"
        await self.start_message(name)
        d = dumps({
            "name":name,
            "data":data,
            "run_id":run_id,
            "tags":tags,
            "metadata":metadata
            }, cls=UUIDEncoder)
        await self.end_message(name, d)

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
        ) -> None:
        name = "on_llm_end"
        await self.start_message(name)
        d = dumps({
            "response":response,
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "tags":tags,
            }, cls=UUIDEncoder)
        await self.end_message(name, d)

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
        ) -> None:
        name = "on_llm_error"
        await self.start_message(name)
        d = dumps({
            "error":str(error),
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "tags":tags,
            }, cls=UUIDEncoder)
        await self.end_message(name, d)

    async def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
        ) -> None:
        name = "on_llm_new_token"
        await self.start_message(name)
        d = dumps({
            "token":token,
            "chunk":chunk,
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "tags":tags,
            }, cls=UUIDEncoder)
        await self.end_message(name, d)

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any
        ) -> None:
        name = "on_llm_start"
        await self.start_message(name)
        d = dumps({
            "serialized":serialized,
            "prompts":prompts,
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "metadata":metadata,
            "tags":tags,
            }, cls=UUIDEncoder)
        await self.end_message(name, d)

    async def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any, 
        ) -> None:
        name = "on_retriever_end"
        await self.start_message(name)
        d = dumps({
            "documents":documents,
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "tags":tags,
            }, cls=UUIDEncoder)
        await self.end_message(name, d)

    async def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any, 
        ) -> None:
        name = "on_retriever_error"
        await self.start_message(name)
        d = dumps({
            "error":str(error),
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "tags":tags,
            }, cls=UUIDEncoder)
        await self.end_message(name, d)

    async def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
        ) -> None:
        name = "on_retriever_start"
        await self.start_message(name)
        d = dumps({
            "serialized":serialized,
            "query":query,
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "tags":tags,
            "metadata":metadata
            }, cls=UUIDEncoder)
        await self.end_message(name, d)

    async def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
        ) -> None:
        name = "on_retry"
        await self.start_message(name)
        d = dumps({
            "retry_state":retry_state,
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            }, cls=UUIDEncoder)
        await self.end_message(name, d)

    async def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
        ) -> None:
        name = "on_text"
        await self.start_message(name)
        d = dumps({
            "text":text,
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "tags":tags
            }, cls=UUIDEncoder)
        await self.end_message(name, d)

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
        ) -> None:
        name = "on_tool_end"
        await self.start_message(name)
        d = dumps({
            "output":output,
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "tags":tags
            }, cls=UUIDEncoder)
        await self.end_message(name, d)

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
        ) -> None:
        name = "on_tool_error"
        await self.start_message(name)
        d = dumps({
            "error":str(error),
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "tags":tags
            }, cls=UUIDEncoder)
        await self.end_message(name, d)

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
        ) -> None:
        name = "on_tool_start"
        await self.start_message(name)
        d = dumps({
            "serialized":serialized,
            "input_str":input_str,
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "tags":tags,
            "metadata":metadata,
            "inputs":inputs,
            }, cls=UUIDEncoder)
        await self.end_message(name, d)
