"""
    https://python.langchain.com/api_reference/core/callbacks/langchain_core.callbacks.base.AsyncCallbackHandler.html#langchain_core.callbacks.base.AsyncCallbackHandler
    https://langchain-ai.github.io/langgraph/how-tos/tool-calling/#llm-provider-tools
    https://docs.langchain.com/langgraph-platform/deploy-standalone-server
    https://docs.langchain.com/langgraph-platform/application-structure
"""

from json import dumps
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.documents.base import Document
from langchain_core.messages.base import BaseMessage
from langchain_core.outputs.chat_generation import ChatGenerationChunk
from langchain_core.outputs.generation import GenerationChunk
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.runnables.graph import UUID
from tenacity import RetryCallState
from typing import Any, Awaitable, Sequence

from utilsCostAndClient import UUIDEncoder, calculateCost, convertSecondsToHMS
from asyncLogsAndMetrics import Tokens, Duration

class CustomAsyncCallbacks(BaseCallbackHandler):
    """Handles logging of graph execution"""

    def __init__(
            self,
            logger:Awaitable,
            populateMetrics:Awaitable,
            *args, **kwargs
            ) -> None:
        super().__init__(*args, **kwargs)
        self.logger = logger
        self.metric = populateMetrics

    async def start_message(self, which:str) -> None:
        await self.logger(f"[START] : {which}")

    async def end_message(self, name, d):
        await self.logger(f"[END]   : {name}\n{d}")

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
        ) -> None:
        name = "on_agent_action"
        await self.start_message(name)
        d = dumps({
            "action":dict(action),
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "tags":tags,
            }, cls=UUIDEncoder, indent=2)
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
            "finish":dict(finish),
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "tags":tags
            }, cls=UUIDEncoder, indent=2)
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
            }, cls=UUIDEncoder, indent=2)
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
            }, cls=UUIDEncoder, indent=2)
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
        try: inputs = dict(inputs.get("input", inputs))
        except (ValueError, AttributeError): pass
        d = dumps({
            "serialized":serialized,
            "inputs":inputs,
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "tags":tags,
            "metadata":metadata
            }, cls=UUIDEncoder, indent=2)
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
            }, cls=UUIDEncoder, indent=2)
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
            }, cls=UUIDEncoder, indent=2)
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
        for generations in response.generations:
            for generations in generations:
                for generation in generations: break
        response = {
            "text":generations.text,
            "generation_info":generations.generation_info,
            'llm_output': response.llm_output,
            'run': response.run,
            'type': response.type
            }
        model = response["generation_info"]["model"]
        completion_tokens = response["generation_info"]["eval_count"]
        prompt_tokens = response["generation_info"]["prompt_eval_count"]
        cost = calculateCost(model, completion_tokens, prompt_tokens)
        duration = convertSecondsToHMS(
            response["generation_info"]["total_duration"]
            )
        await self.metric("Metrics.llm.call_order_duration",
                          Duration(
                              start_time="--",
                              end_time="--",
                              duration=duration
                            )
            )
        await self.metric("Metrics.llm.tokens",
                          Tokens(
                              cost=cost,
                              total_tokens=prompt_tokens+completion_tokens,
                              prompt_tokens=prompt_tokens,
                              completion_tokens=completion_tokens
                              )
                          )
        await self.metric("Metrics.llm.model", model)
        await self.metric("Metrics.llm.generation", response["text"])
        d = dumps({
            "response":response,
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "tags":tags,
            }, cls=UUIDEncoder, indent=2)
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
        await self.metric("Metrics.llm.error.where", name)
        await self.metric("Metrics.llm.error.description", str(error))
        await self.start_message(name)
        d = dumps({
            "error":str(error),
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "tags":tags,
            }, cls=UUIDEncoder, indent=2)
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
            "chunk":dict(chunk),
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "tags":tags,
            }, cls=UUIDEncoder, indent=2)
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
        await self.metric("Metrics.llm.call_order", "llm invoked")
        await self.metric("Metrics.llm.prompt", prompts[-1])
        await self.start_message(name)
        d = dumps({
            "serialized":serialized,
            "prompts":prompts,
            "run_id":run_id,
            "parent_run_id":parent_run_id,
            "metadata":metadata,
            "tags":tags,
            }, cls=UUIDEncoder, indent=2)
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
            }, cls=UUIDEncoder, indent=2)
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
            }, cls=UUIDEncoder, indent=2)
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
            }, cls=UUIDEncoder, indent=2)
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
            }, cls=UUIDEncoder, indent=2)
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
            }, cls=UUIDEncoder, indent=2)
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
            }, cls=UUIDEncoder, indent=2)
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
            }, cls=UUIDEncoder, indent=2)
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
            }, cls=UUIDEncoder, indent=2)
        await self.end_message(name, d)
