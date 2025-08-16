from langchain import hub
from langchain_core.callbacks.manager import AsyncCallbackManager
from langchain_core.tools import tool
# from langchain.agents import AgentExecutor, create_react_agent
from pprint import pprint
from langchain.agents import create_react_agent, initialize_agent, AgentType
from langchain.agents.agent import AgentExecutor

from asyncLogsAndMetrics import monitor
from callbackHandler import CustomAsyncCallbacks
from nodesToolsEdgesGraph import (
    CustomStateGraphBuilder,
    Node,
    Edge,
    Tool
    )
from stateAndContextSchema import (
    ParallelNodesGraphContext,
    ReactToolGraphContext,
    StateSchema,
    )
from utils import prettyPrintMetrics, client, convertToBaseTool

async def parallelNodesGraph(thread_id:str, context:ParallelNodesGraphContext):
    try:
        logger, populateMetrics, metric = await monitor(thread_id)
        metric.user.thread_id = thread_id
        node = Node(logger=logger, populateMetrics=populateMetrics)
        edge = Edge()
        callback_manager = AsyncCallbackManager(
            handlers=[
                CustomAsyncCallbacks(
                    logger=logger,
                    populateMetrics=populateMetrics
                    )
                ]
            )
        builder = CustomStateGraphBuilder(
            state_schema=StateSchema,
            context_schema=ParallelNodesGraphContext
            )
        # add node
        builder.add_node(**(await node("add")))
        builder.add_node(**(await node("divide")))
        builder.add_node(**(await node("multiply")))
        builder.add_node(**(await node("subtract")))
        builder.add_node(**(await node("collect")))
        builder.add_node(**(await node("welcome")))
        # add edge
        builder.add_edge(**(await edge("START -> add")))
        builder.add_edge(**(await edge("START -> divide")))
        builder.add_edge(**(await edge("START -> multiply")))
        builder.add_edge(**(await edge("START -> subtract")))
        builder.add_edge(**(await edge("add -> collect")))
        builder.add_edge(**(await edge("divide -> collect")))
        builder.add_edge(**(await edge("multiply -> collect")))
        builder.add_edge(**(await edge("subtract -> collect")))
        builder.add_conditional_edges(**(await edge(
            "collect --is_welcome_needed-> welcome|END"
        )))
        builder.add_edge(**(await edge("collect -> END")))
        builder.add_edge(**(await edge("welcome -> END")))
        graph = builder.compile()

        # ready to invoke
        state:StateSchema = dict(
            name=list(),
            welcome_msg=list(),
            add_result=list(),
            div_result=list(),
            mul_result=list(),
            sub_result=list(),
            final_result=list()
            )
        result = await graph.ainvoke(
                    state,
                    context=context,
                    verbose=True,
                    config={"callbacks": callback_manager}
                    )
        return result
    except: raise
    finally:
        try:
            pprint(metric)
            prettyPrintMetrics(metric)
        except: pass

async def reactToolGraph(thread_id:str, context:ReactToolGraphContext):
    try:
        logger, populateMetrics, metric = await monitor(thread_id)
        metric.user.thread_id = thread_id
        t = Tool(logger=logger, populateMetrics=populateMetrics)
        prompt = hub.pull("hwchase17/react")
        tools = [t.add, t.divide, t.multiply, t.divide]
        tools = [convertToBaseTool(func, t.trace) for func in tools]
        # -- implementation-1
        agent = initialize_agent(
            verbose=True,
            tools=tools,
            llm=client,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            callbacks=[CustomAsyncCallbacks(
                logger=logger,
                populateMetrics=populateMetrics
                )],
        )
        result = await agent.ainvoke(prompt)
        result = await agent.ainvoke({"input": context['message'].content})
        # -- implementation-2
        # llm_with_tools = client.bind_tools(tools)
        # agent = create_react_agent(llm_with_tools, tools, prompt)
        # agent_executor = AgentExecutor(
        #     agent=agent,
        #     tools=tools,
        #     verbose=True,
        #     callbacks=[CustomAsyncCallbacks(
        #         logger=logger,
        #         populateMetrics=populateMetrics
        #         )],
        #     # handle_parsing_errors=True
        # )
        # result = await agent_executor.ainvoke(
        #     {"input": f"Use the tools provided to calculate the following arithmetic expression: {context['message'].content}"})
        return result
    except: raise
    finally:
        try:
            pprint(metric)
            prettyPrintMetrics(metric)
        except: pass
