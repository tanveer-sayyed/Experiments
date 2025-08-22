from langchain import hub
from langchain_core.callbacks.manager import BaseCallbackManager
from pprint import pprint
from langchain.agents import initialize_agent, AgentType

from asyncLogsAndMetrics import monitor
from callbackHandler import CustomAsyncCallbacks
from nodesToolsEdgesGraph import (
    CustomStateGraphBuilder,
    Node,
    Edge,
    Tool,
    Retriever
    )
from stateAndContextSchema import (
    RetrieverGraphSchema,
    StateSchema,
    )
from utilsCostAndClient import prettyPrintMetrics, client, convertToBaseTool

async def parallelNodesGraph(thread_id:str, context:dict):
    try:
        logger, populateMetrics, metric = await monitor(thread_id)
        metric.user.thread_id = thread_id
        node = Node(logger=logger, populateMetrics=populateMetrics)
        edge = Edge()
        callback_manager = BaseCallbackManager(
            handlers=[
                CustomAsyncCallbacks(
                    logger=logger,
                    populateMetrics=populateMetrics
                    )
                ]
            )
        builder = CustomStateGraphBuilder(
            state_schema=StateSchema
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
        #
        state = StateSchema(
            a=context["a"],
            b=context["b"],
            name=context["name"],
            welcome_msg=str(),
            add_result=float(),
            div_result=float(),
            mul_result=float(),
            sub_result=float(),
            final_result=dict()
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

async def reactToolGraph(thread_id:str, context:dict):
    try:
        logger, populateMetrics, metric = await monitor(thread_id)
        metric.user.thread_id = thread_id
        t = Tool(logger=logger, populateMetrics=populateMetrics)
        prompt = hub.pull("hwchase17/react")
        tools = [t.add, t.divide, t.multiply, t.divide]
        tools = [convertToBaseTool(func, t.trace) for func in tools]
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
        result = await agent.ainvoke({"input": context["message"].content})
        return result
    except: raise
    finally:
        try:
            pprint(metric)
            prettyPrintMetrics(metric)
        except: pass

async def retrievalGraph(thread_id:str, context:dict):
    try:
        edge = Edge()
        logger, populateMetrics, metric = await monitor(thread_id)
        metric.user.thread_id = thread_id
        callback_manager = BaseCallbackManager(
            handlers=[
                CustomAsyncCallbacks(
                    logger=logger,
                    populateMetrics=populateMetrics
                    )
                ]
            )
        retriever = Retriever(
            logger=logger,
            populateMetrics=populateMetrics,
            callback_manager=callback_manager
            )
        builder = CustomStateGraphBuilder(
            state_schema=RetrieverGraphSchema,
            )
        builder.add_node(**(await retriever("split_documents")))
        builder.add_node(**(await retriever("retrieve_docs")))
        builder.add_node(**(await retriever("rag")))
        #
        builder.add_edge(**(await edge("START -> split_documents")))
        builder.add_edge(**(await edge("split_documents -> retrieve_docs")))
        builder.add_edge(**(await edge("retrieve_docs -> rag")))
        builder.add_edge(**(await edge("rag -> END")))
        graph = builder.compile()
        state:RetrieverGraphSchema = dict(
            answer=str(),
            query=context["query"].content,
            retrieved=dict()
            )
        result = await graph.ainvoke(
                    state,
                    verbose=True,
                    config={"callbacks": callback_manager}
                    )
        return result
    except: raise
    finally:
        try:
            pprint(metric)
            prettyPrintMetrics(metric)
        except Exception as e:
            print(e)
            pass