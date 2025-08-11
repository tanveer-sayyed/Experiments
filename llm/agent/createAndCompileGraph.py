from langgraph.graph import StateGraph
from langchain_core.callbacks.manager import CallbackManager

from asyncLogger import alogger
from nodesEdgesSchema import Node, Edge, StateSchema, ContextSchema
from callbackHandler import LoggingCallback

async def sessionGraph(session:str, context:ContextSchema):
    logger = await alogger(session)
    node = Node(logger)
    edge = Edge()
    callback_manager = CallbackManager(handlers=[LoggingCallback(logger)])
    builder = StateGraph(
        state_schema=StateSchema,
        context_schema=ContextSchema
        )
    # add nodes
    builder.add_node(**(await node("add")))
    builder.add_node(**(await node("divide")))
    builder.add_node(**(await node("multiply")))
    builder.add_node(**(await node("subtract")))
    builder.add_node(**(await node("collect")))
    builder.add_node(**(await node("welcome")))
    # add edges
    builder.add_edge(**(await edge("START -> add")))
    builder.add_edge(**(await edge("START -> divide")))
    builder.add_edge(**(await edge("START -> multiply")))
    builder.add_edge(**(await edge("START -> subtract")))
    builder.add_edge(**(await edge("add -> collect")))
    builder.add_edge(**(await edge("divide -> collect")))
    builder.add_edge(**(await edge("multiply -> collect")))
    builder.add_edge(**(await edge("subtract -> collect")))
    builder.add_conditional_edges(**(await edge(
        "collect --is_welcome_need-> welcome|END"
        )))
    builder.add_edge(**(await edge("collect -> END")))
    builder.add_edge(**(await edge("welcome -> END")))
    graph = builder.compile()

    # ready to invoke
    state:StateSchema = dict(
        add_result=list(),
        div_result=list(),
        mul_result=list(),
        sub_result=list(),
        final_result=list()
        )
    return await graph.ainvoke(
                state,
                context=context,
                verbose=True,
                config={"callbacks": callback_manager}
                )
