from langchain_core.messages import HumanMessage
from asyncio import create_task, gather, run
from pprint import pprint

from stateAndContextSchema import ParallelNodesGraphContext, ReactToolGraphContext
from createAndCompileGraph import parallelNodesGraph, reactToolGraph

async def _main1():
    # task1 with user's runtime context
    context:ParallelNodesGraphContext = dict(a=10, b=1)
    task1 = create_task(
        parallelNodesGraph(
            thread_id="parallelNodesGraph_1",
            context=context
        ))
    # task2 with user's runtime context
    context:ParallelNodesGraphContext = dict(a=20, b=8, name="Alan")
    task2 = create_task(
        parallelNodesGraph(
            thread_id="parallelNodesGraph_2",
            context=context
        ))
    # concurrent execution
    results = await gather(task1, task2)
    for result in results: pprint(result)

async def _main2():
    context:ReactToolGraphContext = dict(
        message=HumanMessage(content="""Use the tools provided to evaluate the following arithmetic expression:
96 * 73 = ?

Think step-by-step.
Do not commit any mistakes.
"""))
    task1 = create_task(
        reactToolGraph(
            thread_id="reactToolGraph",
            context=context
        ))
    results = await gather(task1)
    for result in results: pprint(result)

async def main():
    try:
        await _main1()
        # await _main2()
    except Exception as e:
        raise e                       # for debug
        # print("some error occured") # for prod

# await main()
run(main())

