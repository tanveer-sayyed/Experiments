from asyncio import create_task, gather
from pprint import pprint

from nodesEdgesSchema import ContextSchema
from createAndCompileGraph import sessionGraph

async def _main():
    # task1
    context:ContextSchema = dict(a=10, b=7) # user's runtime context
    task1 = create_task(
        sessionGraph(
            thread_id="user_1",
            context=context
        ))
    # task2
    context:ContextSchema = dict(a=20, b=8, name="Alan")  # user's runtime context
    task2 = create_task(
        sessionGraph(
            thread_id="user_2",
            context=context
        ))
    # concurrent execution
    results = await gather(task1, task2)
    for result in results: pprint(result)

async def main():
    try: await _main()
    except Exception as e:
        print(e)
        raise Exception("some error occured")

# await main()
