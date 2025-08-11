from asyncio import create_task, gather

from nodesEdgesSchema import ContextSchema
from createAndCompileGraph import sessionGraph

async def _main():
    # task1
    context:ContextSchema = dict(a=10, b=9) # user's runtime context
    task1 = create_task(sessionGraph(
        session="user_1",
        context=context
        ))
    # task1
    context:ContextSchema = dict(a=20, b=8, name="Alan")  # user's runtime context
    task2 = create_task(sessionGraph(
        session="user_2",
        context=context
        ))
    # concurrent execution
    results = await gather(task1, task2)
    print(results)

async def main():
    try: await _main()
    except Exception as e:
        print(e)
        raise Exception("some error occured")

# await main()
