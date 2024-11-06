import asyncio
from random import random
from time import sleep, time

# 1. define a coroutine
async def coroutine(i:int=0):
    await asyncio.sleep(3 + random())
    print('Hello there')
    return i*100

# 2. run a coroutine; all coroutines must be awaited
print(await coroutine())

# 3. a task runs immediately, independently and effectively in the background
task = asyncio.create_task(coroutine())
print(await task)

# 4. create many coroutines
coros = [coroutine(i) for i in range(100)]
start = time()
results = await asyncio.gather(*coros)
print(time()-start) # --> note that order is correct

# 5. create many tasks
tasks = [asyncio.create_task(coroutine(i)) for i in range(100)]
await asyncio.sleep(3)
start = time()
results = await asyncio.gather(*tasks)
print(time()-start) # --> note the difference in time from 4

# 6. pooling status in a single call
tasks = [asyncio.create_task(coroutine(i)) for i in range(10)]
done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
print(f"{pending=}")
while len(done): print(done.pop().result()) # --> note that order is random

# 7. handling results of tasks 1-by-1
tasks = [asyncio.create_task(coroutine(i)) for i in range(10)]
for task in asyncio.as_completed(tasks):
    result = await task
    print(result) # --> note that order is random
    
# 8. sharing data between coroutines using a queue
async def populate(q):
    for _ in range(10):
        put = round(random(),2)
        print(" >> :", put)
        await queue.put(put)
    await queue.put(None) # queue empty signal
queue = asyncio.Queue()
asyncio.create_task(populate(q=queue)) # task immediately runs independently in background
while True:
    get = await queue.get()
    print("<<  :", get)
    if not get: break
