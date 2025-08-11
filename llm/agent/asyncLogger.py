from asyncio import Lock
from aiofiles import open as aopen
from time import ctime
from typing import Callable

async def alogger(file_name:str) -> Callable:
    lock = Lock()
    print(f"{file_name} :: {id(lock)}")
    async with aopen(file_name, "w") as f: await f.write("READY\n")
    async def f(message:str):
        async with lock:
            try:
                async with aopen(file_name,"a") as f:
                    await f.write(f"{id(lock)} {ctime()} : {message}\n")
            except Exception as e: print("@LOGGING", e)
    return f
