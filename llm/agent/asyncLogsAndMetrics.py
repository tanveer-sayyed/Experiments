from asyncio import Lock
from aiofiles import open as aopen
from time import ctime
from typing import Any, Awaitable, Tuple

from metrics import Metrics

async def alogger(file_name:str) -> Tuple[Awaitable, Awaitable, Metrics]:
    lock = Lock()
    metrics = Metrics()
    print(f"{file_name} :: {id(lock)}")
    async with aopen(f"{file_name}.log", "w") as f: await f.write("READY\n")
    async def logger(message:str):
        async with lock:
            try:
                async with aopen(f"{file_name}.log","a") as f:
                    await f.write(f"{id(lock)} {ctime()} : {message}\n")
            except Exception as e: print("@LOGGING", e)
    async def populateMetrics(position:str, value:Any):
        nonlocal metrics
        async with lock:
            try:
                position = position.split(".")
                levels = [metrics.__getattribute__(position[1])]
                for key in position[2:]:
                    levels.append(levels[-1].__getattribute__(key))
                if isinstance(levels[-1], list): levels[-1].append(value)
                # elif isinstance(levels[-1], dict): levels[-1][key] = value
                else: levels[-1] = value
                i = 0 # counter
                while True:
                    try:
                        i -= 1
                        levels[i-1].__setattr__(position[i], levels[i])
                    except IndexError: break
                metrics.__setattr__(position[i], levels[i])
            except Exception as e: print("@METRIC", e)
    return logger, populateMetrics, metrics
