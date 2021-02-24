from pytz import timezone
from datetime import datetime
from time import sleep, time, ctime
from multiprocessing import Process

from my_logger import log_this_error

def big_function():
    g = 0
    for i in range(100000000):
        g += 1
    print(g)

def recurring_scheduler(f, period_in_seconds = period):
    processes = []
    while True:
        try:
            s = time() + period
            for x in processes:
                if not x.is_alive():
                    x.join()
                    processes.remove(x)
            sleep(round(s - time()))
            print(ctime(time()), len(processes))
            p = Process(target = f)
            p.start()
            processes.append(p)
        except Exception as exc:
            log_this_error(exc)
            [p.terminate for p in processes]
            pass

if __name__ == "__main__":
    recurring_scheduler(big_function)