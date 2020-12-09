from pytz import timezone
from datetime import datetime
from time import sleep, time, ctime
from multiprocessing import Process

from my_logger import log_this_error

# light scheduler
def f():
    g = 0
    for i in range(100000000):
        g += 1
    print(g)
    
processes = []
PERIOD = 10 # in seconds
while True:
    try:
        s = time() + PERIOD
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


## event scheduler
while True:
    hr = int(datetime.now(timezone('Asia/Kolkata')).time().hour)
    minute = int(datetime.now(timezone('Asia/Kolkata')).time().minute)
    if (hr == 10) & (minute == 15):
        for tries in range(10):
            print("====== TRY ====== :", tries+1)
            try:
                f()
                sleep(60)
                break
            except Exception as exc:
                log_this_error(exc)
                continue
    else:
        sleep(50)
