from time import sleep, time, ctime
from multiprocessing import Process

processes = []
PERIOD = 60 # in seconds
while True:
    try:
        s = time() + PERIOD
        for x in processes:
            if not x.is_alive():
                x.join()
                processes.remove(x)
        sleep(round(s - time()))
        print(ctime(time()), len(processes))
        p = Process(target = job)
        p.start()
        processes.append(p)
    except:
        [p.terminate for p in processes]
