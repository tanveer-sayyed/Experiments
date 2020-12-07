from time import sleep, time, ctime
from multiprocessing import Process

# light custom scheduler
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


## event call
while True:
    hr = int(datetime.now(timezone('Asia/Kolkata')).time().hour)
    minute = int(datetime.now(timezone('Asia/Kolkata')).time().minute)
    if (hr == 10) & (minute == 15):
        for tries in range(10):
            print("====== TRY ====== :", tries+1)
            try:
                job()
                sleep(60)
                break
            except Exception as exc:
                log_this_error(exc)
                continue
    else:
        sleep(50)
