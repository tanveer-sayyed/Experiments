from pytz import timezone
from datetime import datetime
from time import sleep, time, ctime
from multiprocessing import Process

from my_logger import log_this_error

class event_scheduler:

    def __init__(self):
        self.processes = []

    def _schedule(self, function, tz, hour, minute):
        """
        Parameters
        ----------
        function : callable function
            name of the function which needs to be scheduled
        tz : str
            timezone
        hour : int
            hour of the time to be scheduled
        minute : int
            minute of the time to be scheduled
        Returns
        -------
        None.
        """
        while True:
            hour_ = int(datetime.now(timezone(tz)).time().hour)
            minute_ = int(datetime.now(timezone(tz)).time().minute)
            print(f"{ctime(time())} :: {function.__qualname__} [event @{hour}:{minute} | {tz} time]")
            if (hour_ == hour) & (minute_ == minute):
                for tries in range(3): # number of attempts for any job
                    print(f"Executing -- {function.__qualname__}")
                    try:
                        function()
                        sleep(60)
                        break
                    except Exception as exc:
                        log_this_error(exc)
                        continue
            else:
                sleep(50)

    def add_event(self, function, tz, hour, minute):
        process = Process(target=self._schedule,
                          args=(function, tz, hour, minute))
        self.processes.append(process)

    def run_events(self):
        try:
            if self.processes:
                [process.start() for process in self.processes]
        except:
            [process.terminate for process in self.processes]
            pass
    
class recurring_scheduler:
    
    def __init__(self):
        self.processes = []
        self.workers = []

    def _schedule(self, function, period_in_seconds):
        """
        Parameters
        ----------
        function : callable function
            name of the function which needs to be scheduled
        period_in_seconds : int
            the time period in seconds

        Returns
        -------
        None.

        """
        while True:
            try:
                s = time() + period_in_seconds
                for x in self.workers:
                    if not x.is_alive():
                        x.join()
                        self.workers.remove(x)
                sleep(round(s - time()))
                p = Process(target = function)
                p.start()
                self.workers.append(p)
                print(f"{ctime(time())} :: {function.__qualname__} [recurring]")
            except Exception as exc:
                log_this_error(exc)
                [p.terminate for p in self.workers]
                self.workers = []
                pass

    def add_event(self, function, period_in_seconds):
        process = Process(target=self._schedule,
                          args=(function, period_in_seconds))
        self.processes.append(process)            

    def run_events(self):
        try:
            if self.processes:
                [process.start() for process in self.processes]
        except:
            [process.terminate for process in self.processes]
            pass            

def I():
    sleep(15)
    
def O():
    sleep(15)


if __name__ == "__main__":
    try:
        s = event_scheduler()
        s.add_event(function=I, tz='US/Pacific', hour=12, minute=34)
        s.add_event(function=O, tz='Hongkong', hour=23, minute=45)
        s.run_events()
        
        r = recurring_scheduler()
        r.add_event(function=I, period_in_seconds=3)
        r.add_event(function=O, period_in_seconds=15)
        r.run_events()
    except Exception as e:
        log_this_error(e)
