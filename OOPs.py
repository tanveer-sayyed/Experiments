from time import sleep
from pytz import timezone
from datetime import datetime
from multiprocessing import Process

from logger import log_this_error

class scheduler():

    def __init__(self):
        self.processes = []

    def schedule(self, function, tz, hour, minute):
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
            print(f"{hour_}:{minute_} -- {function.__qualname__}")
            if (hour_ == hour) & (minute_ == minute):
                for tries in range(3): # number of attempts for any job
                    print(f"Executing -- {function.__qualname__}")
                    try:
                        function()
                        sleep(60)
                        break
                    except Exception as exc:
                        print("Error: Please check logs.")
                        log_this_error(exc)
                        continue
            else:
                sleep(50)

    def add_job(self, function, tz, hour, minute):
        process = Process(target=self.schedule,
                          args=(function, tz, hour, minute))
        self.processes.append(process)

    def run_jobs(self):
        try:
            if self.processes:
                [process.start() for process in self.processes]
        except:
            [process.terminate for process in self.processes]
            pass

if __name__ == "__main__":
    from python_file_1 import function_1
    from python_file_2 import function_2

    try:
        s = scheduler()
        s.add_job(function=function_1, tz='US/Pacific', hour=12, minute=34)
        s.add_job(function=function_1, tz='Hongkong', hour=23, minute=45)
        s.run_jobs()
    except Exception as e:
        log_this_error(e)
