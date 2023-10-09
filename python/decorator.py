"""
A decorator is a callable that returns a callable.
This is similar to packing a gift. The decorator acts as a wrapper.
"""

from time import sleep
from pytz import timezone
from datetime import datetime
from my_logger import log_this_error

from functools import wraps

def scheduler(tz, hour, minute, **kwargs):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs): 
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
                    for tries in range(10):
                        print("Executing -- {function.__qualname__}")
                        try:
                            function(**kwargs)
                            sleep(60)
                            break
                        except Exception as exc:
                            log_this_error(exc)
                            continue
                else:
                    sleep(58)
        return wrapper
    return decorator

@scheduler('Asia/Kolkata', 11, 44)
def call_my_name(name): 
    print(f"Hi {name}, You called me at a scheduled time!")

def smart_divide(func):
    def inner(a, b):
        print("I am going to divide", a, "and", b)
        if b == 0:
            print("Whoops! cannot divide")
            return
        return func(a, b)
    return inner

def divide_1(a, b):
    print(a/b)

@smart_divide
def divide_2(a, b):
    print(a/b)

if __name__ == "__main__":
    smart_divide(divide_1)(5, 0)
    divide_2(5, 0)
    divide_2(6, 3)
    call_my_name(name="Dabholkar")
    divide_1(5,0)