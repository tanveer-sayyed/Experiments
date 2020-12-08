"""
A decorator is a callable that returns a callable.
This is similar to packing a gift. The decorator acts as a wrapper.
"""

def make_pretty(func):
    def inner():
        print("I got decorated")
        func()
    return inner

def ordinary_1():
    print("I am ordinary 1")
    
@make_pretty
def ordinary_2():
    print("I am ordinary 2")
    
if __name__ == '__main__':
    make_pretty(ordinary_1)()
    ordinary_2()
