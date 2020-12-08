"""
A decorator is a callable that returns a callable.
This is similar to packing a gift. The decorator acts as a wrapper.
"""

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
