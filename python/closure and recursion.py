def recurring_dot_finder(s):
"""
    This function removes "." from keys and replaces
    it with "_" in a dictionary of infinite depth.
"""
    if type(s) is dict:
        for key in s.keys():
            if "." in key:
                temp = s[key]
                s.pop(key)
                key = key.replace(".","_")
                s[key] = temp
            recurring_dot_finder(s[key])
            
def fib():
    """
    Closure works 1000 times faster than recursion

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    x1 = 0
    x2 = 1
    def get_next_number():
        nonlocal x1, x2
        x3 = x1 + x2
        x1, x2 = x2, x3
        return x3
    return get_next_number
