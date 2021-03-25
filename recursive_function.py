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
