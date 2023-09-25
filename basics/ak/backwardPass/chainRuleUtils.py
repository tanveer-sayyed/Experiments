a_very_small_value = 0.001

def grad_of_a_during_addition(a, b):
    z = a + b; z.label = "z"
    a.data = a.data + a_very_small_value
    Z = a + b; Z.label = "z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_b_during_addition(a, b):
    z = a + b; z.label = "z"
    b.data = b.data + a_very_small_value
    Z = a + b; Z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_a_during_multiplication(a, b):
    z = a * b; z.label = "z"
    a.data = a.data + a_very_small_value
    Z = a * b; Z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_b_during_multiplication(a, b):
    z = a * b; z.label = "z"
    b.data = b.data + a_very_small_value
    Z = a * b; Z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h


def grad_of_a1(a, b, d):
    c = a + b; c.label = "c"
    z = c * d; z.label = "z"

    a.data = a.data + a_very_small_value
    c = a + b; c.label = "c"
    Z = c * d; z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_b1(a, b, d):
    c = a + b; c.label = "c"
    z = c * d; z.label = "z"

    b.data = b.data + a_very_small_value
    c = a + b; c.label = "c"
    Z = c * d; z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_c1(a, b, d):
    c = a + b; c.label = "c"
    z = c * d; z.label = "z"
    
    c = a + b; c.label = "c"
    c.data = c.data + a_very_small_value
    Z = c * d; z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_d1(a, b, d):
    c = a + b; c.label = "c"
    z = c * d; z.label = "z"

    c = a + b; c.label = "c"
    d.data = d.data + a_very_small_value
    Z = c * d; z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_a2(a, b, d):
    c = a * b; c.label = "c"
    z = c + d; z.label = "z"

    a.data = a.data + a_very_small_value
    c = a * b; c.label = "c"
    Z = c + d; z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_b2(a, b, d):
    c = a * b; c.label = "c"
    z = c + d; z.label = "z"

    b.data = b.data + a_very_small_value
    c = a * b; c.label = "c"
    Z = c + d; z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_c2(a, b, d):
    c = a * b; c.label = "c"
    z = c + d; z.label = "z"
    
    c = a * b; c.label = "c"
    c.data = c.data + a_very_small_value
    Z = c + d; z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_d2(a, b, d):
    c = a * b; c.label = "c"
    z = c + d; z.label = "z"

    c = a * b; c.label = "c"
    d.data = d.data + a_very_small_value
    Z = c + d; z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_a3(a, b, d):
    c = a * b; c.label = "c"
    z = c * d; z.label = "z"

    a.data = a.data + a_very_small_value
    c = a * b; c.label = "c"
    Z = c * d; z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_b3(a, b, d):
    c = a * b; c.label = "c"
    z = c * d; z.label = "z"

    b.data = b.data + a_very_small_value
    c = a * b; c.label = "c"
    Z = c * d; z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_c3(a, b, d):
    c = a * b; c.label = "c"
    z = c * d; z.label = "z"
    
    c = a * b; c.label = "c"
    c.data = c.data + a_very_small_value
    Z = c * d; z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_d3(a, b, d):
    c = a * b; c.label = "c"
    z = c * d; z.label = "z"

    c = a * b; c.label = "c"
    d.data = d.data + a_very_small_value
    Z = c * d; z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_a4(a, b, d):
    c = a + b; c.label = "c"
    z = c + d; z.label = "z"

    a.data = a.data + a_very_small_value
    c = a + b; c.label = "c"
    Z = c + d; z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_b4(a, b, d):
    c = a + b; c.label = "c"
    z = c + d; z.label = "z"

    b.data = b.data + a_very_small_value
    c = a + b; c.label = "c"
    Z = c + d; z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_c4(a, b, d):
    c = a + b; c.label = "c"
    z = c + d; z.label = "z"
    
    c = a + b; c.label = "c"
    c.data = c.data + a_very_small_value
    Z = c + d; z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_d4(a, b, d):
    c = a + b; c.label = "c"
    z = c + d; z.label = "z"

    c = a + b; c.label = "c"
    d.data = d.data + a_very_small_value
    Z = c + d; z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_a5(a, b):
    c = a + b; c.label = "c"
    z = c + b; z.label = "z"

    a.data = a.data + a_very_small_value
    c = a + b; c.label = "c"
    Z = c + b; z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_b5(a, b):
    c = a + b; c.label = "c"
    z = c + b; z.label = "z"

    b.data = b.data + a_very_small_value
    c = a + b; c.label = "c"
    Z = c + b; z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_c5(a, b):
    c = a + b; c.label = "c"
    z = c + b; z.label = "z"
    
    c = a + b; c.label = "c"
    c.data = c.data + a_very_small_value
    Z = c + b; z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_a6(a, b):
    c = a + b; c.label = "c"
    z = c * b; z.label = "z"

    a.data = a.data + a_very_small_value
    c = a + b; c.label = "c"
    Z = c * b; z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_b6(a, b):
    c = a + b; c.label = "c"
    z = c * b; z.label = "z"

    b.data = b.data + a_very_small_value
    c = a + b; c.label = "c"
    Z = c * b; z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h

def grad_of_c6(a, b):
    c = a + b; c.label = "c"
    z = c * b; z.label = "z"
    
    c = a + b; c.label = "c"
    c.data = c.data + a_very_small_value
    Z = c * b; z.label = "Z"
    return (Z.data - z.data)/a_very_small_value # <-- [f(z+h) - f(z)] / h
