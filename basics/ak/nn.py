from random import uniform

class neuron:

    def __init__(self):
        self.bias = 0.0
        self.weight = uniform(-1,1)

    def __call__(self, x):
        return x*self.weight + self.bias

class layer:

    def __init__(self, num_layers, size_tuple):
        self.layers = [[neuron()]*i for i in size_tuple]

    def __call__(self, x):
        return x*self.weight + self.bias

