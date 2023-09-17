from random import uniform

class Neuron:

    def __init__(self, nn_in):
        self.bias = 0.0
        self.weight = [uniform(-1,1) for _ in range(nn_in)]

    def __call__(self, x):
        return sum(
            [xi*wi for (xi, wi) in zip(x, self.weight)],
            default = self.bias
            ).tanh()

    def __repr__(self):
        return f"neuron[{len(self.weight)}]"

    def parameters(self):
        return self.weight + [self.bias]

class Layer:

    def __init__(self, nn_in, nn_out):
        self.layer = [Neuron(nn_in) for i in range(nn_out)]

    def __call__(self, x):
        return [Neuron(x) for i in self.in_layer]

    def parameters(self):
        return self.weight + [self.bias]

    def __repr__(self):
        return str([n for n in self.layer])

l = Layer(3,4)