"""
    In the 1950s, the perceptron (Rosenblatt , 1958 , 1962 ) became the ﬁrst model that could learn the weights deﬁning the categories given examples of inputs from each category.
    
    These models would learn a set of weights w1 , . . . , wn and compute their output f(x,w) = x1w1 + · · · + xnwn
    
    Thus, perceptron is a LINEAR model. Linear models have many limitations. Most famously, they cannot learn the XOR function,
    where f ([0, 1], w) = 1 and 
          f ([1, 0], w) = 1 but,
          f ([1, 1], w) = 0 and
          f ([0, 0], w) = 0.

Here is perceptron from scratch, with just the forward pass

"""

from dataclasses import dataclass, field
from random import uniform
from typing import List


@dataclass
class Neuron:
    nn_in:int
    weight:List[float] = field(default_factory=list)
    bias:float = 0.0
    def __post_init__(self) -> None:
        self.weight = [uniform(-1,1) for _ in range(self.nn_in)]
    def __call__(self, x:float) -> float:
        return sum(
            [xi*wi for (xi, wi) in zip(x, self.weight)],
            self.bias
            )
    @property
    def parameters(self) -> List[float]:
        return self.weight + [self.bias]

@dataclass
class Layer:
    nn_in:int
    nn_out:int
    def __post_init__(self) -> None:
        self.neurons = [Neuron(self.nn_in) for i in range(self.nn_out)]
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out)==1 else out
    @property
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters]

@dataclass
class MLP:
    nn_in:int
    nn_out:tuple
    def __post_init__(self) -> None:
        sizes = (self.nn_in,) + self.nn_out
        self.layers = [
            Layer(
                nn_in=sizes[i],
                nn_out=sizes[i+1]
                ) for i in range(len(self.nn_out))
            ]
    def __call__(self, x) -> List[float]:
        for layer in self.layers: x = layer(x)
        return x
    @property
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters]

mlp = MLP(3, (4,2,1))
print(mlp.layers)