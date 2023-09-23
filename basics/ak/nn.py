"""
    [reference]: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/micrograd_lecture_second_half_roughly.ipynb
    
a neural network from scratch, with just the forward pass
for backward pass see:
    chainRule1.ipynb
    chainRule2.ipynb
"""

from dataclasses import dataclass, field
from random import uniform
from typing import List


@dataclass
class Neuron:
    nn_in: int
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
