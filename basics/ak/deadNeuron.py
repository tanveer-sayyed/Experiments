from matplotlib import pyplot as plt
from torch import float32, ones, rand
from torch.nn import Tanh

logits = ones(size=(20,30), dtype=float32) + rand(size=(20, 30))
logits[:,0] = 100.0
# assume the above layer as an intermediate layer in our stacks and throws
# out these logits to the following Tanh() layer, then ...
out = Tanh()(logits)
plt.imshow(out.abs() > 0.9, cmap='gray')
plt.title("The 0th (completely white) column will create a dead neouron")
plt.show()
