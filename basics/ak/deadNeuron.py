from matplotlib import pyplot as plt
from torch import linspace
from torch.nn import Tanh

# assume a layer has thrown out the following logits of shape (30,20)
logits = linspace(-2, 2, 600).view(30, 20)
out = Tanh()(logits)
plt.imshow(out.abs() > 0.9, cmap='gray')
plt.title("white rows will create dead neourons")
plt.show()
