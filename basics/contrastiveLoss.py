from pandas import DataFrame
from torch import arange, eye, float32, tensor
from torch.nn import CrossEntropyLoss

df = DataFrame(
    [
        [4.3, 1.2, 0.05, 1.07],
        [0.18, 3.2, 0.09, 0.05],
        [0.85, 0.27, 2.2, 1.03],
        [0.23, 0.57, 0.12, 5.1]
    ]
)
data = tensor(df.values, dtype=float32)
def contrastive_loss(data):
    target = arange(data.size(0))
    return CrossEntropyLoss()(data, target)
N = data.size(0)
non_diag_mask = ~eye(N, N, dtype=bool)
for inx in range(3):
    data = tensor(df.values, dtype=float32)
    data[range(N), range(N)] += inx*0.5
    data[non_diag_mask] -= inx*0.02
    print(data)
    print(f"Loss = {contrastive_loss(data)}")