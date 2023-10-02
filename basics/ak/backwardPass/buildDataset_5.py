"""
Keep in mind:
    1. shape of tensor whose gradient needs to be calculated
    2. shape of the tensor == shape of its gradient
    3. broadcasting affects gradient
    4. operations along axes affects gradients:
        - grads might accumulate
        - grads might not accumulate
    5. variables, only meant for numerical stability, should have grad = 0.0
    6. broadcast in forward pass means a sum in backward pass
    7. sum in forward pass means a broadcast in backward pass
"""
from string import ascii_lowercase
from torch import Generator, randn, tensor

G = Generator().manual_seed(103)

data = [c for c in ascii_lowercase]
vocab_size = len(data) # unique
char2int = {c:i for (i,c) in enumerate(data)}
int2char = {i:c for (i,c) in enumerate(data)}
embedding_dim = 1
embeddings = randn(
    size=(vocab_size, embedding_dim),
    generator=G
    )

# a->b | b->c | ... | z->a :: no tts possible here
X = tensor([char2int[char] for char in data])
y = tensor(
        [char2int[int2char[(i+1)%vocab_size]] for i in range(vocab_size)]
           )
