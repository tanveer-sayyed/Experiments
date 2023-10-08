from string import ascii_lowercase
from torch import Generator, tensor

# reproducible results
G = Generator().manual_seed(103)

data = [c for c in ascii_lowercase]
vocab_size = len(data) # unique
char2int = {c:i for (i,c) in enumerate(data)}
int2char = {i:c for (i,c) in enumerate(data)}

# out dataset is as follow:
#       a->b | b->c | ... | z->a :: no tts possible here
X = tensor([char2int[char] for char in data])
y = tensor([char2int[int2char[(k+1)%vocab_size]] for k in int2char.keys()])
