from torch import arange, randint, randn
from torch.nn.functional import cross_entropy

from buildDataset_5 import (
    G,
    vocab_size,
    X,
    y,
    )

ACROSS_COLUMNS = 0 # helper variable
ACROSS_ROWS = 1    # helper variable
BATCH_SIZE = n = 2
# our characters would be scattered within this space of these number of dimensions
embedding_dim = 2 
neurons_in_hidden_layer = 26
# creating the mapping to scatter the characters onto this space
embeddings = randn(
    size=(vocab_size, embedding_dim),
    generator=G
    )

X_train = embeddings[X] # scattered the training data
idxs = randint(0, X_train.shape[0], (BATCH_SIZE,), generator=G) # random batch
X_train_batch, y_train_batch = X_train[idxs], y[idxs]
# a manual linear layer with its own weights and biases
weights = randn(
    size=(embedding_dim, neurons_in_hidden_layer), generator=G
    ) * (5/3) / (embedding_dim**0.5) # kiaming_normal with golden ration
bias = randn(size=(neurons_in_hidden_layer,), generator=G) * 0.1

# initialise parameters
parameters = [embeddings, weights, bias]
for p in parameters: p.requires_grad = True # else None

# running data through the nn
logits = X_train_batch @ weights + bias
# manually implement cross entropy
logits_maxes = logits.max(ACROSS_ROWS, keepdims=True) # get max to normalise
# postive numbers can overflow exponential
# also because final probabilities are invariant to offsets
logits_normalised = logits - logits_maxes.values
# expanding log signals to increase the distance between them
exponentiated = logits_normalised.exp()
# normalise, to get probabilities
probabilities = exponentiated / exponentiated.sum(ACROSS_ROWS, keepdims=True)
# negativeNormalisedLogLikelihood
loss = -probabilities[arange(n), y_train_batch].log().mean()
print(cross_entropy(logits, y_train_batch), loss)
# based on this loss the rearrangement of the scattered points will take place
