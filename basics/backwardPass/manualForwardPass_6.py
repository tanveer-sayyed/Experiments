from torch import (
    allclose, arange, randint, randn
    )
from torch.nn.functional import cross_entropy

from buildDataset_5 import (
    G, vocab_size, X, y
    )

ALONG_COLUMNS = 0 # helper variable
ALONG_ROWS = 1    # helper variable
BATCH_SIZE = n = 2
# our characters would be initially scattered randomly 
# within this space; this space has 2 dimensions
# the purpose here is to convert them into controllable vectors
embedding_dim = 2
neurons_in_hidden_layer = 26
# creating the mapping to scatter the characters randomly onto this space
embeddings = randn(
    size=(vocab_size, embedding_dim),
    generator=G
    )
# get a random batch
idxs = randint(0, X.shape[0], (BATCH_SIZE,), generator=G)
# get the randomly scattered embeddings, of the current random batch
X_train_batch, y_train_batch = X[idxs], y[idxs]

# a manual linear layer with its own weights and biases
weights = randn(
    size=(embedding_dim, neurons_in_hidden_layer), generator=G
    ) * (5/3) / (embedding_dim**0.5) # kiaming_normal with golden ration
bias = randn(size=(neurons_in_hidden_layer,), generator=G) * 0.1

# initialise parameters; ensure this step comes 
# BEFORE your variables start appearing on RHS in the forward pass
# else gradients will not flow backwards
parameters = [embeddings, weights, bias]
for p in parameters: p.requires_grad = True # else None


# Manual forward pass
X_train_batch_embeds = embeddings[X_train_batch]                            ##11.
logits = X_train_batch_embeds @ weights + bias                              ##10.
# - postive numbers can overflow exponential, hence offseting by max
# - final probabilities are invariant to offsets
# - which means the derivatives of these offsets must be (close to) zero
logits_max_along_rows = logits.max(ALONG_ROWS, keepdims=True).values        ##9.
logits_normalised = logits - logits_max_along_rows                          ##8.
# expanding log signals to increase the distance between them
exponentiated = logits_normalised.exp()                                     ##7.
exponentiated_sum_along_rows = exponentiated.sum(ALONG_ROWS, keepdims=True) ##6.
# normalise, to get probabilities
normalised_exponents = exponentiated / exponentiated_sum_along_rows         ##5.
# select only the required ones (current probabilities in our batch)
req_normalised_exponents = normalised_exponents[arange(n), y_train_batch]   ##4.
# after getting probabilities lets return to the log world
log_probabilities = req_normalised_exponents.log()                          ##3.
# likelihood is just a scalar denoting probability, here it's mean
mean_log_likelihood = log_probabilities.mean()                              ##2.
negative_mean_log_likelihood = -mean_log_likelihood                         ##1.
# backpropogation would be initiated from ##1. (i.e. a scalar) through ##11.
# based on the loss, the rearrangement of the scattered points in embeddings 
# will take place, the gradients nudge them to their final places, eventually


### PyTorch backward pass
assert allclose( # sanity check
    cross_entropy(logits, y_train_batch), 
    negative_mean_log_likelihood
    ), "something went wrong"
for p in parameters: p.grad = None
for p in parameters + [
        logits,
        X_train_batch_embeds,
        logits_max_along_rows,
        logits_normalised,
        exponentiated,
        exponentiated_sum_along_rows,
        normalised_exponents,
        req_normalised_exponents,
        log_probabilities,
        mean_log_likelihood,
        negative_mean_log_likelihood,
        ]: p.retain_grad() # for all variables that appear in the forward pass
negative_mean_log_likelihood.backward()
