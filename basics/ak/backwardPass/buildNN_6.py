from torch import (
    arange, float32, ones_like, randint, randn, tensor, zeros_like
    )
from torch.nn.functional import cross_entropy

from buildDataset_5 import (
    G, vocab_size, X, y
    )

ALONG_COLUMNS = 0 # helper variable
ALONG_ROWS = 1    # helper variable
BATCH_SIZE = n = 2
# our characters would be initially scattered randomly 
# within this space; this space has 2 dimensions; (2 is actually too small)
embedding_dim = 2
neurons_in_hidden_layer = 26
# creating the mapping to scatter randomly the characters onto this space
embeddings = randn(
    size=(vocab_size, embedding_dim),
    generator=G
    )

X_train = embeddings[X] # get the randomly scattered embeddings of the training data
idxs = randint(0, X_train.shape[0], (BATCH_SIZE,), generator=G) # sample a random batch
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
### manually implementing cross entropy
# postive numbers can overflow exponential, also because final probabilities are invariant to offsets
logits_normalised = logits - logits.max(ALONG_ROWS, keepdims=True).values
# expanding log signals to increase the distance between them
exponentiated = logits_normalised.exp()
exponentiated_sum_along_rows = exponentiated.sum(ALONG_ROWS, keepdims=True)
# normalise, to get probabilities
all_probabilities = exponentiated / exponentiated_sum_along_rows # <---
req_probabilities = all_probabilities[arange(n), y_train_batch]
log_probabilities = req_probabilities.log()
mean_log_probabilities = log_probabilities.mean()
# negativeNormalisedLogLikelihood
negative_mean_log_probabilities = -mean_log_probabilities
print(cross_entropy(logits, y_train_batch), negative_mean_log_probabilities)
# based on this loss the rearrangement of the scattered points will take place

# PyTorch backward pass
for p in parameters: p.grad = None
for variables in [
        logits, logits_normalised, exponentiated, exponentiated_sum_along_rows,
        all_probabilities, req_probabilities, log_probabilities,
        mean_log_probabilities, negative_mean_log_probabilities
        ]: variables.retain_grad()
negative_mean_log_probabilities.backward() # loss.backward()

print("calculating drivatives using chain rule::")
mean_log_probabilities_global_grad = tensor([-1.0])
print(f"{mean_log_probabilities_global_grad.item()=}")
log_probabilities_local_grad = ones_like(log_probabilities, dtype=float32) * 0.5
print(f"\t{log_probabilities_local_grad=}", )
log_probabilities_global_grad = log_probabilities_local_grad * mean_log_probabilities_global_grad
print(f"{log_probabilities_global_grad=}")
req_probabilities_local_grad = 1/req_probabilities
print(f"\t{req_probabilities_local_grad=}")
req_probabilities_global_grad = req_probabilities_local_grad * log_probabilities_global_grad
print(f"\t{req_probabilities_global_grad=}")
all_probabilities_global_grad = zeros_like(all_probabilities, dtype=float32)
all_probabilities_global_grad[arange(n), y_train_batch] = req_probabilities_global_grad
print(f"{all_probabilities_global_grad=}")
# global derivative must be directly multiplied during sum/broadcast
exponentiated_sum_along_rows_global_grad = (
     -exponentiated/exponentiated_sum_along_rows**2 * all_probabilities_global_grad
     ).sum(ALONG_ROWS, keepdim=True)
print(f"{exponentiated_sum_along_rows_global_grad=}")
exponentiated_local_grad_1 = exponentiated_sum_along_rows **(-1)
exponentiated_global_grad_1 = exponentiated_local_grad_1 * all_probabilities_global_grad
# in a sum operation the derivatives flow equally to all its elements of the tensor
exponentiated_global_grad_2 = ones_like(exponentiated, dtype=float32) * exponentiated_sum_along_rows_global_grad
exponentiated_global_grad = exponentiated_global_grad_1 + exponentiated_global_grad_2
logits_normalised_local = logits_normalised.exp()
logits_normalised_global = logits_normalised_local * exponentiated_global_grad
# print(f"{=}")
# print(f"{=}")
# print(f"{=}")
mean_log_probabilities.grad
log_probabilities.grad
req_probabilities.grad
all_probabilities.grad
exponentiated_sum_along_rows.grad
exponentiated.grad
logits_normalised.grad
